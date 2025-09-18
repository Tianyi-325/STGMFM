import torch
import os
import sys
import argparse
import time
import numpy as np
from tensorboardX import SummaryWriter
from utils.Dataload_LOSO import create_dataset_for_cross_validation
from utils.tools import (
    set_seed,
    set_save_path,
    Logger,
    sliding_window_eeg,
    load_adj,
    build_tranforms,
    build_tranforms2,
    EEGDataSet,
    save,
)
from models.STGENet import STGENET
from utils.RepeatedTrialAugmentation import RepeatedTrialAugmentation
from utils.run_epoch import train_one_epoch, evaluate_one_epoch
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import swanlab


def start_run(args):
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)
    sys.stdout = Logger(os.path.join(args.log_path, f"information-{args.id}.txt"))
    tensorboard = SummaryWriter(args.tensorboard_path)
    swanlab.init(
        project="EEG-STGENET",
        name=f"subject_{args.id}_spatialGCN_{args.spatial_GCN}_timeGCN_{args.time_GCN}",
    )

    start_epoch = 0
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    if args.data_type == "ob3000":
        train_X, train_y, test_X, test_y, person_folder = (
            create_dataset_for_cross_validation(
                base_dir=args.data_path,
                num_classes=args.n_class,
                test_person_idx=args.id,
            )
        )
    else:
        raise ValueError(
            "Unsupported data_type: only 'ob3000' is supported in this version."
        )

    channel_num = args.channel_num

    slide_window_length = args.window_length
    slide_window_stride = args.window_padding

    slide_train_X, slide_train_y = sliding_window_eeg(
        train_X, train_y, slide_window_length, slide_window_stride
    )
    slide_test_X, slide_test_y = sliding_window_eeg(
        test_X, test_y, slide_window_length, slide_window_stride
    )

    slide_train_X = torch.tensor(slide_train_X, dtype=torch.float32)
    slide_test_X = torch.tensor(slide_test_X, dtype=torch.float32)
    slide_train_y = torch.tensor(slide_train_y, dtype=torch.int64)
    slide_test_y = torch.tensor(slide_test_y, dtype=torch.int64)

    slide_window_num = slide_train_X.shape[0]

    if args.spatial_adj_mode == "l":
        Adj = torch.tensor(load_adj("bciciv2a"), dtype=torch.float32)
    elif args.spatial_adj_mode == "ll":
        Adj = torch.tensor(load_adj("oy"), dtype=torch.float32)
    elif args.spatial_adj_mode == "p":
        temp = train_X
        train_data = temp.permute(0, 2, 1).contiguous().reshape(-1, channel_num)
        Adj = torch.tensor(
            np.corrcoef(train_data.numpy().T, ddof=1), dtype=torch.float32
        )
    elif args.spatial_adj_mode == "r":
        Adj = torch.randn(channel_num, channel_num)
    else:
        raise ValueError("adj_mode must be 'l', 'p', or 'r'")

    model_classifier = STGENET(
        Adj=Adj,
        in_chans=channel_num,
        n_classes=args.n_class,
        time_window_num=slide_window_num,
        spatial_GCN=args.spatial_GCN,
        time_GCN=args.time_GCN,
        k_spatial=args.k_spatial,
        k_time=args.k_time,
        dropout=args.dropout,
        input_time_length=slide_window_length,
        out_chans=args.out_chans,
        kernel_size=args.kernel_size,
        slide_window=slide_window_num,
        sampling_rate=args.sampling_rate,
        device=args.device,
    )

    optimizer = torch.optim.AdamW(
        model_classifier.parameters(), lr=args.lr, weight_decay=args.w_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0

    train_loader = DataLoader(
        EEGDataSet(slide_train_X, slide_train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    test_loader = DataLoader(
        EEGDataSet(slide_test_X, slide_test_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    transform = build_tranforms()
    transform2 = build_tranforms2()
    rta = RepeatedTrialAugmentation(transform, m=5)
    rta2 = RepeatedTrialAugmentation(transform2, m=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=2**-12
    )

    acc_list = []
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        node_weights, space_node_weights = train_one_epoch(
            epoch,
            train_loader,
            (slide_train_X, slide_train_y),
            model_classifier,
            args.device,
            optimizer,
            criterion,
            tensorboard,
            start_time,
            args,
            rta,
        )
        avg_acc, avg_loss = evaluate_one_epoch(
            epoch,
            test_loader,
            (slide_test_X, slide_test_y),
            model_classifier,
            args.device,
            criterion,
            tensorboard,
            args,
            start_time,
            rta2,
        )
        acc_list.append(avg_acc)

        save_checkpoints = {
            "model": model_classifier.state_dict(),
            "epoch": epoch + 1,
            "acc": avg_acc,
        }
        if avg_acc > best_acc:
            best_acc = avg_acc
            save(save_checkpoints, os.path.join(args.model_path, "model_best.pth.tar"))
        save(save_checkpoints, os.path.join(args.model_path, "model_newest.pth.tar"))

    with open(args.spatial_adj_path + "/spatial_node_weights.txt", "a") as f:
        f.write(str(space_node_weights) + "\r\n")
    with open(args.time_adj_path + "/time_node_weights.txt", "a") as f:
        f.write(str(node_weights) + "\r\n")

    plt.figure()
    plt.plot(acc_list, label="test_acc")
    plt.legend()
    plt.title("Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.savefig(args.result_path + f"/test_acc_{str(args.id)}.png")
    df = pd.DataFrame(acc_list)
    df.to_csv(args.result_path + f"/test_acc_{str(args.id)}.csv", header=0, index=0)

    swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", type=int, default=0)
    parser.add_argument("-channel_num", type=int, default=23)
    parser.add_argument("-n_class", type=int, default=3)
    parser.add_argument("-data_path", type=str, required=True)
    parser.add_argument("-id", type=int, default=0)
    parser.add_argument("-data_type", type=str, default="ob3000")
    parser.add_argument("-out_chans", type=int, default=64)
    parser.add_argument("-kernel_size", type=int, default=63)
    parser.add_argument(
        "-spatial_adj_mode", type=str, default="p", choices=["l", "p", "r", "ll"]
    )
    parser.add_argument("-rta", type=bool, default=True)
    parser.add_argument("-window_length", type=int, default=125)
    parser.add_argument("-window_padding", type=int, default=100)
    parser.add_argument("-sampling_rate", type=int, default=250)
    parser.add_argument("-spatial_GCN", type=bool, default=True)
    parser.add_argument("-time_GCN", type=bool, default=True)
    parser.add_argument("-k_spatial", type=int, default=2)
    parser.add_argument("-k_time", type=int, default=2)
    parser.add_argument("-dropout", type=float, default=0.5)
    parser.add_argument("-epochs", type=int, default=200)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-w_decay", type=float, default=0.01)
    parser.add_argument("-log_path", type=str, default="./log")
    parser.add_argument("-model_path", type=str, default="./model")
    parser.add_argument("-result_path", type=str, default="./results")
    parser.add_argument("-spatial_adj_path", type=str, default="./adj/spatial")
    parser.add_argument("-time_adj_path", type=str, default="./adj/time")
    parser.add_argument("-print_freq", type=int, default=1)
    parser.add_argument("-seed", type=int, default=2024)
    parser.add_argument("-father_path", type=str, default="")
    parser.add_argument("-tensorboard_path", type=str, default="./runs")

    args_ = parser.parse_args()
    start_run(args_)
