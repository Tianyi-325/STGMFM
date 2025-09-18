# ===== Train_Cross_Session_STGENet.py =====
import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tensorboardX import SummaryWriter
import swanlab
from utils.Dataload_LOS_Filter import create_dataset_for_cross_validation
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
from utils.run_epoch_STGENet import train_one_epoch, evaluate_one_epoch
from torch.utils.data import DataLoader


def _sanitize(name: str) -> str:
    import re

    s = re.sub(r"[^0-9A-Za-z_\-]+", "_", name.strip())
    return s or "Unknown"


def run_one(args, sid: int):
    set_seed(args.seed + sid)
    args.id = sid

    # 路径 & 日志
    args = set_save_path(args.father_path, args)
    os.makedirs(args.log_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.log_path, f"information-{args.id}.txt"))
    tensorboard = SummaryWriter(args.tensorboard_path)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # ===== 数据：Cross-Session（当前被试的 s1→train, s2→test）=====
    train_X, train_y, test_X, test_y, person_folders = (
        create_dataset_for_cross_validation(
            base_dir=args.data_path, num_classes=args.n_class, test_person_idx=args.id
        )
    )
    subj_raw = (
        person_folders[args.id]
        if (0 <= args.id < len(person_folders))
        else f"sub{args.id:02d}"
    )
    subj_name = _sanitize(subj_raw)

    swanlab.init(
        project="Main-STGENET-LOS_Filter",
        name=f"subject_{subj_name}_spatialGCN_{args.spatial_GCN}_timeGCN_{args.time_GCN}",
        config={**vars(args), "subject_name": subj_raw},
    )

    channel_num = args.channel_num
    slide_window_length = args.window_length
    slide_window_stride = args.window_padding

    # ===== 滑窗切片（与原 LOSO 一致）=====
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

    # ===== 构图方式（与原 LOSO 一致）=====
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
        raise ValueError("adj_mode must be 'l', 'p', 'r' 或 'll'")

    # ===== 模型（STGENet：会返回 logits + 两种权重）=====
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

    acc_list, best_acc = [], 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        # ✅ 训练：返回两个权重（保持与你原始逻辑一致）
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
        # ✅ 评估：多指标 + no_grad
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

        # 保存
        ckpt = {
            "model": model_classifier.state_dict(),
            "epoch": epoch + 1,
            "acc": avg_acc,
        }
        if avg_acc > best_acc:
            best_acc = avg_acc
            save(ckpt, os.path.join(args.model_path, f"model_best_{subj_name}.pth.tar"))
        save(ckpt, os.path.join(args.model_path, f"model_newest_{subj_name}.pth.tar"))

    # ===== 将两类权重落盘（与原 LOSO 一致，只是带上受试者名）=====
    os.makedirs(args.spatial_adj_path, exist_ok=True)
    os.makedirs(args.time_adj_path, exist_ok=True)
    with open(
        os.path.join(args.spatial_adj_path, f"spatial_node_weights_{subj_name}.txt"),
        "a",
    ) as f:
        f.write(str(space_node_weights) + "\r\n")
    with open(
        os.path.join(args.time_adj_path, f"time_node_weights_{subj_name}.txt"), "a"
    ) as f:
        f.write(str(node_weights) + "\r\n")

    # 曲线
    os.makedirs(args.result_path, exist_ok=True)
    plt.figure()
    plt.plot(acc_list, label="test_acc")
    plt.legend()
    plt.title(f"Test Acc - {subj_raw}")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.savefig(os.path.join(args.result_path, f"test_acc_{subj_name}.png"))
    pd.DataFrame(acc_list).to_csv(
        os.path.join(args.result_path, f"test_acc_{subj_name}.csv"), header=0, index=0
    )

    swanlab.finish()


def main():
    parser = argparse.ArgumentParser("Cross-Session Training for single-branch STGENet")
    parser.add_argument("-device", type=int, default=6)
    parser.add_argument("-channel_num", type=int, default=23)
    parser.add_argument("-n_class", type=int, default=3)
    parser.add_argument("-data_path", type=str, required=True)
    parser.add_argument("-id", type=int, default=-1, help="-1 表示依次跑完 19 位被试")
    parser.add_argument("-out_chans", type=int, default=64)
    parser.add_argument("-kernel_size", type=int, default=63)
    parser.add_argument(
        "-spatial_adj_mode", type=str, default="ll", choices=["l", "p", "r", "ll"]
    )
    parser.add_argument("-window_length", type=int, default=125)
    parser.add_argument("-window_padding", type=int, default=125)
    parser.add_argument("-sampling_rate", type=int, default=250)
    parser.add_argument("-spatial_GCN", type=bool, default=True)
    parser.add_argument("-time_GCN", type=bool, default=True)
    parser.add_argument("-k_spatial", type=int, default=2)
    parser.add_argument("-k_time", type=int, default=2)
    parser.add_argument("-dropout", type=float, default=0.5)
    parser.add_argument("-epochs", type=int, default=250)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-w_decay", type=float, default=1e-2)
    parser.add_argument("-log_path", type=str, default="./log")
    parser.add_argument("-model_path", type=str, default="./model")
    parser.add_argument("-result_path", type=str, default="./results")
    parser.add_argument("-spatial_adj_path", type=str, default="./adj/spatial")
    parser.add_argument("-time_adj_path", type=str, default="./adj/time")
    parser.add_argument("-print_freq", type=int, default=50)  # 多行批次打印
    parser.add_argument("-seed", type=int, default=2024)
    parser.add_argument("-father_path", type=str, default="")
    parser.add_argument("-tensorboard_path", type=str, default="./runs")
    parser.add_argument("-rta", type=bool, default=True)
    args = parser.parse_args()

    if args.id == -1:
        for sid in range(19):
            run_one(args, sid)
    else:
        run_one(args, args.id)


if __name__ == "__main__":
    main()
