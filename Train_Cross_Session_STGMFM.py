import torch
import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
import swanlab

from tensorboardX import SummaryWriter
from utils.Dataload_Cross_Session import create_dataset_for_cross_validation
from utils.plot_connectome_ccg import plot_connectome_from_ccg
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
    compute_plv_adj,
)
from utils.RepeatedTrialAugmentation import RepeatedTrialAugmentation

# from utils.run_epoch_L1onST_L2_Triple_Timemixer import (
#     train_one_epoch,
#     evaluate_one_epoch,
# )
from utils.run_epoch_L1onST_L2_Triple_Timemixer_revise import (
    train_one_epoch,
    evaluate_one_epoch,
)
from STGMFM.models.STGMFM import STGMFM
from torch.utils.data import DataLoader


def start_run(args, fold_id):
    args.id = fold_id
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)
    sys.stdout = Logger(os.path.join(args.log_path, f"information-{args.id}.txt"))
    tensorboard = SummaryWriter(args.tensorboard_path)

    start_epoch = 0
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    # ---Dual device settings--- #
    # device_ab = f"cuda:{args.device_ab}" if torch.cuda.is_available() else "cpu"
    # device_c = f"cuda:{args.device_c}" if torch.cuda.is_available() else "cpu"

    train_X, train_y, test_X, test_y, person_folder = (
        create_dataset_for_cross_validation(
            base_dir=args.data_path, num_classes=args.n_class, test_person_idx=args.id
        )
    )

    swanlab.init(
        project="Main-STETM-Cross-Session-spatialGCN_False",  # ---##########修改新建##########--- #
        name=f"subject_{person_folder[args.id]}_spatialGCN_{args.spatial_GCN}_timeGCN_{args.time_GCN}",
        config=vars(args),
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
    elif args.spatial_adj_mode == "plv":
        # 用原始 train_X 计算一张“全局”PLV 图（也可用 slide_train_X 的窗口平均）
        # 建议用 topk 或 tau 控制稀疏度
        Adj = compute_plv_adj(
            train_X, tau=None, topk=3
        ).cpu()  # 例：每个通道保留4条最强边
        Adj = Adj.to(dtype=torch.float32)
    else:
        raise ValueError("adj_mode must be 'l', 'll', 'p', 'plv' or 'r'")

    model_classifier = STGMFM(
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
        # ---Dual device settings--- #
        # device=args.device_ab,
        # device_c=args.device_c,  # A/B 分支在 device_ab，Branch-C 在 device_c
        device=device,  # 原单卡
    )
    model_classifier.to(device)  # 原单卡
    # ---Dual device settings--- #
    # model_classifier.to(device_ab)  # 整体上主卡
    # model_classifier.branchC.to(device_c)  # 再保险：C 分支保持专卡

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    acc_list = []
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        ccg_w_A, ccg_w_B = train_one_epoch(
            epoch,
            train_loader,
            (slide_train_X, slide_train_y),
            model_classifier,
            # ---Dual device settings--- #
            # args.device_ab,
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
            # ---Dual device settings--- #
            # args.device_ab,
            args.device,
            criterion,
            tensorboard,
            args,
            start_time,
            rta2,
        )
        acc_list.append(avg_acc)
        scheduler.step()

        save_checkpoints = {
            "model": model_classifier.state_dict(),
            "epoch": epoch + 1,
            "acc": avg_acc,
        }
        if avg_acc > best_acc:
            best_acc = avg_acc
            save(
                save_checkpoints,
                os.path.join(args.model_path, f"model_best_{args.id}.pth.tar"),
            )
        save(
            save_checkpoints,
            os.path.join(args.model_path, f"model_newest_{args.id}.pth.tar"),
        )

    # transfrom ccg matrices to symmetric and save
    ccg_w_A = 0.5 * (ccg_w_A + ccg_w_A.T)
    np.fill_diagonal(ccg_w_A, 0.0)
    ccg_w_B = 0.5 * (ccg_w_B + ccg_w_B.T)
    np.fill_diagonal(ccg_w_B, 0.0)

    with open(
        args.spatial_adj_path + f"/spatial_node_weights_a_{args.id}.txt", "a"
    ) as f:
        f.write(str(ccg_w_A) + "\r\n")
    with open(
        args.spatial_adj_path + f"/spatial_node_weights_b_{args.id}.txt", "a"
    ) as f:
        f.write(str(ccg_w_B) + "\r\n")
    # with open(args.time_adj_path + f"/time_node_weights_{args.id}.txt", "a") as f:
    #     f.write(str(node_weights) + "\r\n")

    ch_names_in_order = [
        "F8",
        "Fp2",
        "Fpz",
        "Fp1",
        "F7",
        "F4",
        "Fz",
        "AFz",
        "F3",
        "C4",
        "CPz",
        "Cz",
        "C3",
        "T8",
        "P4",
        "Pz",
        "P3",
        "P8",
        "O1",
        "Oz",
        "O2",
        "P7",
        "T7",
    ]

    # Save category-wise connectomes
    connectome_dir = os.path.join(args.result_path, f"connectomes_subj_{args.id+1}")
    os.makedirs(connectome_dir, exist_ok=True)

    # ---plot_connectome_from_ccg Settings--- #
    # dis1 = plot_connectome_from_ccg(
    #     ccg_w_A,
    #     ch_names_in_order,
    #     save_path=os.path.join(connectome_dir, "CCG_A_branch.png"),
    #     title=f"Subject {args.id+1} - CCG (Branch-A)",
    # )

    # dis2 = plot_connectome_from_ccg(
    #     ccg_w_B,
    #     ch_names_in_order,
    #     save_path=os.path.join(connectome_dir, "CCG_B_branch.png"),
    #     title=f"Subject {args.id+1} - CCG (Branch-B)",
    # )

    plot_connectome_from_ccg(
        ccg_w_A,
        ch_names_in_order,
        save_path=os.path.join(connectome_dir, "CCG_A_branch.png"),
        title=f"Subject {args.id+1} - CCG (Branch-A)",
    )

    plot_connectome_from_ccg(
        ccg_w_B,
        ch_names_in_order,
        save_path=os.path.join(connectome_dir, "CCG_B_branch.png"),
        title=f"Subject {args.id+1} - CCG (Branch-B)",
    )

    plt.figure()
    plt.plot(acc_list, label="test_acc")
    plt.legend()
    plt.title(f"Test Acc Subject {args.id}")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.savefig(args.result_path + f"/test_acc_{str(args.id)}.png")
    df = pd.DataFrame(acc_list)
    df.to_csv(args.result_path + f"/test_acc_{str(args.id)}.csv", header=0, index=0)

    swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", type=int, default=7)  # ---###运行时修改###--- #
    parser.add_argument("-device_ab", type=int, default=0, help="主卡 (A/B)")
    parser.add_argument("-device_c", type=int, default=1, help="C-分支专用卡")
    parser.add_argument("-channel_num", type=int, default=23)
    parser.add_argument("-n_class", type=int, default=3)
    parser.add_argument("-data_path", type=str, default="/data/raw_data/eegmi_OB3000")
    parser.add_argument("-data_type", type=str, default="ob3000")
    parser.add_argument("-out_chans", type=int, default=64)
    parser.add_argument("-kernel_size", type=int, default=63)
    parser.add_argument(
        "-spatial_adj_mode",
        type=str,
        default="ll",
        choices=["l", "p", "r", "ll", "plv"],
    )
    parser.add_argument("-rta", type=bool, default=True)
    parser.add_argument("-window_length", type=int, default=125)
    parser.add_argument("-window_padding", type=int, default=125)
    parser.add_argument("-sampling_rate", type=int, default=250)
    parser.add_argument("-spatial_GCN", type=bool, default=False)
    parser.add_argument("-time_GCN", type=bool, default=True)
    parser.add_argument("-k_spatial", type=int, default=2)
    parser.add_argument("-k_time", type=int, default=2)
    parser.add_argument("-dropout", type=float, default=0.3)
    parser.add_argument("-epochs", type=int, default=1000)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=0.002)
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

    for fold_id in range(19):
        print(f"Starting Fold {fold_id+1}")
        start_run(args_, fold_id)
