# ===== Train_Cross_Session_EEGNet.py =====
import os
import time
import argparse
import random
from pathlib import Path
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Cross-Session 数据接口
from utils.Dataload_LOS_Filter import create_dataset_for_cross_validation

# EEGNet 模型
from models.eegnet import EEGNet

# 仅 logits 的 run_epoch（带多行打印 & no_grad & 五指标）
from utils.run_epoch_single_EEGNet import train_one_epoch, evaluate_one_epoch

# 可选日志
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    import swanlab
except Exception:
    swanlab = None


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(base_dir, num_classes, test_person_idx, batch_size):
    """
    返回：
      - train_loader, test_loader
      - person_folders: list[str]，用于取受试者姓名
    """
    Xtr, Ytr, Xte, Yte, person_folders = create_dataset_for_cross_validation(
        base_dir=base_dir, num_classes=num_classes, test_person_idx=test_person_idx
    )
    # 避免 dtype 警告
    Xtr = Xtr.clone().detach().float()
    Ytr = Ytr.clone().detach().long()
    Xte = Xte.clone().detach().float()
    Yte = Yte.clone().detach().long()

    tr_ds = TensorDataset(Xtr, Ytr)
    te_ds = TensorDataset(Xte, Yte)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    te_loader = DataLoader(
        te_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    return tr_loader, te_loader, person_folders


def _sanitize_name(name: str) -> str:
    """将受试者姓名安全化为文件夹/文件名友好格式。"""
    name = name.strip()
    # 只保留字母数字、连字符、下划线，其余替换为下划线
    name = re.sub(r"[^0-9A-Za-z_\-]+", "_", name)
    # 避免空字符串
    return name if name else "Unknown"


def run_one(args, sid: int):
    """按既定流程训练一个受试者（严格单设备，无 DataParallel），并使用受试者姓名命名结果。"""
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 数据与受试者姓名
    train_loader, test_loader, person_folders = build_loaders(
        args.data_path, args.n_class, sid, args.batch_size
    )
    subj_name_raw = (
        person_folders[sid] if (0 <= sid < len(person_folders)) else f"sub{sid:02d}"
    )
    subj_name = _sanitize_name(subj_name_raw)

    # 构建 EEGNet（严格单设备）
    model = EEGNet(n_classes=args.n_class, channels=args.channels, samples=args.samples)
    model = model.to(device)

    # 优化器 / 损失 / 调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 日志目录（以受试者姓名命名）
    sub_logdir = os.path.join(args.logdir, subj_name)
    os.makedirs(sub_logdir, exist_ok=True)
    writer = SummaryWriter(sub_logdir) if SummaryWriter is not None else None

    if swanlab is not None:
        try:
            swanlab.init(
                project="Main-EEGNet-LOS_Filter",
                name=f"cs_{subj_name}",
                config={
                    **vars(args),
                    "subject_id": sid,
                    "subject_name": subj_name_raw,
                    "logdir": sub_logdir,
                },
            )
        except Exception:
            pass

    print(f"\n==== Running subject: {subj_name_raw} (id {sid}) ====\n")

    # 训练循环
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # 训练阶段（多行批次打印由 run_epoch 控制）
        _ccgA, _ccgB = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            writer=writer,
            swanlab=swanlab,
            n_classes=args.n_class,
            print_freq=args.print_freq,  # << 按批次打印频率
        )

        # 测试阶段（同样可多行打印）
        val_metrics = evaluate_one_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            split="Test",
            writer=writer,
            swanlab=swanlab,
            n_classes=args.n_class,
            print_freq=args.print_freq_eval,  # << 测试阶段打印频率
        )

        scheduler.step()  ###############余弦退火###############

        # 保存最优
        acc = val_metrics["acc"]
        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                os.path.join(sub_logdir, f"best_eegnet_cs_{subj_name}.pth"),
            )

    # 收尾
    if writer is not None:
        writer.close()
    if swanlab is not None:
        try:
            swanlab.finish()
        except Exception:
            pass

    print(f"\n==> Subject {subj_name_raw} Finished. Best Test Acc: {best_acc:.2%}\n")


def main():
    parser = argparse.ArgumentParser(
        "Cross-Session Training for EEGNet (logits-only, single device)"
    )
    parser.add_argument(
        "-data_path", type=str, required=True, help="OB-3000 数据根目录"
    )
    # -id: 指定被试；若为 -1，则顺序跑完 19 个样本
    parser.add_argument(
        "-id", type=int, default=-1, help="被试索引；-1 表示依次跑完 19 个样本"
    )
    parser.add_argument("-epochs", type=int, default=200)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-wd", type=float, default=1e-4)
    parser.add_argument("-n_class", type=int, default=3)
    parser.add_argument("-channels", type=int, default=23)
    parser.add_argument("-samples", type=int, default=1125)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument(
        "-device", type=int, default=6
    )  # 单设备索引（建议配合 CUDA_VISIBLE_DEVICES）
    parser.add_argument("-logdir", type=str, default="./runs_eegnet_cs")
    # 打印频率：训练/测试阶段可分别设置（按批次数）
    parser.add_argument("-print_freq", type=int, default=1)
    parser.add_argument(
        "-print_freq_eval", type=int, default=0, help="测试阶段打印频率；0 表示只打汇总"
    )
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    if args.id == -1:
        # 顺序跑完 19 个样本（名字由数据接口返回）
        for sid in range(19):
            run_one(args, sid)
    else:
        run_one(args, args.id)


if __name__ == "__main__":
    main()
