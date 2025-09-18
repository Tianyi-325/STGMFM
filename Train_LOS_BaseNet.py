# ===== Train_Cross_Session_BaseNet.py =====
import os
import re
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Cross-Session 数据接口（同人 session1→train, session2→test）
from utils.Dataload_LOS_Filter import create_dataset_for_cross_validation

# BaseNet 模型
from models.basenet import BaseNet

# “logits-only” 的训练/评估循环（多行打印 + no_grad + 五指标）
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


def _sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^0-9A-Za-z_\-]+", "_", name)
    return name if name else "Unknown"


def build_loaders(base_dir, num_classes, test_person_idx, batch_size):
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


def run_one(args, sid: int):
    """训练/评估一个受试者（单设备；日志/权重命名使用受试者姓名）。"""
    # 建议让每位受试者的 seed 略作偏移，避免曲线视觉上完全一致
    set_seed(args.seed + sid)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 数据 + 受试者姓名
    train_loader, test_loader, person_folders = build_loaders(
        args.data_path, args.n_class, sid, args.batch_size
    )
    subj_raw = (
        person_folders[sid] if (0 <= sid < len(person_folders)) else f"sub{sid:02d}"
    )
    subj_name = _sanitize_name(subj_raw)

    # 动态获取 T（时间长度），对齐 BaseNet 的 samples
    T = train_loader.dataset.tensors[0].shape[-1]

    # ===== 构建 BaseNet（仅 logits 输出）=====
    model = BaseNet(
        chans=args.channels,
        samples=T,
        num_classes=args.n_class,
        F1=args.F1,
        time_kernel1=args.time_kernel1,
        pool_kernel1=args.pool_kernel1,
        pool_stride=args.pool_stride,
        F2=args.F2,
        time_kernel2=args.time_kernel2,
        pool_kernel2=args.pool_kernel2,
    )
    model = model.to(device)  # 单设备

    # 优化器 / 损失 / 调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 日志目录（受试者姓名）
    sub_logdir = os.path.join(args.logdir, subj_name)
    os.makedirs(sub_logdir, exist_ok=True)
    writer = SummaryWriter(sub_logdir) if SummaryWriter is not None else None

    if swanlab is not None:
        try:
            swanlab.init(
                project="Main-BaseNet-LOS_Filter",
                name=f"cs_{subj_name}",
                config={
                    **vars(args),
                    "subject_id": sid,
                    "subject_name": subj_raw,
                    "logdir": sub_logdir,
                },
            )
        except Exception:
            pass

    print(f"\n==== Running subject: {subj_raw} (id {sid}) ====\n")

    # 训练循环
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # 训练（多行批次打印）
        _ = train_one_epoch(
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
            print_freq=args.print_freq,
        )

        # 测试（同样可多行打印）
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
            print_freq=args.print_freq_eval,
        )

        scheduler.step()

        # 保存最优（以受试者姓名命名）
        acc = val_metrics["acc"]
        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                os.path.join(sub_logdir, f"best_basenet_cs_{subj_name}.pth"),
            )

    # 收尾
    if writer is not None:
        writer.close()
    if swanlab is not None:
        try:
            swanlab.finish()
        except Exception:
            pass

    print(f"\n==> Subject {subj_raw} Finished. Best Test Acc: {best_acc:.2%}\n")


def main():
    parser = argparse.ArgumentParser(
        "Cross-Session Training for BaseNet (logits-only, single device)"
    )
    parser.add_argument(
        "-data_path", type=str, required=True, help="OB-3000 数据根目录"
    )
    parser.add_argument(
        "-id", type=int, default=-1, help="被试索引；-1 表示依次跑完 19 个样本"
    )
    parser.add_argument("-epochs", type=int, default=200)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-wd", type=float, default=1e-4)
    parser.add_argument("-n_class", type=int, default=3)
    parser.add_argument("-channels", type=int, default=23)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument(
        "-device", type=int, default=3
    )  # 单设备索引（建议配合 CUDA_VISIBLE_DEVICES）
    parser.add_argument("-logdir", type=str, default="./runs_basenet_cs")
    # —— BaseNet 结构超参（默认与源码一致，可按需调整）——
    parser.add_argument("-F1", type=int, default=40)
    parser.add_argument("-time_kernel1", type=int, default=25)
    parser.add_argument("-pool_kernel1", type=int, default=75)
    parser.add_argument("-pool_stride", type=int, default=15)
    parser.add_argument("-F2", type=int, default=16)
    parser.add_argument("-time_kernel2", type=int, default=15)
    parser.add_argument("-pool_kernel2", type=int, default=8)
    # 打印频率：训练/测试阶段可分别设置（按批次数）
    parser.add_argument("-print_freq", type=int, default=50)
    parser.add_argument(
        "-print_freq_eval", type=int, default=0, help="测试阶段打印频率；0 表示只打汇总"
    )
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    if args.id == -1:
        for sid in range(19):
            run_one(args, sid)
    else:
        run_one(args, args.id)


if __name__ == "__main__":
    main()
