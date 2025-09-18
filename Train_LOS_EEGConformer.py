# ===== Train_Cross_Session_EEGConformer.py =====
import os
import re
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Cross-Session 数据接口（同人 session1→train, session2→test）
from utils.Dataload_LOS_Filter import create_dataset_for_cross_validation

# EEGConformer 模型
from models.eegconformer import EEGConformer

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


def _center_crop_time(X: torch.Tensor, target_T: int) -> torch.Tensor:
    """对 (N,C,T) 的张量在时间维做中心裁剪到 target_T。"""
    T = X.shape[-1]
    if T == target_T:
        return X
    if T < target_T:
        raise ValueError(
            f"Time length {T} < target_T {target_T}; 请调整数据/参数或修改模型分类头。"
        )
    start = (T - target_T) // 2
    end = start + target_T
    return X[..., start:end]


def build_loaders(base_dir, num_classes, test_person_idx, batch_size, target_T):
    """
    返回：
      - train_loader, test_loader
      - person_folders: list[str]，用于取受试者姓名
    并在此处对时间维做中心裁剪到 target_T（默认为 1000）。
    """
    Xtr, Ytr, Xte, Yte, person_folders = create_dataset_for_cross_validation(
        base_dir=base_dir, num_classes=num_classes, test_person_idx=test_person_idx
    )
    # 类型规范
    Xtr = Xtr.clone().detach().float()
    Ytr = Ytr.clone().detach().long()
    Xte = Xte.clone().detach().float()
    Yte = Yte.clone().detach().long()

    # 关键：裁剪到 1000（或 args.samples_target）
    Xtr = _center_crop_time(Xtr, target_T)
    Xte = _center_crop_time(Xte, target_T)

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
    # 建议让每位受试者 seed 略作偏移（基于 args.seed），避免曲线“视觉上一样”
    set_seed(args.seed + sid)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 数据 + 受试者姓名（并裁剪到 target_T）
    train_loader, test_loader, person_folders = build_loaders(
        args.data_path, args.n_class, sid, args.batch_size, args.samples_target
    )
    subj_raw = (
        person_folders[sid] if (0 <= sid < len(person_folders)) else f"sub{sid:02d}"
    )
    subj_name = _sanitize_name(subj_raw)

    # ===== 构建 EEGConformer（仅 logits 输出）=====
    # 注意：embedding_size 保持 40，以匹配分类头 2440 (=61*40)
    model = EEGConformer(
        in_channels=args.channels,
        embedding_size=args.embedding_size,
        depth=args.depth,
        n_classes=args.n_class,
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
                project="Main-EEGConformer-LOS_Filter",
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
                os.path.join(sub_logdir, f"best_eegconformer_cs_{subj_name}.pth"),
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
        "Cross-Session Training for EEGConformer (logits-only, single device)"
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
        "-device", type=int, default=4
    )  # 单设备索引（建议配合 CUDA_VISIBLE_DEVICES）
    parser.add_argument("-logdir", type=str, default="./runs_eegconformer_cs")
    # —— 关键：将时间长度裁剪到 1000（使 tokens=61，匹配分类头的 2440 输入）——
    parser.add_argument("-samples_target", type=int, default=1000)
    # —— EEGConformer 结构超参 ——（embedding_size=40 对齐分类头）
    parser.add_argument("-embedding_size", type=int, default=40)
    parser.add_argument("-depth", type=int, default=6)
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
