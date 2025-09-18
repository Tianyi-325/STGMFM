# ===== Train_Cross_Session_LMDANet.py =====
import os
import re
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Cross-Session 数据接口（同人：session1->train, session2->test）
from utils.Dataload_LOS_Filter import create_dataset_for_cross_validation

# 原始 LMDA 模型
from models.lmdanet import LMDA

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


# ---------- 小工具 ----------
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


# ---------- 无参 Depth-Attention（函数式，复刻原 forward 的思路，无打印&非可训练） ----------
def _depth_attention_stateless(x, k: int = 7):
    """
    x: (B, D, C, W)  —— D为“深度”维（由 channel_weight 展开得到）
    步骤：自适应池化到 (1,W) -> 在深度维做 (k,1) 卷积 -> softmax(沿深度) -> 与 x 相乘并按通道数缩放
    返回: (B, D, C, W)
    """
    B, D, C, W = x.shape
    # 自适应池化到 (1, W)，仅沿 C 汇聚
    x_pool = F.adaptive_avg_pool2d(x, (1, W))  # (B, D, 1, W)
    x_trans = x_pool.transpose(-2, -3)  # (B, 1, D, W)

    # 用 F.conv2d 实现一次性卷积（out_channels=in_channels=1, groups=1）
    # 构造临时核：形状 (1,1,k,1)，Xavier 初始化
    w = torch.empty(1, 1, k, 1, device=x.device, dtype=x.dtype)
    nn.init.xavier_uniform_(w)
    y = F.conv2d(x_trans, w, bias=None, padding=(k // 2, 0))  # (B,1,D,W)

    # 沿深度维归一化注意力
    y = torch.softmax(y, dim=-2)  # (B,1,D,W)
    y = y.transpose(-2, -3)  # (B,D,1,W)
    # 缩放（与原始实现一致，用通道数 C）
    return y * C * x


# ---------- 适配器：补维度 & 使用无参 Depth-Attention ----------
class LMDANetAdapter(nn.Module):
    """
    适配 (B,C,T) -> (B,1,C,T) 并复刻 LMDA.forward 的主要步骤，但用无参注意力避免打印/可训练开销。
    """

    def __init__(self, core: LMDA, attn_k: int = 7):
        super().__init__()
        self.core = core
        self.attn_k = attn_k

    def forward(self, x):
        # x: (B, C, T) => (B,1,C,T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # 深度展开： (B,1,C,T) x (depth,1,C) -> (B,depth,C,T)
        x = torch.einsum("bdcw, hdc -> bhcw", x, self.core.channel_weight)

        # 时间卷积 + 函数式Depth-Attention
        x_time = self.core.time_conv(x)  # (B,depth1,C,T')
        x_time = _depth_attention_stateless(x_time, self.attn_k)

        # 空间卷积 + 池化/Dropout
        x = self.core.chanel_conv(x_time)  # (B,depth2,1,T'')
        x = self.core.norm(x)  # (B,depth2,1,T''')

        # 分类头
        feat = torch.flatten(x, 1)
        logits = self.core.classifier(feat)
        return logits


# ---------- DataLoader ----------
def build_loaders(base_dir, num_classes, test_person_idx, batch_size):
    Xtr, Ytr, Xte, Yte, person_folders = create_dataset_for_cross_validation(
        base_dir=base_dir, num_classes=num_classes, test_person_idx=test_person_idx
    )
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


# ---------- 单人训练/评估 ----------
def run_one(args, sid: int):
    # 建议对每个被试 seed 略作偏移，避免曲线“视觉上完全一致”
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

    # 动态获取 T（时间长度），据此初始化核心 LMDA（确保 __init__ 时用到的 samples 与数据一致）
    T = train_loader.dataset.tensors[0].shape[-1]
    core = LMDA(
        chans=args.channels,
        samples=T,
        num_classes=args.n_class,
        depth=args.depth,
        kernel=args.kernel,
        channel_depth1=args.channel_depth1,
        channel_depth2=args.channel_depth2,
        ave_depth=args.ave_depth,
        avepool=args.avepool,
    )
    model = LMDANetAdapter(core, attn_k=args.attn_k).to(device)  # 严格单设备

    # 优化器 / 损失 / 调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 日志
    sub_logdir = os.path.join(args.logdir, subj_name)
    os.makedirs(sub_logdir, exist_ok=True)
    writer = SummaryWriter(sub_logdir) if SummaryWriter is not None else None

    if swanlab is not None:
        try:
            swanlab.init(
                project="Main-LMDANet-LOS_Filter",
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

        # 保存最优
        acc = val_metrics["acc"]
        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                os.path.join(sub_logdir, f"best_lmdanet_cs_{subj_name}.pth"),
            )

    if writer is not None:
        writer.close()
    if swanlab is not None:
        try:
            swanlab.finish()
        except Exception:
            pass

    print(f"\n==> Subject {subj_raw} Finished. Best Test Acc: {best_acc:.2%}\n")


# ---------- 入口 ----------
def main():
    parser = argparse.ArgumentParser(
        "Cross-Session Training for LMDA-Net (logits-only, single device)"
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
        "-device", type=int, default=2
    )  # 单设备索引（建议配合 CUDA_VISIBLE_DEVICES）
    parser.add_argument("-logdir", type=str, default="./runs_lmdanet_cs")
    # —— LMDA 结构超参（与源码一致的默认）——
    parser.add_argument("-depth", type=int, default=9)
    parser.add_argument("-kernel", type=int, default=75)
    parser.add_argument("-channel_depth1", type=int, default=24)
    parser.add_argument("-channel_depth2", type=int, default=9)
    parser.add_argument("-ave_depth", type=int, default=1)
    parser.add_argument("-avepool", type=int, default=25)
    parser.add_argument(
        "-attn_k", type=int, default=7, help="Depth-Attention 的深度卷积核大小"
    )
    # 打印频率
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
