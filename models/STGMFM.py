from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from utils.init import glorot_weight_zero_bias
from utils.tools import normalize

# ─────────────────  引入 Branch-C  ──────────────────
from STGMFM.models.EnvelopeFrequencyMixer_Single import STGMFM_C


# ─────────────────  Graph & Conv 定义（原样保留） ──────────────────
class SpatialGraph(nn.Module):
    def __init__(self, n_nodes, adj, k=2, spatial_GCN=True, device=0):
        super().__init__()
        xs, ys = torch.tril_indices(n_nodes, n_nodes, offset=-1)
        self.register_buffer("I", torch.eye(n_nodes))
        self.edge_weight = nn.Parameter(adj[xs, ys].clone(), requires_grad=spatial_GCN)
        # self.edge_weight = adj[xs, ys].clone()
        self.n_nodes, self.k, self.spatial_GCN = n_nodes, k, spatial_GCN
        self.xs, self.ys = xs, ys

    # def forward(self, x):  # x: (B,W,C,L)
    #     device = x.device
    #     if not self.spatial_GCN:
    #         A = self.I.to(device)
    #     else:
    #         A = torch.zeros(self.n_nodes, self.n_nodes, device=device)
    #         A[self.xs, self.ys] = self.edge_weight.to(device)
    #         A = A + A.T + self.I.to(device)
    #         A = normalize(A) + self.I.to(device)
    #     x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), self.n_nodes, -1)
    #     for _ in range(self.k):
    #         x = torch.matmul(A, x)
    #     return x, A

    def forward(self, x):  # x: (B,W,C,L)
        device = x.device
        A = torch.zeros(self.n_nodes, self.n_nodes, device=device)
        A[self.xs, self.ys] = self.edge_weight.to(device)
        A = A + A.T + self.I.to(device)
        # A = normalize(A) + self.I.to(device)
        A = normalize(A)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), self.n_nodes, -1)
        for _ in range(self.k):
            x = torch.matmul(A, x)
        return x, A


class TimeGraph(nn.Module):
    def __init__(self, window, k, channels, time_GCN=True):
        super().__init__()
        self.adj = nn.Parameter(
            0.5 * torch.ones(window, window) + 1.5 * torch.eye(window),
            requires_grad=time_GCN,
        )
        self.register_buffer("I", torch.eye(window))
        self.window, self.channels, self.k, self.time_GCN = (
            window,
            channels,
            k,
            time_GCN,
        )

    def forward(self, x):  # (B,W,C,L)
        device = x.device
        A = (
            self.I.to(device)
            if not self.time_GCN
            else normalize((self.adj + self.adj.T) / 2).to(device)
        )
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), self.window, -1)
        for _ in range(self.k):
            x = torch.matmul(A, x)
        x = x.view(x.size(0), self.window, self.channels, -1).permute(0, 2, 1, 3)
        return x, A


class Conv(nn.Module):
    def __init__(self, conv, bn=None, activation=None):
        super().__init__()
        self.conv, self.bn, self.activation = conv, bn, activation
        if bn:
            self.conv.bias = None

    def forward(self, x):
        # # -----New----- #
        # dev = x.device
        # # —— 设备保险：把包裹的算子/BN 挪到输入所在设备（若已在则无开销）
        # self.conv.to(dev)
        # if self.bn is not None:
        #     self.bn.to(dev)
        # # -----New----- #

        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class GateFuse(nn.Module):
    """
    Soft-gate 融合（初始化偏置抑制 C 分支）＋ 示例性熵加权
    """

    def __init__(
        self, T: float = 1.0, init_bias: tuple[float, float, float] = (0.0, 0.0, -2.0)
    ):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(init_bias))  # A,B,C
        self.T = T

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        p = logits.softmax(-1)
        return (-p * logits.log_softmax(-1)).sum(-1)  # (B,)

    def forward(
        self,
        a_logits: torch.Tensor,
        b_logits: torch.Tensor,
        c_logits: torch.Tensor,
        use_entropy: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (B,n_cls) → 置信度
        if use_entropy:
            ent = torch.stack(
                [
                    self._entropy(a_logits),
                    self._entropy(b_logits),
                    self._entropy(c_logits),
                ],
                dim=-1,
            )  # (B,3)
            conf = (-ent).softmax(-1).mean(0)  # (3,)
        else:
            conf = torch.ones(3, device=a_logits.device)

        a = torch.softmax(self.gate / self.T, 0) * conf  # (3,)
        fused = a[0] * a_logits + a[1] * b_logits + a[2] * c_logits
        return fused, a  # logits, 权重


# ─────────────────────  Triple-Branch 主干  ─────────────────────
class STGMFM(nn.Module):
    """
    forward(x) → 5-tuple:
        logits , ccg_w_A , tsg_A , tsg_B , ccg_w_B
    """

    def __init__(
        self,
        Adj,
        in_chans,
        n_classes,
        time_window_num,
        spatial_GCN=True,
        time_GCN=True,
        k_spatial=2,
        k_time=2,
        dropout=0.2,
        input_time_length=125,
        out_chans=64,
        kernel_size=63,
        slide_window=9,
        sampling_rate=250,
        # device=0,
        # 多设备修改：
        device=0,
        # device_c=1,  # A/B 分支在 device_ab，Branch-C 在 device_c
        #  Branch-C 额外可调:
        tm_d_model=16,
        tm_e_layers=1,
        tm_moving_avg=5,
        tm_top_k=1,
    ):
        super().__init__()
        self.__dict__.update(locals())
        del self.self  # 清理 self 影子

        # ---------- Device 分配 ----------
        # self.device_ab = device
        # self.device_c = device_c  # Branch-C 也在主卡上，原单卡模式

        # ---------- 共享卷积 A/B ----------
        self.spatialconv = Conv(
            nn.Conv1d(in_chans, out_chans, 1, bias=False),
            bn=nn.BatchNorm1d(out_chans),
        )
        self.timeconv = nn.ModuleList(
            [
                Conv(
                    nn.Conv1d(
                        out_chans, out_chans, kernel_size, padding="same", bias=False
                    ),
                    bn=nn.BatchNorm1d(out_chans),
                )
            ]
        )

        # ---------- Graph ----------
        self.ccg = SpatialGraph(in_chans, Adj, k_spatial, spatial_GCN, device)
        # self.ccg = SpatialGraph(in_chans, Adj, k_spatial, spatial_GCN, device)
        self.tsg_after = TimeGraph(slide_window, k_time, out_chans, time_GCN)
        self.tsg_before = TimeGraph(slide_window, k_time, in_chans, time_GCN)

        pool_len = input_time_length * slide_window // (sampling_rate // 2)
        self.down = nn.AvgPool1d(sampling_rate // 2, sampling_rate // 2)
        self.dp = nn.Dropout(dropout)
        self.fc_a = nn.Linear(out_chans * pool_len, n_classes)
        self.fc_b = nn.Linear(out_chans * pool_len, n_classes)

        # ---------- Branch-C : Envelope-TimeMixer ----------
        total_T = input_time_length * slide_window
        self.branchC = STGMFM_C(
            in_chans=in_chans,
            n_classes=n_classes,
            seq_len=total_T,
            d_model=tm_d_model,
            e_layers=tm_e_layers,
            moving_avg=tm_moving_avg,
            top_k=tm_top_k,
            down_sampling_layers=1,
            device=device,
            # device=device_c,
        )

        # ---------- Learnable fusion parameters ----------
        self.gate = nn.Parameter(torch.ones(3) / 3)  # soft gate
        self.tau = nn.Parameter(torch.ones(3))  # temperature

        # ① 追加：3 条分支特征可能维度不同，投影到同一 d_fuse
        d_fuse = 128
        self.projA = nn.Linear(out_chans * pool_len, d_fuse)  # featA 维度
        self.projB = nn.Linear(out_chans * pool_len, d_fuse)  # featB
        self.projC = nn.Linear(tm_d_model, d_fuse)  # featC 维度 = d_model

        # ② gate：可学习软权重 (初始平均)
        self.gate = nn.Parameter(torch.ones(3) / 3)

        # ③ 融合后的最终分类头
        # self.fc_fuse = nn.Linear(d_fuse, n_classes)
        # self.fc_fuse = nn.Linear(3 * d_fuse, n_classes)
        self.fc_fuse = nn.Linear(n_classes * 3, n_classes)  # ← 三分支 × n_classes
        self.fc_fuse2 = nn.Linear(
            n_classes * 2, n_classes
        )  # ← 双分支 × n_classes 保留AC分支

        self.fuser = GateFuse(
            T=1.0, init_bias=(0.0, 0.0, -2.0)
        )  # Gatefuse: C 分支默认优先级低

        # ---------- 初始化 (安全版) ----------
        def _safe_glorot(m: nn.Module):
            if hasattr(m, "weight") and m.weight is not None and m.weight.dim() >= 2:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

        self.apply(_safe_glorot)

    # ───────── helper ─────────
    def _stem(self, x):  # (B,C,T) -> (B,W,C,L)
        return x.view(x.size(0), x.size(1), self.time_window_num, -1).permute(
            0, 2, 1, 3
        )

    def _shared_tail(self, x):
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        x = self.down(x)
        return self.dp(x)

    # ───────── forward ─────────
    def forward(self, x):
        """
        x : (B,C,T_total)   ;  T_total = input_time_length × slide_window
        """
        x_stem = self._stem(x)

        # ===== Branch-A =====
        a, ccg_w = self.ccg(x_stem)
        a = self.spatialconv(a)
        for conv in self.timeconv:
            a = conv(a)
        a = F.gelu(a).view(a.size(0), a.size(1), self.slide_window, -1)
        a, tsg_a = self.tsg_after(a)
        a = self._shared_tail(a)
        featA = a.view(a.size(0), -1)  # (B, out_chans*pool_len)
        logitsA = self.fc_a(featA)
        # logitsA = self.fc_a(a.view(a.size(0), -1))

        # ===== Branch-B =====
        b, tsg_b = self.tsg_before(x_stem)
        b = b.permute(0, 2, 1, 3)
        b, ccg_w_b = self.ccg(b)
        b = self.spatialconv(b)
        for conv in self.timeconv:
            b = conv(b)
        b = F.gelu(b)
        b = self._shared_tail(b)
        featB = b.view(b.size(0), -1)
        logitsB = self.fc_b(featB)
        # logitsB = self.fc_b(b.view(b.size(0), -1))

        # ===== Branch-C =====
        logitsC, featC, *_ = self.branchC(x)
        # -------- Branch-C 独立 GPU --------
        # x_c = x.to(self.device_c, non_blocking=True)
        # logitsC, *_ = self.branchC(x_c)
        # logitsC = logitsC.to(x.device, non_blocking=True)  # 搬回主卡

        # ===== 可学习温度 + gate 融合（测试） =====
        # stack = torch.stack(
        #     [logitsA / self.tau[0], logitsB / self.tau[1], logitsC / self.tau[2]], dim=0
        # )  # (3,B,Cls)
        # w = F.softmax(self.gate, dim=0).view(3, 1, 1)
        # logits = (w * stack).sum(dim=0)

        # -------- 决策级融合（主实验） -------------------
        fused_logits = self.fc_fuse(
            torch.cat([logitsA, logitsB, logitsC], dim=1)
        )  # 形状 [B, n_classes]
        logits = fused_logits

        return logits, ccg_w, tsg_a, tsg_b, ccg_w_b

        # # -------- 决策级融合(双分支消融实验) -------------------
        # fused_logits = self.fc_fuse2(
        #     torch.cat([logitsA, logitsB], dim=1)
        # )  # 形状 [B, n_classes]
        # logits = fused_logits

        # return logits, ccg_w, tsg_a, tsg_b, ccg_w_b

        # -------- 单分支C消融实验 -------------------
        # logits = logitsC

        # return logits, ccg_w, tsg_a, tsg_b, ccg_w_b

        # # # ========== 特征级 + Gate 融合 ==========
        # # 1) 维度对齐
        # fA = self.projA(featA)  # (B, d_fuse)
        # fB = self.projB(featB)
        # fC = self.projC(featC)

        # # 2) 软门控权重
        # w = torch.softmax(self.gate, dim=0)  # (3,)

        # # 3) 加权求和得到融合特征
        # fuse_feat = w[0] * fA + w[1] * fB + w[2] * fC  # (B, d_fuse)
        # # fuse_feat = torch.cat([fA, fB, fC], dim=1)

        # # 4) 最终分类
        # logits = self.fc_fuse(fuse_feat)  # (B, n_classes)

        # return logits, ccg_w, tsg_a, tsg_b, ccg_w_b

        # 2) GateFuse
        # 2) GateFuse
        # logits, alpha = self.fuser(logitsA, logitsB, logitsC, use_entropy=True)
        # logits = logitsC

        # 3) 维持原返回顺序（先 fused，再各分支），以免 run_epoch 报 shape 错
        # return logits, ccg_w, tsg_a, tsg_b, ccg_w_b


# ───────── quick sanity check ─────────
if __name__ == "__main__":
    B, C, T = 4, 23, 1125
    dummy_adj = torch.ones(C, C)
    net = STGMFM(
        Adj=dummy_adj,
        in_chans=C,
        n_classes=3,
        time_window_num=9,
        sampling_rate=250,
        device=0,
        # ---Dual device settings--- #
        # device_ab=0,
        # device_c=1,  # A/B 分支在 device_ab，Branch-C 在 device_c
    )
    x = torch.randn(B, C, T)
    y, *rest = net(x)
    print("logits:", y.shape, " gates:", net.gate.softmax(0).data)
