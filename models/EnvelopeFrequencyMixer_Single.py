# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------ Tools ------------------------------
def _moving_avg_1d(x: torch.Tensor, win: int) -> torch.Tensor:
    """
    x: (B, D, T)
    return: (B, D, T)  -- same length, edge-replicate
    """
    if win <= 1:
        return x
    pad = (win - 1) // 2
    xpad = F.pad(x, (pad, pad), mode="replicate")
    ker = torch.ones(1, 1, win, device=x.device, dtype=x.dtype) / win
    y = F.conv1d(xpad, ker.repeat(x.size(1), 1, 1), groups=x.size(1))
    return y


def _rfft_topk_periods(
    x: torch.Tensor,  # (B, D, T)
    top_k: int = 2,
    pmin: int = 4,
    pmax: int | None = None,
    harmonic_tol: float = 0.12,  # ~±12% 视为谐波/倍频
) -> Tuple[List[int], torch.Tensor]:
    """
    在最粗尺度/当前层上做 RFFT, 选 Top-K 频率 (能量), 转成周期(按采样点).
    返回: periods(list[int]), energies_at_peaks (K,)
    """
    B, D, T = x.shape
    x_center = x - x.mean(dim=-1, keepdim=True)
    # Hann 窗
    n = torch.arange(T, device=x.device, dtype=x.dtype)
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * n / (T - 1))
    xw = x_center * w  # (B, D, T)
    X = torch.fft.rfft(xw, dim=-1)  # (B, D, T//2+1)
    A2 = (X.real**2 + X.imag**2).mean(dim=(0, 1))  # (F,) 按B、D聚合后的能量谱
    A2[0] = 0.0  # 去直流

    # 频段/周期范围约束
    Fbins = A2.shape[0]  # = T//2+1
    k = torch.arange(Fbins, device=x.device)
    if pmax is None:
        pmax = T // 2
    # 以周期限制频率索引范围：k ∈ [ceil(T/pmax), floor(T/pmin)]
    kmin = int(math.ceil(T / max(pmax, 1)))
    kmax = int(math.floor(T / max(pmin, 1)))
    kmin = max(kmin, 1)
    kmax = min(kmax, Fbins - 1)
    mask = torch.zeros_like(A2, dtype=torch.bool)
    if kmin <= kmax:
        mask[kmin : kmax + 1] = True
    A2_mask = torch.where(mask, A2, torch.zeros_like(A2))

    # 寻找候选峰：简单局部极大 + 排序
    # （工程上可替换为更稳健的 peak picking）
    peaks = []
    for idx in range(1, Fbins - 1):
        if not mask[idx]:
            continue
        if A2_mask[idx] > A2_mask[idx - 1] and A2_mask[idx] >= A2_mask[idx + 1]:
            peaks.append((A2_mask[idx].item(), idx))
    peaks.sort(key=lambda t: t[0], reverse=True)

    # 谐波去重 + 取 Top-K
    selected = []
    for amp, idx in peaks:
        ok = True
        for _, j in selected:
            r = idx / j if j > 0 else 1.0
            if abs(r - 2.0) < 2 * harmonic_tol or abs(r - 0.5) < 2 * harmonic_tol:
                ok = False
                break
        if ok:
            selected.append((amp, idx))
        if len(selected) >= top_k:
            break

    if len(selected) == 0:
        # 兜底：若无峰，取中频索引
        mid = (kmin + kmax) // 2
        selected = [(A2_mask[mid].item(), mid)]
    # 组装
    idxs = [idx for _, idx in selected]
    energies = torch.tensor([A2[i] for i in idxs], device=x.device, dtype=x.dtype)
    periods = [int(math.ceil(T / max(i, 1))) for i in idxs]
    return periods, energies


def _fold_to_image(x: torch.Tensor, period: int) -> Tuple[torch.Tensor, int]:
    """
    x: (B, D, T)  ->  Z: (B, D, P, N)
    返回 Z 以及 N 列数
    """
    B, D, T = x.shape
    P = max(int(period), 1)
    N = math.ceil(T / P)
    pad_len = P * N - T
    if pad_len > 0:
        pad_tail = x[..., -1:].repeat(1, 1, pad_len)
        x = torch.cat([x, pad_tail], dim=-1)
    Z = x.view(B, D, N, P).transpose(2, 3).contiguous()  # (B,D,P,N)
    return Z, N


def _unfold_image(Z: torch.Tensor, T_target: int) -> torch.Tensor:
    """
    Z: (B, D, P, N)  ->  x: (B, D, T_target) （截断补齐的尾部）
    """
    B, D, P, N = Z.shape
    x = Z.transpose(2, 3).reshape(B, D, P * N)
    return x[..., :T_target]


def _bilinear_align_to(Z: torch.Tensor, P_tgt: int, N_tgt: int) -> torch.Tensor:
    """
    Z: (B, D, P, N)  -> aligned to (B, D, P_tgt, N_tgt)
    """
    B, D, P, N = Z.shape
    Z4 = Z  # (B,D,P,N)
    Z4 = Z4.unsqueeze(2)  # (B,D,1,P,N) 
    return F.interpolate(Z, size=(P_tgt, N_tgt), mode="bilinear", align_corners=False)


# 轴向卷积模块：T-centr ic（列向强、行向轻）
class AxialOps(nn.Module):
    def __init__(
        self, d_model: int, k_row: int = 3, k_col: int = 5, dropout: float = 0.0
    ):
        super().__init__()
        pad_r = k_row // 2
        pad_c = k_col // 2
        # 行向（相位轴）轻量：depthwise 3x1 + pointwise 1x1
        self.row_dw = nn.Conv2d(
            d_model, d_model, kernel_size=(k_row, 1), padding=(pad_r, 0), groups=d_model
        )
        self.row_pw = nn.Conv2d(d_model, d_model, kernel_size=1)
        # 列向（跨周期轴）更强：depthwise 1xk + pointwise 1x1 + 再叠一层
        self.col_dw1 = nn.Conv2d(
            d_model, d_model, kernel_size=(1, k_col), padding=(0, pad_c), groups=d_model
        )
        self.col_pw1 = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.col_dw2 = nn.Conv2d(
            d_model, d_model, kernel_size=(1, k_col), padding=(0, pad_c), groups=d_model
        )
        self.col_pw2 = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Z: (B, D, P, N)
        return S, T with the same shape
        """
        # S: 行向
        S = self.row_pw(self.act(self.row_dw(Z)))
        # T: 列向（两层）
        T = self.col_pw1(self.act(self.col_dw1(Z)))
        T = self.col_pw2(self.act(self.col_dw2(T)))
        S = self.drop(S)
        T = self.drop(T)
        return S, T


# ---------------------------- 主干：单分支 ----------------------------
class STGMFM_C(nn.Module):
    """
    单分支：Envelope-MRTI + T-centric + K=2 + Harmonic de-dup
    形参/forward 与 TimeMixer_Single 保持一致，返回兼容的 5-tuple。
    """

    def __init__(
        self,
        in_chans: int,
        n_classes: int,
        seq_len: int = 1125,
        dropout: float = 0.3,
        d_model: int = 64,
        down_sampling_window: int = 2,
        down_sampling_layers: int = 2,  # 3
        e_layers: int = 2,
        moving_avg: int = 5,
        decomp_method: str = "moving_avg",  # 保留占位，无直接使用
        top_k: int = 2,  # 默认 K=2
        channel_independence: int = 0,  # 0=通道混合
        device: int | str = 0,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.dropout = dropout
        self.d_model = d_model
        self.down_sampling_layers = down_sampling_layers
        self.e_layers = e_layers
        self.moving_avg = max(int(moving_avg), 1)
        self.top_k = max(int(top_k), 1)
        self.channel_independence = channel_independence
        self.device_id = device

        # 通道嵌入（C->D）
        if channel_independence == 1:
            # 每通道各自线性，等价 depthwise 1x1
            self.embed = nn.Conv1d(in_chans, d_model, kernel_size=1, groups=in_chans)
        else:
            self.embed = nn.Conv1d(in_chans, d_model, kernel_size=1, groups=1)

        # 多尺度下采样：反混叠 depthwise conv stride=2 堆叠
        downs = []
        for _ in range(down_sampling_layers):
            downs.append(
                nn.Sequential(
                    nn.Conv1d(
                        d_model,
                        d_model,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        groups=d_model,
                    ),
                    nn.Conv1d(d_model, d_model, kernel_size=1),
                    nn.GELU(),
                )
            )
        self.downs = nn.ModuleList(downs)

        # 每层的轴向模块
        self.axial = nn.ModuleList(
            [
                AxialOps(d_model, k_row=3, k_col=5, dropout=dropout)
                for _ in range(e_layers)
            ]
        )

        # MRM 共享投影（对齐后，保持维度不变）
        self.proj = nn.ModuleList(
            [nn.Conv2d(d_model, d_model, kernel_size=1) for _ in range(e_layers)]
        )

        # 逐尺度残差归一化
        self.ln_per_scale = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(down_sampling_layers + 1)]
        )

        # 分类头：最细尺度池化后线性
        self.cls = nn.Linear(d_model, n_classes)

    # 占位，兼容旧代码期望的“图”输出
    @staticmethod
    def _dummy_graphs(batch: int, chans: int, device):
        ccg_w = torch.zeros(chans, chans, device=device)
        tsg = torch.zeros(1, 1, device=device)
        return ccg_w, tsg

    def _build_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: (B, D, T)
        return: list of scales [x0(细), x1, ..., xM(粗)], each (B, D, T_m)
        """
        xs = [x]
        cur = x
        for op in self.downs:
            cur = op(cur)
            xs.append(cur)
        return xs

    def _envelope(self, x: torch.Tensor, win: int) -> torch.Tensor:
        """
        简单 RMS 包络：sqrt( avg_pool(x^2) )；为数值稳定取 log-power 可改用 log1p
        """
        power = x * x
        env = _moving_avg_1d(power, win)  # (B,D,T)
        return torch.sqrt(env + 1e-8)

    def forward(self, x: torch.Tensor):
        """
        x : (B, C, T)   →  logits, ccg_w_A, tsg_A, tsg_B, ccg_w_B
        """
        B, C, T = x.shape
        assert T == self.seq_len, f"seq_len mismatch: {T} != {self.seq_len}"
        device = x.device

        # 1) 通道嵌入  (B,C,T) -> (B,D,T)
        feat = self.embed(x)  # (B, D, T)

        # 2) 构建多尺度金字塔（细→粗）
        scales = self._build_pyramid(feat)  # List[(B,D,T_m)], len = M+1

        # L 个 MixerBlock
        for l in range(self.e_layers):
            # 2.1 包络（每尺度，无相位）
            env_scales = [
                self._envelope(s, self.moving_avg) for s in scales
            ]  # (B,D,T_m)

            # 2.2 在最粗尺度上选 Top-K 周期（可加频段约束：这里按样本周期 p∈[4, T/2]）
            coarse = env_scales[-1]  # (B,D,T_M)
            # 修改模块 #
            # fs = 250.0  # 原始采样率（Hz）
            # M = self.down_sampling_layers  # 你设置的下采样层数
            # fs_eff = fs / (2**M)  # 最粗尺度的有效采样率

            # fmin, fmax = 8.0, 30.0  # 目标频段
            # pmin = int(math.ceil(fs_eff / fmax))
            # pmax = int(math.ceil(fs_eff / fmin))

            # periods, energies = _rfft_topk_periods(
            #     coarse, top_k=self.top_k, pmin=pmin, pmax=pmax
            # )
            # end #
            periods, energies = _rfft_topk_periods(
                coarse, top_k=self.top_k, pmin=4, pmax=coarse.shape[-1] // 2
            )
            K = len(periods)
            # 归一化能量供 MRM 权重
            w = torch.softmax(energies, dim=0)  # (K,)

            # 2.3 MRTI + TID：对每个尺度、每个周期折图像并做轴向卷积
            S_all: List[List[torch.Tensor]] = []  # [m][k] -> (B,D,P_k,N_mk)
            T_all: List[List[torch.Tensor]] = []
            Ns = []  # 保存每尺度在各 k 下的列数
            for m, em in enumerate(env_scales):
                Sm_list, Tm_list, Nm_list = [], [], []
                for k_idx, p in enumerate(periods):
                    Z, Nmk = _fold_to_image(em, p)  # (B,D,P,N)
                    # 轴向卷积
                    S, Tz = self.axial[l](Z)  # (B,D,P,N)
                    Sm_list.append(S)
                    Tm_list.append(Tz)
                    Nm_list.append(Nmk)
                S_all.append(Sm_list)
                T_all.append(Tm_list)
                Ns.append(Nm_list)

            # 2.4 MCM：季节 S 自下而上，趋势 T 自上而下（逐尺度目标）
            H_per_scale_k: List[List[torch.Tensor]] = []
            M = len(env_scales) - 1
            for m in range(M + 1):  # 目标尺度
                Hm_k: List[torch.Tensor] = []
                for k_idx in range(K):
                    # 目标几何
                    Pk = S_all[m][k_idx].shape[2]
                    Nm = Ns[m][k_idx]
                    # S: 汇聚细→粗（下采样对齐到 m）
                    S_fused = S_all[m][k_idx]
                    # 叠加更细的
                    for j in range(0, m):
                        Sj = _bilinear_align_to(S_all[j][k_idx], Pk, Nm)
                        S_fused = S_fused + 0.7 * Sj
                    # 叠加更粗的（少量）
                    for j in range(m + 1, M + 1):
                        Sj = _bilinear_align_to(S_all[j][k_idx], Pk, Nm)
                        S_fused = S_fused + 0.3 * Sj

                    # T: 传递粗→细（上采样对齐到 m）
                    T_fused = T_all[m][k_idx]
                    for j in range(m + 1, M + 1):
                        Tj = _bilinear_align_to(T_all[j][k_idx], Pk, Nm)
                        T_fused = T_fused + 0.7 * Tj
                    for j in range(0, m):
                        Tj = _bilinear_align_to(T_all[j][k_idx], Pk, Nm)
                        T_fused = T_fused + 0.3 * Tj

                    Hm_k.append(S_fused + T_fused)  # (B,D,P,N)
                H_per_scale_k.append(Hm_k)

            # 2.5 MRM（逐尺度）：跨 k 对齐→共享投影→能量加权
            new_scales: List[torch.Tensor] = []
            for m in range(M + 1):
                # 目标网格：P* = max P_k,  N* = max N_mk
                P_star = max(H_per_scale_k[m][k].shape[2] for k in range(K))
                N_star = max(H_per_scale_k[m][k].shape[3] for k in range(K))
                # 对齐并共享投影
                Hk_aligned = []
                for k_idx in range(K):
                    Hmk = _bilinear_align_to(H_per_scale_k[m][k_idx], P_star, N_star)
                    Hmk = self.proj[l](Hmk)  # (B,D,P*,N*)
                    Hk_aligned.append(Hmk)
                # 能量加权
                Y_img = sum(
                    w[k_idx] * Hk_aligned[k_idx] for k_idx in range(K)
                )  # (B,D,P*,N*)
                # 反折叠回 1D，并残差+LN
                y = _unfold_image(Y_img, scales[m].shape[-1])  # (B,D,T_m)
                # 残差 + LN（按最后维 D 归一化，需要转置）
                x_in = scales[m].transpose(1, 2)  # (B,T_m,D)
                y_in = y.transpose(1, 2)  # (B,T_m,D)
                x_out = self.ln_per_scale[m](x_in + y_in)  # (B,T_m,D)
                new_scales.append(x_out.transpose(1, 2))  # (B,D,T_m)
            # 更新
            scales = new_scales

        # 3) 分类头：取最细尺度 m=0，做 GAP→FC
        x_finest = scales[0].transpose(1, 2)  # (B,T,D)
        h = x_finest.mean(dim=1)  # (B,D)
        logits = self.cls(
            F.dropout(h, p=self.dropout, training=self.training)
        )  # (B,nc)

        # 兼容旧训练流程的“图”占位
        ccg_w_A, tsg_A = self._dummy_graphs(B, self.in_chans, device)
        tsg_B, ccg_w_B = tsg_A, ccg_w_A
        return logits, h, ccg_w_A, tsg_A, tsg_B, ccg_w_B
        # return logits, ccg_w_A, tsg_A, tsg_B, ccg_w_B


# -------------------------- quick sanity check --------------------------
if __name__ == "__main__":
    B, C, T = 1, 23, 1125
    net = STGMFM_C(in_chans=C, n_classes=3, seq_len=T, d_model=64, top_k=2, e_layers=2)
    x = torch.randn(B, C, T)
    logits, *_ = net(x)
    print("logits:", logits.shape)
    print(logits)
