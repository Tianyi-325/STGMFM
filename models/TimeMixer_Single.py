from __future__ import annotations
from types import SimpleNamespace

import torch
import torch.nn as nn

# 原作者提供的 TimeMixer 实现
from models.TimeMixer import Model as _TimeMixerCore


# ----------------------------------------------------------------------
# ------------------------ 单分支 TimeMixer Net ------------------------
# ----------------------------------------------------------------------
class STGENET(nn.Module):
    """
    Thin wrapper around the official *TimeMixer* implementation
    that (1) converts the EEG tensor from [B,C,T] → [B,T,C],
    (2) fabricates the minimal time-mask required by TimeMixer,
    (3) returns dummy graph tensors so that the old training code
    (`run_epoch_*`) remains untouched.
    """

    def __init__(
        self,
        in_chans: int,
        n_classes: int,
        seq_len: int = 1125,
        dropout: float = 0.3,
        d_model: int = 64,
        down_sampling_window: int = 2,
        down_sampling_layers: int = 3,
        e_layers: int = 2,
        moving_avg: int = 5,
        decomp_method: str = "moving_avg",  # or 'dft_decomp'
        top_k: int = 5,
        channel_independence: int = 0,
        device: int | str = 0,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Build TimeMixer **configs** (SimpleNamespace) exactly as the demo
        # ------------------------------------------------------------------
        self.cfg = SimpleNamespace(
            # --- DATA / TASK ---
            task_name="classification",
            num_class=n_classes,
            enc_in=in_chans,
            seq_len=seq_len,
            label_len=0,
            pred_len=0,
            # --- PDM / multiscale ---
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
            e_layers=e_layers,
            moving_avg=moving_avg,
            decomp_method=decomp_method,
            top_k=top_k,
            # --- channels / misc ---
            channel_independence=channel_independence,
            c_out=in_chans,
            d_model=d_model,
            d_ff=d_model * 2,
            embed="timeF",
            freq="h",
            dropout=dropout,
            use_future_temporal_feature=False,
            use_norm=1,
            down_sampling_method="max",
        )

        self.core = _TimeMixerCore(self.cfg).to(device)
        self.in_chans = in_chans
        self.device = device

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _dummy_graphs(batch: int, chans: int, device):
        """Return constant tensors so that old code (`*_graph`) never breaks."""
        # (1) spatial graph占位   (chans × chans)
        ccg_w = torch.zeros(chans, chans, device=device)
        # (2) time graph占位      (1 × 1)
        tsg = torch.zeros(1, 1, device=device)
        return ccg_w, tsg

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor):
        """
        x : (B, C, T)  — identical to old STGENet inputs
        Returns:
            logits, ccg_w_A, tsg_A, tsg_B, ccg_w_B   (all tensors)
        """
        B, C, T = x.shape
        assert C == self.in_chans, f"Channel mismatch: {C} ≠ {self.in_chans}"
        x_enc = x.permute(0, 2, 1).contiguous()  # (B,T,C)
        # 全 1 时间掩码即可（TimeMixer 只在 classification 前向用来 zero-out paddings）
        x_mark_enc = torch.ones(B, T, device=x.device)

        # TimeMixer 其它输入在分类任务下可设 None
        logits = self.core(x_enc, x_mark_enc, None, None)  # (B, n_classes)

        # fabricate dummy graphs for compatibility
        ccg_w_A, tsg_A = self._dummy_graphs(B, C, x.device)
        tsg_B, ccg_w_B = tsg_A, ccg_w_A

        return logits, ccg_w_A, tsg_A, tsg_B, ccg_w_B


# -------------------------- quick sanity check --------------------------
if __name__ == "__main__":
    B, C, T = 32, 23, 1125
    net = STGENET(in_chans=C, n_classes=3, seq_len=T, device=0)
    x = torch.randn(B, C, T, device=net.device)  # (B, C, T)
    logits, *_ = net(x)
    print("logits.shape =", logits.shape)  # → (32, 3)
