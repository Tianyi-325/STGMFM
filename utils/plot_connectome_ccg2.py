# utils/plot_connectome_ccg.py
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn.plotting import plot_connectome

# 你当前 Plot_cor.py 里的自定义坐标，搬过来即可
CUSTOM_COORDS = {
    "F8": [47.816485, 37.899935, -5.797835],
    "Fp2": [17.687171, 64.404845, -0.911275],
    "Fpz": [0, 65.054019, -1.662449],
    "Fp1": [-17.687171, 64.404845, -0.911275],
    "F7": [-47.721767, 37.546411, -6.602358],
    "F4": [24.681453, 28.099366, 40.78425],
    "Fz": [0, 14.857909, 47.316343],
    "AFz": [0, 39.955964, 24.903584],
    "F3": [-24.681453, 28.099366, 40.78425],
    "C4": [28.212411, -18.624961, 67.445004],
    "CPz": [0, -39.84933, 74.525815],
    "Cz": [0, -17.134411, 72.873756],
    "C3": [-28.212411, -18.624961, 67.445004],
    "T8": [61.970197, -18.602323, -14.974811],
    "P4": [31.366912, -64.40286, 42.930524],
    "Pz": [0, -67.020154, 58.499698],
    "P3": [-31.366912, -64.40286, 42.930524],
    "P8": [55.272316, -58.257183, -1.244697],
    "O1": [-20.98644, -99.451091, 6.327694],
    "Oz": [0, -99.012474, 9.670115],
    "O2": [22.100274, -98.920295, 5.670776],
    "P7": [-55.272316, -58.257183, -1.244697],
    "T7": [-62.167252, -20.150615, -14.782996],
}


def _parse_threshold(W: np.ndarray, edge_threshold: str | float | None) -> float:
    """把 '90%' 或 float 或 None 统一成数值阈值（基于 |W| 上三角）。"""
    absW = np.abs(W[np.triu_indices_from(W, k=1)])
    if isinstance(edge_threshold, str) and edge_threshold.endswith("%"):
        p = float(edge_threshold[:-1])
        return float(np.percentile(absW, p)) if absW.size else 0.0
    if edge_threshold is None:
        return 0.0
    return float(edge_threshold)


def plot_connectome_from_ccg(
    ccg_matrix: np.ndarray,
    ch_names_in_ccg_order: list[str],
    save_path: str,
    title: str = "EEG Channel Connectome",
    edge_threshold: str | float = "50%",
):
    """
    根据通道顺序 ch_names_in_ccg_order 对齐坐标，绘制并保存 connectome。
    - 解决 Nilearn 在 GlassBrain 上 colorbar 报错：关闭内置 colorbar，转而手动画色条
    - 色条范围与“通过阈值后真正被绘制的边”一致；若有正负权重，自动使用对称范围和双向色图
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 只保留坐标里存在的通道，并按 ch_names 顺序对齐
    keep_idx, keep_names = [], []
    for i, ch in enumerate(ch_names_in_ccg_order):
        if ch in CUSTOM_COORDS:
            keep_idx.append(i)
            keep_names.append(ch)
    A = np.asarray(ccg_matrix, dtype=float)[np.ix_(keep_idx, keep_idx)]
    pos = np.array([CUSTOM_COORDS[ch] for ch in keep_names], dtype=float)

    # 对称化 + 去自环
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)

    # 基于阈值后可见的边计算 vmin/vmax，保证色条和可见颜色一致
    thr = _parse_threshold(A, edge_threshold)
    mask = np.abs(A) >= thr
    if np.any(mask):
        vmin = float(A[mask].min())
        vmax = float(A[mask].max())
    else:
        # 没边可画时，给一个稳定范围，避免色条异常
        vmin = float(A.min()) if A.size else 0.0
        vmax = float(A.max()) if A.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

    # 自动选择色图与对称范围
    edge_cmap = "plasma"
    if vmin < 0 < vmax:
        a = max(abs(vmin), abs(vmax))
        vmin, vmax = -a, a
        edge_cmap = "RdBu_r"

    # 关键：禁用 Nilearn 自带 colorbar，避免 GlassBrainAxes.cmap 报错
    display = plot_connectome(
        A,
        pos,
        title=title,
        edge_threshold=edge_threshold,
        node_size=20,
        node_color="skyblue",
        edge_cmap=edge_cmap,
        edge_kwargs={"linewidth": 1.5, "alpha": 0.9},
        node_kwargs={"edgecolor": "black"},
        display_mode="ortho",
        colorbar=False,  # <-- 修复点：关闭内置色条
    )

    # 手动画色条（与可见边一致的范围）
    cmap = mpl.cm.get_cmap(edge_cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # 把色条挂到第一个切面的轴上
    fig = plt.gcf()
    ax_for_cbar = display.axes[0] if isinstance(display.axes, (list, tuple)) else None
    cbar = fig.colorbar(sm, ax=ax_for_cbar, fraction=0.046, pad=0.04)
    cbar.set_label("Edge weight")

    # 保存并关闭
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return display
