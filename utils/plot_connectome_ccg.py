# utils/plot_connectome_ccg.py
import os
import numpy as np
from nilearn.plotting import plot_connectome

# Cordinates for EEG channels, used for plotting connectomes
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


def plot_connectome_from_ccg(
    ccg_matrix: np.ndarray,
    ch_names_in_ccg_order: list[str],
    save_path: str,
    title: str = "EEG Channel Connectome",
    edge_threshold: str | float = "50%",
):
    """
    根据通道顺序 ch_names_in_ccg_order 对齐坐标，绘制并保存 connectome。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Only keep channels that have coordinates
    keep_idx, keep_names = [], []
    for i, ch in enumerate(ch_names_in_ccg_order):
        if ch in CUSTOM_COORDS:
            keep_idx.append(i)
            keep_names.append(ch)
    A = ccg_matrix[np.ix_(keep_idx, keep_idx)]
    pos = np.array([CUSTOM_COORDS[ch] for ch in keep_names])

    display = plot_connectome(
        A,
        pos,
        title=title,
        edge_threshold=edge_threshold,
        node_size=20,
        node_color="skyblue",
        edge_cmap="plasma",
        edge_kwargs={"linewidth": 1.5, "alpha": 0.9},
        node_kwargs={"edgecolor": "black"},
        display_mode="ortho",
        colorbar=True,
    )
    display.savefig(save_path, dpi=200)
    display.close()
