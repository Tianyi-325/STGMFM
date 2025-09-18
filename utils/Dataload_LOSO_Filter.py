# utils/Dataload_LOSO.py
import os
from typing import List, Tuple

import numpy as np
import torch
import mne

from utils.load_ob3000_bdf_data import get_ob3000_eeg_data

# 屏蔽的两位受试者（不区分大小写）
EXCLUDED_PERSONS = {"richard", "dushipan"}

# 固定带通滤波参数（按你的要求：4–40 Hz；采样率如为 250 Hz）
_BANDPASS_L_FREQ = 4.0
_BANDPASS_H_FREQ = 40.0
_BANDPASS_SFREQ = 250.0  # 如你的数据采样率不同，请据实修改


# --------------------------------- 小工具 ---------------------------------
def _list_persons(base_dir: str) -> List[str]:
    """列出（按字典序）过滤后的受试者文件夹名。"""
    persons = [
        p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))
    ]
    persons.sort()
    persons = [p for p in persons if p.lower() not in EXCLUDED_PERSONS]
    return persons


def _collect_bdf_files_under(path: str) -> List[str]:
    """收集某个目录（不递归）的 .bdf 文件，按文件名排序。"""
    if not os.path.isdir(path):
        return []
    files = [
        os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".bdf")
    ]
    files.sort()
    return files


def _collect_person_bdfs(base_dir: str, person: str) -> List[str]:
    """
    收集该受试者的全部 .bdf：
    - base_dir/person/session1/*.bdf
    - base_dir/person/session2/*.bdf
    - 兼容可能直接放在 base_dir/person/*.bdf 的情况
    """
    root = os.path.join(base_dir, person)
    files = []
    files += _collect_bdf_files_under(os.path.join(root, "session1"))
    files += _collect_bdf_files_under(os.path.join(root, "session2"))
    files += _collect_bdf_files_under(root)  # 兼容无 session 子目录的情况
    # 去重并保持相对稳定的顺序
    files = sorted(list(dict.fromkeys(files)))
    return files


def _load_many(files: List[str], num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    datas, labels = [], []
    for f in files:
        X, y = get_ob3000_eeg_data(f, num_classes)
        datas.append(X)
        labels.append(y)
    if len(datas) == 0:
        return None, None
    X = np.concatenate(datas, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def _bandpass_4_40(X: np.ndarray) -> np.ndarray:
    """
    对形状 (N_trials, N_channels, N_times) 的数组沿时间维做 4–40 Hz 带通。
    固定采样率 _BANDPASS_SFREQ。
    """
    # mne 推荐双精度计算，之后再转回 float32
    Xf = mne.filter.filter_data(
        X.astype(np.float64, copy=False),
        sfreq=_BANDPASS_SFREQ,
        l_freq=_BANDPASS_L_FREQ,
        h_freq=_BANDPASS_H_FREQ,
        method="iir",
        verbose=False,
    )
    return Xf.astype(np.float32, copy=False)


# --------------------------------- 主函数 ---------------------------------
def create_dataset_for_cross_validation(
    base_dir: str, num_classes: int = 3, test_person_idx: int = 0
):
    """
    LOSO 划分（屏蔽 EXCLUDED_PERSONS），并在拼接完成后对 train/test **固定执行 4–40Hz 带通滤波**：
      - 从过滤后的受试者列表里选择 1 人为测试，其余为训练
      - 训练集：其他所有人的全部 .bdf（含 session1+session2 或根目录 .bdf）
      - 测试集：被留出这 1 人的全部 .bdf
      - 返回的 person_folders 仅包含未屏蔽的受试者名（按字典序）

    返回：
      X_train[Tensor(float32)], y_train[Tensor(long)],
      X_test[Tensor(float32)],  y_test[Tensor(long)],
      person_folders[List[str]]
    """
    # 1) 受试者列表（已屏蔽）
    person_folders = _list_persons(base_dir)
    if len(person_folders) == 0:
        raise RuntimeError("No valid subjects found after exclusion.")

    # 2) 选择测试受试者
    sid = test_person_idx % len(person_folders)
    test_person = person_folders[sid]

    # 3) 收集文件
    # 训练：其余 18 人的全部 .bdf
    train_files: List[str] = []
    for p in person_folders:
        if p == test_person:
            continue
        train_files.extend(_collect_person_bdfs(base_dir, p))

    # 测试：被留出受试者的全部 .bdf
    test_files = _collect_person_bdfs(base_dir, test_person)

    if len(train_files) == 0 or len(test_files) == 0:
        raise RuntimeError(
            f"[{test_person}] split found no files: train={len(train_files)}, test={len(test_files)}. "
            "Check directory structure and file extensions (.bdf)."
        )

    # 4) 读取并拼接（numpy）
    X_train_np, y_train_np = _load_many(train_files, num_classes)
    X_test_np, y_test_np = _load_many(test_files, num_classes)
    if X_train_np is None or X_test_np is None:
        raise RuntimeError("Empty data after loading; please verify file lists.")

    # 5) ★ 固定执行 4–40 Hz 带通滤波（axis=-1）
    X_train_np = _bandpass_4_40(X_train_np)
    X_test_np = _bandpass_4_40(X_test_np)

    # 6) 转 tensor（与你的训练脚本保持一致）
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    # 7) （可选调试）设置 EEG_DEBUG_SPLIT=1 打印一次摘要
    if os.environ.get("EEG_DEBUG_SPLIT", "0") == "1":

        def _brief(xs: List[str]):
            base = [os.path.basename(x) for x in xs[:3]]
            return base + (["..."] if len(xs) > 3 else [])

        print(
            f"[DEBUG][LOSO+BP 4-40Hz] test_person={test_person} "
            f"| train_files={len(train_files)} { _brief(train_files) } "
            f"| test_files={len(test_files)} { _brief(test_files) } "
            f"| shapes: Xtr={X_train.shape}, Xte={X_test.shape}"
        )

    return X_train, y_train, X_test, y_test, person_folders


# ----------------- 自测（可选） -----------------
if __name__ == "__main__":
    base_dir = "/data/raw_data/eegmi_OB3000"
    Xtr, ytr, Xte, yte, pf = create_dataset_for_cross_validation(
        base_dir, num_classes=3, test_person_idx=0
    )
    print("train:", Xtr.shape, ytr.shape, "test:", Xte.shape, yte.shape)
    print("subjects:", pf)
