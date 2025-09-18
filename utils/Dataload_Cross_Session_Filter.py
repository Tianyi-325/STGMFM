# utils/Dataload_Cross_Session_Filter.py
import os
from typing import List, Tuple

import numpy as np
import torch
import mne

from utils.load_ob3000_bdf_data import get_ob3000_eeg_data

# 屏蔽名单（不区分大小写）
EXCLUDED_PERSONS = {"richard", "dushipan"}

# 固定带通参数
_BANDPASS_L_FREQ = 4.0
_BANDPASS_H_FREQ = 40.0
_BANDPASS_SFREQ = 250.0  # 如你的真实采样率不同，请改这里


# ------------------ 小工具 ------------------
def _list_persons(base_dir: str) -> List[str]:
    persons = [
        p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))
    ]
    persons.sort()
    persons = [p for p in persons if p.lower() not in EXCLUDED_PERSONS]
    return persons


def _collect_bdf_files(base_dir: str, person: str, session_name: str) -> List[str]:
    ses_dir = os.path.join(base_dir, person, session_name)
    if not os.path.isdir(ses_dir):
        return []
    files = [
        os.path.join(ses_dir, f)
        for f in os.listdir(ses_dir)
        if f.lower().endswith(".bdf")
    ]
    files.sort()
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
    """对 (N, C, T) 的时间维做 4–40Hz 带通；兼容旧版 mne（不传 axis）。"""
    X64 = X.astype(np.float64, copy=False)
    Xf = mne.filter.filter_data(
        X64,
        sfreq=_BANDPASS_SFREQ,
        l_freq=_BANDPASS_L_FREQ,
        h_freq=_BANDPASS_H_FREQ,
        method="iir",
        verbose=False,
    )
    return Xf.astype(np.float32, copy=False)


# ------------------ 主函数 ------------------
def create_dataset_for_cross_validation(
    base_dir: str, num_classes: int = 3, test_person_idx: int = 0
):
    """
    Cross-Session（同被试）：session1 -> Train, session2 -> Test。
    读取后对 Train/Test **固定执行 4–40Hz 带通**。
    返回：X_train, y_train, X_test, y_test, person_folders
    """
    # 1) 可用受试者（保证同时有 s1 与 s2）
    persons_all = _list_persons(base_dir)
    person_folders: List[str] = []
    for p in persons_all:
        has_s1 = len(_collect_bdf_files(base_dir, p, "session1")) > 0
        has_s2 = len(_collect_bdf_files(base_dir, p, "session2")) > 0
        if has_s1 and has_s2:
            person_folders.append(p)
    if len(person_folders) == 0:
        raise RuntimeError("No valid subjects with both session1 and session2 found.")

    # 2) 选择当前被试
    sid = test_person_idx % len(person_folders)
    person = person_folders[sid]

    # 3) s1 -> train, s2 -> test
    train_files = _collect_bdf_files(base_dir, person, "session1")
    test_files = _collect_bdf_files(base_dir, person, "session2")
    if len(train_files) == 0 or len(test_files) == 0:
        raise RuntimeError(
            f"[{person}] Missing bdf files: session1={len(train_files)}, session2={len(test_files)}."
        )

    # 4) 读取 & 拼接（numpy）
    X_train_np, y_train_np = _load_many(train_files, num_classes)
    X_test_np, y_test_np = _load_many(test_files, num_classes)
    if X_train_np is None or X_test_np is None:
        raise RuntimeError(f"Empty data after loading for subject [{person}].")

    # 5) ★ 带通滤波
    X_train_np = _bandpass_4_40(X_train_np)
    X_test_np = _bandpass_4_40(X_test_np)

    # 6) 转 tensor
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    # 7) 调试输出（可选）
    if os.environ.get("EEG_DEBUG_SPLIT", "0") == "1":

        def _brief(xs: List[str]):
            return [os.path.basename(x) for x in xs[:3]] + (
                ["..."] if len(xs) > 3 else []
            )

        print(
            f"[DEBUG][CS + BP 4-40Hz] Subject={person} | "
            f"Train s1 files={len(train_files)} {_brief(train_files)} | "
            f"Test s2 files={len(test_files)} {_brief(test_files)} | "
            f"Shapes: Xtr={X_train.shape}, Xte={X_test.shape}"
        )

    return X_train, y_train, X_test, y_test, person_folders


# ----------------------- 自测（可选） -----------------------
if __name__ == "__main__":
    base_dir = "/data/raw_data/eegmi_OB3000"
    Xtr, ytr, Xte, yte, pf = create_dataset_for_cross_validation(
        base_dir, num_classes=3, test_person_idx=0
    )
    print("train:", Xtr.shape, ytr.shape, "test:", Xte.shape, yte.shape)
    print("persons:", pf)
