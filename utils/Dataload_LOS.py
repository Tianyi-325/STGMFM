# utils/Dataload_LOSO_CrossSession.py
import os
import numpy as np
import torch
from typing import List, Tuple

from utils.load_ob3000_bdf_data import get_ob3000_eeg_data

# from load_ob3000_bdf_data import get_ob3000_eeg_data

# 屏蔽两位受试者（不区分大小写）
EXCLUDED_PERSONS = {"richard", "dushipan"}


# --------------------------------- 小工具 ---------------------------------
def _list_persons(base_dir: str) -> List[str]:
    """列出（按字典序）过滤后的受试者文件夹名。"""
    persons = [
        p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))
    ]
    persons.sort()
    persons = [p for p in persons if p.lower() not in EXCLUDED_PERSONS]
    return persons


def _has_session(base_dir: str, person: str, session: str) -> bool:
    ses_dir = os.path.join(base_dir, person, session)
    return os.path.isdir(ses_dir) and any(
        f.lower().endswith(".bdf") for f in os.listdir(ses_dir)
    )


def _collect_bdf_files(base_dir: str, person: str, session: str) -> List[str]:
    ses_dir = os.path.join(base_dir, person, session)
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
        return None, None  # 由上层抛错
    X = np.concatenate(datas, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


# --------------------------------- 主函数 ---------------------------------
def create_dataset_for_cross_validation(
    base_dir: str, num_classes: int = 3, test_person_idx: int = 0
):
    """
    LOSO × Cross-Session 混合划分（屏蔽 EXCLUDED_PERSONS）：
      - 选择 1 位目标被试 S_test（必须同时具备 session1 与 session2）
      - 训练集 = 其他 18 位受试者（session1+session2 全部） + S_test 的 session1
      - 测试集 = S_test 的 session2

    返回：
      X_train[Tensor(float32)], y_train[Tensor(long)],
      X_test[Tensor(float32)],  y_test[Tensor(long)],
      person_folders[List[str]]  # 可用作姓名列表与索引映射（仅包含“同时拥有 s1&s2”的受试者）
    """
    # 1) 构建“候选被试列表”：过滤后且 **同时拥有 session1 与 session2** 的受试者
    persons_all = _list_persons(base_dir)
    person_folders = [
        p
        for p in persons_all
        if _has_session(base_dir, p, "session1")
        and _has_session(base_dir, p, "session2")
    ]
    if len(person_folders) == 0:
        raise RuntimeError(
            "No valid subjects with both session1 and session2 found after exclusion."
        )
    # 目标被试索引容错
    sid = test_person_idx % len(person_folders)
    test_person = person_folders[sid]

    # 2) 训练文件：
    #    - 其余 18 位受试者（s1+s2）
    #    - 加上 test_person 的 session1
    train_files: List[str] = []
    for p in person_folders:
        if p == test_person:
            continue
        train_files.extend(_collect_bdf_files(base_dir, p, "session1"))
        train_files.extend(_collect_bdf_files(base_dir, p, "session2"))
    # 目标被试的 session1
    train_files.extend(_collect_bdf_files(base_dir, test_person, "session1"))

    # 3) 测试文件：目标被试的 session2
    test_files = _collect_bdf_files(base_dir, test_person, "session2")

    if len(train_files) == 0 or len(test_files) == 0:
        raise RuntimeError(
            f"[{test_person}] split found no files: train={len(train_files)}, test={len(test_files)}. "
            "Check directory structure and file extensions (.bdf)."
        )

    # 4) 读取数据并拼接
    X_train_np, y_train_np = _load_many(train_files, num_classes)
    X_test_np, y_test_np = _load_many(test_files, num_classes)
    if X_train_np is None or X_test_np is None:
        raise RuntimeError("Empty data after loading; please verify file lists.")

    # 5) 转为 Tensor（与现有训练脚本对齐）
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    # 6) （可选调试）设置 EEG_DEBUG_SPLIT=1 打印一次摘要
    if os.environ.get("EEG_DEBUG_SPLIT", "0") == "1":

        def _brief(xs: List[str]):
            return [os.path.basename(x) for x in xs[:3]] + (
                ["..."] if len(xs) > 3 else []
            )

        print(
            f"[DEBUG][Hybrid LOSO×CS] test_person={test_person} "
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
