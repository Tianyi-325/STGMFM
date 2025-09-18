# utils/Dataload_Cross_Session.py
import os
import numpy as np
import torch

from utils.load_ob3000_bdf_data import get_ob3000_eeg_data

# 可选：排除名单（不区分大小写），按需保留/删除
EXCLUDED_PERSONS = {"richard", "dushipan"}

# 预期目录结构：
# base_dir/
#   ├─ PersonA/
#   │    ├─ session1/*.bdf
#   │    └─ session2/*.bdf
#   └─ PersonB/
#        ├─ session1/*.bdf
#        └─ session2/*.bdf


def _list_persons(base_dir):
    persons = [
        p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))
    ]
    persons.sort()
    persons = [p for p in persons if p.lower() not in EXCLUDED_PERSONS]
    return persons


def _collect_bdf_files(base_dir, person, session_name):
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


def _load_many(files, num_classes):
    datas, labels = [], []
    for f in files:
        X, y = get_ob3000_eeg_data(f, num_classes)  # 由你现有的解析器提供
        datas.append(X)
        labels.append(y)
    if len(datas) == 0:
        return None, None
    X = np.concatenate(datas, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def create_dataset_for_cross_validation(base_dir, num_classes=3, test_person_idx=0):
    """
    【严格的“同被试跨会话”划分】
    - 仅使用 test_person_idx 对应“这一个人”的数据；
    - 固定使用该人的 session1 作为训练集，session2 作为测试集；
    - 不混入任何其他人的样本。

    返回：X_train[tensor], y_train[tensor], X_test[tensor], y_test[tensor], person_folders[list[str]]
    其中 person_folders 为可用受试者（两个会话均存在）的有序列表，用于上层脚本显示姓名。
    """
    # 1) 构建可用受试者列表（两个 session 都存在）
    persons_all = _list_persons(base_dir)
    person_folders = []
    for p in persons_all:
        has_s1 = len(_collect_bdf_files(base_dir, p, "session1")) > 0
        has_s2 = len(_collect_bdf_files(base_dir, p, "session2")) > 0
        if has_s1 and has_s2:
            person_folders.append(p)

    if len(person_folders) == 0:
        raise RuntimeError("No valid subjects with both session1 and session2 found.")

    # 2) 选择当前被试（安全取模，保证索引不越界）
    sid = test_person_idx % len(person_folders)
    person = person_folders[sid]

    # 3) 固定：该人 session1 → Train；该人 session2 → Test
    train_files = _collect_bdf_files(base_dir, person, "session1")
    test_files = _collect_bdf_files(base_dir, person, "session2")

    if len(train_files) == 0 or len(test_files) == 0:
        raise RuntimeError(
            f"[{person}] Missing bdf files: session1={len(train_files)}, session2={len(test_files)}."
        )

    # 4) 读取
    X_train_np, y_train_np = _load_many(train_files, num_classes)
    X_test_np, y_test_np = _load_many(test_files, num_classes)
    if X_train_np is None or X_test_np is None:
        raise RuntimeError(f"Empty data after loading for subject [{person}].")

    # 5) 转 tensor（你的上层训练代码期望 torch.Tensor）
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    # 6) （可选调试）设置环境变量 EEG_DEBUG_SPLIT=1 可打印划分摘要
    if os.environ.get("EEG_DEBUG_SPLIT", "0") == "1":

        def _brief(xs):
            return [os.path.basename(x) for x in xs[:3]] + (
                ["..."] if len(xs) > 3 else []
            )

        print(
            f"[DEBUG] Subject={person} | Train=session1 files={len(train_files)} {_brief(train_files)} | "
            f"Test=session2 files={len(test_files)} {_brief(test_files)} | "
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
