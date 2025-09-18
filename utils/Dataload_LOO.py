import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from load_ob3000_bdf_data import get_ob3000_eeg_data

# from utils.load_ob3000_bdf_data import get_ob3000_eeg_data


def get_person_bdf_files(base_dir, person_idx):
    """
    返回 base_dir 下第 person_idx 号人的所有 .bdf 文件路径列表。
    """
    # 按字典序列出所有人的文件夹并选中指定 idx
    persons = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )
    selected_person = persons[person_idx]
    person_path = os.path.join(base_dir, selected_person)

    bdf_files = []
    for session in ("session1", "session2"):
        session_dir = os.path.join(person_path, session)
        if os.path.isdir(session_dir):
            for fn in os.listdir(session_dir):
                if fn.lower().endswith(".bdf"):
                    bdf_files.append(os.path.join(session_dir, fn))

    return selected_person, bdf_files


def create_dataset_for_person(
    base_dir, person_idx=0, num_classes=3, test_size=0.2, random_state=42
):
    """
    对同一个人（由 person_idx 指定）：
      1. 读取其所有 .bdf 数据；
      2. 将该人的所有样本按 test_size 比例拆分为训练集/测试集；
      3. 返回 PyTorch Tensor 格式的 X_train, y_train, X_test, y_test 以及该人的文件夹名。
    """
    # 1. 找到该人的所有 .bdf 文件
    person_name, bdf_files = get_person_bdf_files(base_dir, person_idx)

    # 2. 读入数据并拼接
    all_data, all_labels = [], []
    for bdf in bdf_files:
        data, labels = get_ob3000_eeg_data(bdf, num_classes)
        all_data.append(data)
        all_labels.append(labels)
    all_data = np.concatenate(
        all_data, axis=0
    )  # shape=(n_samples, n_channels, n_timepoints)
    all_labels = np.concatenate(all_labels, axis=0)  # shape=(n_samples,)

    # 3. 按比例拆分
    X_train, X_test, y_train, y_test = train_test_split(
        all_data,
        all_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=all_labels,  # 保证每个类别比例一致（可选）
    )

    # 4. 转成 PyTorch Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_test, y_test, person_name


# —— 示例：依次对每个人单独拆分并测试 —— #
if __name__ == "__main__":
    base_dir = "/data/raw_data/eegmi_OB3000"
    # 首先统计一下有多少人
    persons = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )
    print(f"总共有 {len(persons)} 个人：{persons}")

    # 例如，我们只想处理第 2 号人（索引 1）
    idx = 0
    X_tr, y_tr, X_te, y_te, person = create_dataset_for_person(
        base_dir, person_idx=idx, num_classes=3, test_size=0.2
    )
    print(f"Selected person: {person}")
    print(f"→ 训练集样本：{X_tr.shape[0]}，   测试集样本：{X_te.shape[0]}")
