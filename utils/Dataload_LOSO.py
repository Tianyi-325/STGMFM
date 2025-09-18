import os
import numpy as np
import mne
import torch

from utils.load_ob3000_bdf_data import get_ob3000_eeg_data

# from load_ob3000_bdf_data import get_ob3000_eeg_data


# 放在文件顶部附近（import 之后任意位置也可）
EXCLUDED_PERSONS = {"richard", "dushipan"}  # 不区分大小写


# def get_all_bdf_files(base_dir):
#     """
#     遍历数据文件夹，返回所有人的 bdf 文件路径。
#     """
#     bdf_files = []
#     for person_folder in os.listdir(base_dir):
#         person_folder_path = os.path.join(base_dir, person_folder)
#         if os.path.isdir(person_folder_path):
#             for session in ["session1", "session2"]:
#                 session_folder_path = os.path.join(person_folder_path, session)
#                 if os.path.isdir(session_folder_path):
#                     for file in os.listdir(session_folder_path):
#                         if file.endswith(".bdf"):
#                             bdf_files.append(os.path.join(session_folder_path, file))
#     return bdf_files


# def create_dataset_for_cross_validation(base_dir, num_classes=3, test_person_idx=0):
#     """
#     为交叉验证创建训练和测试数据集。
#     指定 test_person_idx 的人的所有 bdf 文件作为测试数据，其余作为训练数据。
#     """
#     # 获取所有人的文件夹
#     person_folders = [
#         f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
#     ]
#     person_folders.sort()  # 保证顺序一致

#     # 获取测试集和训练集的文件夹
#     test_person = person_folders[test_person_idx]
#     train_persons = [f for i, f in enumerate(person_folders) if i != test_person_idx]

#     # 获取测试集文件
#     test_person_files = []
#     test_person_path = os.path.join(base_dir, test_person)
#     for session in ["session1", "session2"]:
#         session_folder_path = os.path.join(test_person_path, session)
#         if os.path.isdir(session_folder_path):
#             for file in os.listdir(session_folder_path):
#                 if file.endswith(".bdf"):
#                     test_person_files.append(os.path.join(session_folder_path, file))

#     # 获取训练集文件
#     train_person_files = []
#     for person in train_persons:
#         person_path = os.path.join(base_dir, person)
#         for session in ["session1", "session2"]:
#             session_folder_path = os.path.join(person_path, session)
#             if os.path.isdir(session_folder_path):
#                 for file in os.listdir(session_folder_path):
#                     if file.endswith(".bdf"):
#                         train_person_files.append(
#                             os.path.join(session_folder_path, file)
#                         )

#     # 获取测试集数据
#     test_data, test_label = [], []
#     for file in test_person_files:
#         data, label = get_ob3000_eeg_data(file, num_classes)
#         test_data.append(data)
#         test_label.append(label)
#     test_data = np.concatenate(test_data, axis=0)
#     test_label = np.concatenate(test_label, axis=0)

#     # 获取训练集数据
#     train_data, train_label = [], []
#     for file in train_person_files:
#         data, label = get_ob3000_eeg_data(file, num_classes)
#         train_data.append(data)
#         train_label.append(label)
#     train_data = np.concatenate(train_data, axis=0)
#     train_label = np.concatenate(train_label, axis=0)
#     train_data = torch.tensor(train_data, dtype=torch.float32)
#     train_label = torch.tensor(train_label, dtype=torch.long)
#     test_data = torch.tensor(test_data, dtype=torch.float32)
#     test_label = torch.tensor(test_label, dtype=torch.long)

#     # 返回格式： X_train, y_train, X_test, y_test
#     # return train_data, train_label, test_data, test_label
#     return train_data, train_label, test_data, test_label, person_folders


def get_all_bdf_files(base_dir):
    """
    遍历数据文件夹，返回所有人的 bdf 文件路径。
    —— 已过滤 EXCLUDED_PERSONS 中的个体。
    """
    bdf_files = []
    for person_folder in os.listdir(base_dir):
        # 屏蔽名单（不区分大小写）
        if person_folder.lower() in EXCLUDED_PERSONS:
            continue
        person_folder_path = os.path.join(base_dir, person_folder)
        if os.path.isdir(person_folder_path):
            for session in ["session1", "session2"]:
                session_folder_path = os.path.join(person_folder_path, session)
                if os.path.isdir(session_folder_path):
                    for file in os.listdir(session_folder_path):
                        if file.endswith(".bdf"):
                            bdf_files.append(os.path.join(session_folder_path, file))
    return bdf_files


def create_dataset_for_cross_validation(base_dir, num_classes=3, test_person_idx=0):
    """
    为交叉验证创建训练和测试数据集。
    指定 test_person_idx 的人的所有 bdf 文件作为测试数据，其余作为训练数据。
    —— 已过滤 EXCLUDED_PERSONS 中的个体；同时对 test_person_idx 做取模容错。
    """
    # 1) 获取并过滤受试者文件夹
    all_persons = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]
    all_persons.sort()  # 保证顺序一致
    person_folders = [p for p in all_persons if p.lower() not in EXCLUDED_PERSONS]

    if len(person_folders) == 0:
        raise RuntimeError(
            "No subjects left after exclusion. Check EXCLUDED_PERSONS or data path."
        )

    # 2) 取模防越界（若外部仍按原总人数循环也可正常运行）
    test_person_idx = test_person_idx % len(person_folders)

    # 3) 划分训练/测试受试者
    test_person = person_folders[test_person_idx]
    train_persons = [f for i, f in enumerate(person_folders) if i != test_person_idx]

    # 4) 收集测试集文件
    test_person_files = []
    test_person_path = os.path.join(base_dir, test_person)
    for session in ["session1", "session2"]:
        session_folder_path = os.path.join(test_person_path, session)
        if os.path.isdir(session_folder_path):
            for file in os.listdir(session_folder_path):
                if file.endswith(".bdf"):
                    test_person_files.append(os.path.join(session_folder_path, file))

    # 5) 收集训练集文件
    train_person_files = []
    for person in train_persons:
        person_path = os.path.join(base_dir, person)
        for session in ["session1", "session2"]:
            session_folder_path = os.path.join(person_path, session)
            if os.path.isdir(session_folder_path):
                for file in os.listdir(session_folder_path):
                    if file.endswith(".bdf"):
                        train_person_files.append(
                            os.path.join(session_folder_path, file)
                        )

    # 6) 读取数据
    test_data, test_label = [], []
    for file in test_person_files:
        data, label = get_ob3000_eeg_data(file, num_classes)
        test_data.append(data)
        test_label.append(label)
    test_data = np.concatenate(test_data, axis=0)
    test_label = np.concatenate(test_label, axis=0)

    train_data, train_label = [], []
    for file in train_person_files:
        data, label = get_ob3000_eeg_data(file, num_classes)
        train_data.append(data)
        train_label.append(label)
    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)

    # 7) 转 tensor 并返回（保持原返回格式）
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_label = torch.tensor(test_label, dtype=torch.long)

    return train_data, train_label, test_data, test_label, person_folders


# 示例使用
base_dir = "/data/raw_data/eegmi_OB3000"
X_train, y_train, X_test, y_test, pf = create_dataset_for_cross_validation(
    base_dir, num_classes=3, test_person_idx=0  # 可修改为任意编号
)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")
# print(X_test)
# print(X_train)
# print(y_test)
# print(y_train)
print(pf)
