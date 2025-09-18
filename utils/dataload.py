import argparse
import logging
import os
import pickle
from collections import OrderedDict
from typing import Dict
import mne
import numpy as np
import scipy.io as scio
import torch
from scipy import signal
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from numpy import multiply


log = logging.getLogger(__name__)
log.setLevel("INFO")
logging.basicConfig(level=logging.INFO)


def scale_amplitude(data, factor):
    return multiply(data, factor)


def load_bcic(
    dataset_id: str = "2a",
    subject_id: int = 1,
    preprocessing_dict: Dict = None,
    verbose: str = "WARNING",
):
    dataset_name = "BNCI2014001" if dataset_id == "2a" else "BNCI2014004"
    dataset = MOABBDataset(dataset_name, subject_ids=[subject_id])

    # * pre-processing
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        Preprocessor(scale_amplitude, factor=1e6),
    ]

    # filtering or not
    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        preprocessors.append(
            Preprocessor("filter", l_freq=l_freq, h_freq=h_freq, verbose=verbose)
        )
    # resample or not
    if dataset.datasets[0].raw.info["sfreq"] != preprocessing_dict["sfreq"]:
        preprocessors.append(
            Preprocessor(
                "resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose
            ),
        )

    preprocess(dataset, preprocessors)

    # {'feet': 0, 'left_hand': 1, 'right_hand': 2, 'tongue': 3}
    if dataset_id == "2a":
        mapping = {
            "left_hand": 0,
            "right_hand": 1,
            "feet": 2,
            "tongue": 3,
        }

    elif dataset_id == "2b":
        mapping = {"left_hand": 0, "right_hand": 1}

    # create windows
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        mapping=mapping,
        preload=False,
    )

    # split the data
    splitted_ds = windows_dataset.split("session")
    X_ses, y_ses = [], []
    if dataset_id == "2a":
        train_dataset, test_dataset = (
            splitted_ds["0train"],
            splitted_ds["1test"],
        )
        # load the data
        X_ses.append(np.stack([sample[0] for sample in train_dataset], axis=0))
        X_ses.append(np.stack([sample[0] for sample in test_dataset], axis=0))
        y_ses.append(np.stack([sample[1] for sample in train_dataset], axis=0))
        y_ses.append(np.stack([sample[1] for sample in test_dataset], axis=0))

    elif dataset_id == "2b":
        # train_datasets = [splitted_ds[f"session_{session}"] for session in [0, 1, 2]]
        # test_datasets = [splitted_ds[f"session_{session}"] for session in [3, 4]]
        train_datasets = [splitted_ds[f"{session}train"] for session in [0, 1, 2]]
        test_datasets = [splitted_ds[f"{session}test"] for session in [3, 4]]

        # load the data
        for train_dataset in train_datasets:
            X_ses.append(np.stack([sample[0] for sample in train_dataset], axis=0))
            y_ses.append(np.stack([sample[1] for sample in train_dataset], axis=0))
        for test_dataset in test_datasets:
            X_ses.append(np.stack([sample[0] for sample in test_dataset], axis=0))
            y_ses.append(np.stack([sample[1] for sample in test_dataset], axis=0))

    return X_ses, y_ses


def load_bciciv2a_data_single_subject(subject_id):
    # subject_id = "/A0"+str(subject_id)

    preprocessing_2a = {
        "sfreq": 250,
        "low_cut": 0.5,
        "high_cut": 40,
        "start": 0,
        "stop": 0,
        "z_scale": False,
    }
    X, y = load_bcic("2a", subject_id, preprocessing_2a)
    return X[0], y[0], X[1], y[1]


# def load_bciciv2a_data_single_subject(filename, subject_id):
#     subject_id = "/A0"+str(subject_id)

#     filename = filename+subject_id
#     test_path = os.path.join(filename,'evaluation.mat')
#     train_path = os.path.join(filename,'training.mat')
#     train_data = scio.loadmat(train_path)
#     test_data = scio.loadmat(test_path)
#     train_X = train_data['EEG_data']
#     train_Y = train_data['label']-1
#     test_X = test_data['EEG_data']
#     test_Y = test_data['label']-1

#     train_X = torch.tensor(train_X, dtype=torch.float32).permute(2,0,1)
#     test_X = torch.tensor(test_X, dtype=torch.float32).permute(2,0,1)
#     train_Y = torch.tensor(train_Y, dtype=torch.int64).view(-1)
#     test_Y = torch.tensor(test_Y, dtype=torch.int64).view(-1)


#     b, a = signal.butter(5,[0.5,40],btype='bandpass',fs=250)
#     filteredtrain_signal = signal.lfilter(b,a,train_X)
#     filteredtest_signal = signal.lfilter(b,a,test_X)

#     filteredtest_signal = torch.tensor(filteredtest_signal,dtype=torch.float32)
#     filteredtrain_signal = torch.tensor(filteredtrain_signal,dtype=torch.float32)

#     return filteredtrain_signal,train_Y,filteredtest_signal,test_Y


def load_bciciv2a_data_subject(filename, subject_id):
    subject_id = "/A0" + str(subject_id)

    filename = filename + subject_id
    test_path = os.path.join(filename, "evaluation.mat")
    train_path = os.path.join(filename, "training.mat")
    train_data = scio.loadmat(train_path)
    test_data = scio.loadmat(test_path)
    train_X = train_data["EEG_data"]
    train_Y = train_data["label"] - 1
    test_X = test_data["EEG_data"]
    test_Y = test_data["label"] - 1

    train_X = torch.tensor(train_X, dtype=torch.float32).permute(2, 0, 1)
    test_X = torch.tensor(test_X, dtype=torch.float32).permute(2, 0, 1)
    train_Y = torch.tensor(train_Y, dtype=torch.int64).view(-1)
    test_Y = torch.tensor(test_Y, dtype=torch.int64).view(-1)

    b, a = signal.butter(5, [0.5, 40], btype="bandpass", fs=250)
    filteredtrain_signal = signal.lfilter(b, a, train_X)
    filteredtest_signal = signal.lfilter(b, a, test_X)

    filteredtest_signal = torch.tensor(filteredtest_signal, dtype=torch.float32)
    filteredtrain_signal = torch.tensor(filteredtrain_signal, dtype=torch.float32)
    data = torch.cat([filteredtrain_signal, filteredtest_signal], dim=0)
    label = torch.cat([train_Y, test_Y], dim=0)

    return data, label


def load_physionet_data_single_subject_binary(filename, subject_id):
    filename = filename + str(subject_id) + ".mat"
    data = scio.loadmat(filename)
    data_X = data["data"]
    data_Y = data["label"]
    data_X = torch.tensor(data_X, dtype=torch.float32)
    data_Y = torch.tensor(data_Y, dtype=torch.int64).view(-1)
    random_indices = np.random.permutation(data_X.shape[0])
    data_X = data_X[random_indices]
    data_Y = data_Y[random_indices]

    indices = np.where((data_Y == 0) | (data_Y == 1))

    data_X = data_X[indices]
    data_Y = data_Y[indices]

    indices = np.where(data_Y == 3)
    data_Y[indices] = 1

    b, a = signal.butter(5, [0.5, 40], btype="bandpass", fs=160)
    filtered_signal = signal.lfilter(b, a, data_X)
    filtered_signal = torch.tensor(filtered_signal, dtype=torch.float32)
    # return filtered_signal,data_Y
    return data_X, data_Y


def load_physionet_data_single_subject(filename, subject_id):
    filename = filename + str(subject_id) + ".mat"
    data = scio.loadmat(filename)
    data_X = data["data"]
    data_Y = data["label"]
    data_X = torch.tensor(data_X, dtype=torch.float32)
    data_Y = torch.tensor(data_Y, dtype=torch.int64).view(-1)
    random_indices = np.random.permutation(data_X.shape[0])
    data_X = data_X[random_indices]
    data_Y = data_Y[random_indices]
    b, a = signal.butter(5, [0.5, 40], btype="bandpass", fs=160)
    filtered_signal = signal.lfilter(b, a, data_X)
    filtered_signal = torch.tensor(filtered_signal, dtype=torch.float32)
    return filtered_signal, data_Y


def load_bci2b_data_single_subject(filename, subject_id):
    # path = /disk1/wangxuhui/data/processed_data_EEG_MOTOR_MOVEMENT
    train_path = filename + "/A0" + str(subject_id) + "E.mat"
    test_path = filename + "/A0" + str(subject_id) + "T.mat"
    train_data = scio.loadmat(train_path)
    test_data = scio.loadmat(test_path)
    train_data_X = train_data["EEG_data"]
    train_data_Y = train_data["label"]
    test_data_X = test_data["EEG_data"]
    test_data_Y = test_data["label"]

    train_data_X = torch.tensor(train_data_X, dtype=torch.float32).permute(2, 0, 1)
    train_data_Y = torch.tensor(train_data_Y, dtype=torch.int64).view(-1) - 1
    test_data_X = torch.tensor(test_data_X, dtype=torch.float32).permute(2, 0, 1)
    test_data_Y = torch.tensor(test_data_Y, dtype=torch.int64).view(-1) - 1

    b, a = signal.butter(5, [0.5, 40], btype="bandpass", fs=250)

    filtered_train_signal = signal.lfilter(b, a, train_data_X)
    filtered_test_signal = signal.lfilter(b, a, test_data_X)

    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)

    return filtered_train_signal, train_data_Y, filtered_test_signal, test_data_Y


def load_labdata_data_single_subject(filename, subject_id):
    # path = /disk1/wangxuhui/data/processed_data_EEG_MOTOR_MOVEMENT

    # train_path = filename + "/A0" + str(subject_id) + "E.mat"
    # test_path = filename + "/A0" + str(subject_id) + "T.mat"
    # train_path = '/disk1/wangxuhui/data/labdata/20240624-YCX-4class-total.mat'
    # test_path = '/disk1/wangxuhui/data/labdata/20240703-YCX-4class-total.mat'
    train_path = "/disk1/wangxuhui/data/labdata/ZK/20240619-ZK-4class-total.mat"
    test_path = "/disk1/wangxuhui/data/labdata/ZK/20240627-ZK-4class-total.mat"
    train_data = scio.loadmat(train_path)
    test_data = scio.loadmat(test_path)
    train_data_X = train_data["data"]
    train_data_Y = train_data["label"]
    test_data_X = test_data["data"]
    test_data_Y = test_data["label"]

    # downsampled_signal_train = np.zeros((288,64,1000))
    # downsampled_signal_test = np.zeros((288,64,1000))

    # for i in range(train_data_X.shape[0]):
    #     for j in range(train_data_X.shape[1]):
    #         downsampled_signal_train[i, j, :] = signal.resample(train_data_X[i, j, :], 1000)
    #         downsampled_signal_test[i, j, :] = signal.resample(test_data_X[i, j, :], 1000)

    # train_data_X = downsampled_signal_train
    # test_data_X = downsampled_signal_test

    train_data_X = torch.tensor(train_data_X, dtype=torch.float32)
    train_data_Y = torch.tensor(train_data_Y, dtype=torch.int64).view(-1)
    test_data_X = torch.tensor(test_data_X, dtype=torch.float32)
    test_data_Y = torch.tensor(test_data_Y, dtype=torch.int64).view(-1)

    b, a = signal.butter(5, [0.5, 40], btype="bandpass", fs=250)

    filtered_train_signal = signal.lfilter(b, a, train_data_X)
    filtered_test_signal = signal.lfilter(b, a, test_data_X)

    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)

    return filtered_train_signal, train_data_Y, filtered_test_signal, test_data_Y
