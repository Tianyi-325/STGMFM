import os
import argparse
import torch
from collections import OrderedDict
import logging
import pickle
import scipy.io as scio
from scipy import signal
import numpy as np
import mne

log = logging.getLogger(__name__)
log.setLevel('INFO')
logging.basicConfig(level=logging.INFO)

def load_bciciv2a_data_single_subject(filename, subject_id):
    subject_id = "/A0"+str(subject_id)

    filename = filename+subject_id
    test_path = os.path.join(filename,'evaluation.mat')
    train_path = os.path.join(filename,'training.mat')
    train_data = scio.loadmat(train_path)
    test_data = scio.loadmat(test_path)
    train_X = train_data['EEG_data']
    train_Y = train_data['label']-1
    test_X = test_data['EEG_data']
    test_Y = test_data['label']-1

    train_X = torch.tensor(train_X, dtype=torch.float32).permute(2,0,1)
    test_X = torch.tensor(test_X, dtype=torch.float32).permute(2,0,1)
    train_Y = torch.tensor(train_Y, dtype=torch.int64).view(-1)
    test_Y = torch.tensor(test_Y, dtype=torch.int64).view(-1)

    
    b, a = signal.butter(5,[0.5,40],btype='bandpass',fs=250)
    filteredtrain_signal = signal.lfilter(b,a,train_X)
    filteredtest_signal = signal.lfilter(b,a,test_X)

    filteredtest_signal = torch.tensor(filteredtest_signal,dtype=torch.float32)
    filteredtrain_signal = torch.tensor(filteredtrain_signal,dtype=torch.float32)

    return filteredtrain_signal,train_Y,filteredtest_signal,test_Y

def load_bciciv2a_data_subject(filename, subject_id):
    subject_id = "/A0"+str(subject_id)

    filename = filename+subject_id
    test_path = os.path.join(filename,'evaluation.mat')
    train_path = os.path.join(filename,'training.mat')
    train_data = scio.loadmat(train_path)
    test_data = scio.loadmat(test_path)
    train_X = train_data['EEG_data']
    train_Y = train_data['label']-1
    test_X = test_data['EEG_data']
    test_Y = test_data['label']-1

    train_X = torch.tensor(train_X, dtype=torch.float32).permute(2,0,1)
    test_X = torch.tensor(test_X, dtype=torch.float32).permute(2,0,1)
    train_Y = torch.tensor(train_Y, dtype=torch.int64).view(-1)
    test_Y = torch.tensor(test_Y, dtype=torch.int64).view(-1)

    
    b, a = signal.butter(5,[0.5,40],btype='bandpass',fs=250)
    filteredtrain_signal = signal.lfilter(b,a,train_X)
    filteredtest_signal = signal.lfilter(b,a,test_X)

    filteredtest_signal = torch.tensor(filteredtest_signal,dtype=torch.float32)
    filteredtrain_signal = torch.tensor(filteredtrain_signal,dtype=torch.float32)
    data = torch.cat([filteredtrain_signal,filteredtest_signal],dim=0)
    label = torch.cat([train_Y,test_Y],dim=0)

    return data,label

def load_physionet_data_single_subject_binary(filename,subject_id):
    filename = filename+str(subject_id)+".mat"
    data = scio.loadmat(filename)
    data_X = data['data']
    data_Y = data['label']
    data_X = torch.tensor(data_X,dtype=torch.float32)
    data_Y = torch.tensor(data_Y,dtype=torch.int64).view(-1)
    random_indices = np.random.permutation(data_X.shape[0])
    data_X = data_X[random_indices]
    data_Y = data_Y[random_indices]

    indices = np.where((data_Y == 0) | (data_Y == 1) )

    data_X = data_X[indices]
    data_Y = data_Y[indices]

    indices = np.where(data_Y==3)
    data_Y[indices] = 1

    b,a = signal.butter(5,[0.5,40],btype='bandpass',fs=160)
    filtered_signal = signal.lfilter(b,a,data_X)
    filtered_signal = torch.tensor(filtered_signal,dtype=torch.float32)
    # return filtered_signal,data_Y
    return data_X,data_Y




def load_physionet_data_single_subject(filename,subject_id):
    filename = filename+str(subject_id)+".mat"
    data = scio.loadmat(filename)
    data_X = data['data']
    data_Y = data['label']
    data_X = torch.tensor(data_X,dtype=torch.float32)
    data_Y = torch.tensor(data_Y,dtype=torch.int64).view(-1)
    random_indices = np.random.permutation(data_X.shape[0])
    data_X = data_X[random_indices]
    data_Y = data_Y[random_indices]
    b,a = signal.butter(5,[0.5,40],btype='bandpass',fs=160)
    filtered_signal = signal.lfilter(b,a,data_X)
    filtered_signal = torch.tensor(filtered_signal,dtype=torch.float32)
    return filtered_signal,data_Y




def load_bci2b_data_single_subject(filename, subject_id):
    # path = /disk1/wangxuhui/data/processed_data_EEG_MOTOR_MOVEMENT
    train_path = filename + "/A0" + str(subject_id) + "E.mat"
    test_path = filename + "/A0" + str(subject_id) + "T.mat"
    train_data = scio.loadmat(train_path)
    test_data = scio.loadmat(test_path)
    train_data_X = train_data['EEG_data']
    train_data_Y = train_data['label']
    test_data_X = test_data['EEG_data']
    test_data_Y = test_data['label']

    train_data_X = torch.tensor(train_data_X, dtype=torch.float32).permute(2, 0, 1)
    train_data_Y = torch.tensor(train_data_Y, dtype=torch.int64).view(-1)-1
    test_data_X = torch.tensor(test_data_X, dtype=torch.float32).permute(2, 0, 1)
    test_data_Y = torch.tensor(test_data_Y, dtype=torch.int64).view(-1)-1

    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)

    filtered_train_signal = signal.lfilter(b, a, train_data_X)
    filtered_test_signal = signal.lfilter(b, a, test_data_X)

    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)

    return filtered_train_signal,train_data_Y,filtered_test_signal,test_data_Y



def load_labdata_data_single_subject(filename, subject_id):
    # path = /disk1/wangxuhui/data/processed_data_EEG_MOTOR_MOVEMENT

    # train_path = filename + "/A0" + str(subject_id) + "E.mat"
    # test_path = filename + "/A0" + str(subject_id) + "T.mat"
    # train_path = '/disk1/wangxuhui/data/labdata/20240624-YCX-4class-total.mat'
    # test_path = '/disk1/wangxuhui/data/labdata/20240703-YCX-4class-total.mat'
    train_path = '/disk1/wangxuhui/data/labdata/ZK/20240619-ZK-4class-total.mat'
    test_path = '/disk1/wangxuhui/data/labdata/ZK/20240627-ZK-4class-total.mat'
    train_data = scio.loadmat(train_path)
    test_data = scio.loadmat(test_path)
    train_data_X = train_data['data']
    train_data_Y = train_data['label']
    test_data_X = test_data['data']
    test_data_Y = test_data['label']

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

    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)

    filtered_train_signal = signal.lfilter(b, a, train_data_X)
    filtered_test_signal = signal.lfilter(b, a, test_data_X)

    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)

    return filtered_train_signal,train_data_Y,filtered_test_signal,test_data_Y


def get_class_epochs(class_id, tmp, events, event_id, t_min, t_max, base_line=None):
    # selected_channels = ["FZ", "C4", "CPZ", "CZ", "C3", "PZ"]
    selected_channels = None
    epochs = mne.Epochs(tmp, events, event_id, tmin=t_min, tmax=t_max, baseline=base_line, preload=True,
                        event_repeated='drop', picks=selected_channels)
    class_epochs = epochs[class_id]

    channel_unit_dict = tmp._orig_units
    for key in channel_unit_dict:
        unit = channel_unit_dict[key]
        break
    if unit == "µV":
        scale = 1e6
    elif unit == "mV":
        scale = 1e3
    else:
        scale = 1

    class_data = class_epochs.get_data()
    class_data = class_data * scale
    return class_data


def get_ob3000_eeg_data(file_path, num_class, bandpass_filter=None, window_l=(-0.5, 4)):

    raw = mne.io.read_raw_bdf(file_path, preload=True)
    if bandpass_filter:
        tmp = raw.copy().filter(l_freq=bandpass_filter[0], h_freq=bandpass_filter[1],
                                fir_design='firwin', skip_by_annotation='edge')
    else:
        tmp = raw
    sampling_rate = tmp.info['sfreq']
    events, event_id = mne.events_from_annotations(tmp)

    rest_id = 'relax_start'
    rest_data = get_class_epochs(rest_id, tmp, events, event_id, t_min=0, t_max=60-(1/sampling_rate))

    window_size = int((window_l[1] - window_l[0]) * sampling_rate)
    step_size = int((rest_data.shape[-1] - window_size) / 24)

    rest_data_win = []
    for start in range(0, rest_data.shape[-1] - window_size + 1, step_size):
        rest_data_win.append(rest_data[0, :, start:start + window_size])

    rest_data = np.stack(rest_data_win)
    rest_label = np.zeros(rest_data.shape[0], dtype=int)
    # rest_data = eeg_data_process(rest_data, sampling_rate, normalize, use_ea, combined_filter)

    if num_class == 2:
        action_id = ['action1_start', 'action2_start']
        action_data = get_class_epochs(action_id, tmp, events, event_id,
                                       t_min=window_l[0], t_max=window_l[1]-(1/sampling_rate))

        action_label = np.ones(action_data.shape[0], dtype=int)

        # action_data = eeg_data_process(action_data, sampling_rate, normalize, use_ea, combined_filter)

        data = np.concatenate((action_data, rest_data), axis=0)
        label = np.concatenate((action_label, rest_label), axis=0)

    elif num_class == 3:
        action1_id = 'action1_start'
        action1_data = get_class_epochs(action1_id, tmp, events, event_id,
                                        t_min=window_l[0], t_max=window_l[1]-(1/sampling_rate))
        action2_id = 'action2_start'
        action2_data = get_class_epochs(action2_id, tmp, events, event_id,
                                        t_min=window_l[0], t_max=window_l[1]-(1/sampling_rate))

        action1_label = np.ones(action1_data.shape[0], dtype=int) * 1
        action2_label = np.ones(action2_data.shape[0], dtype=int) * 2

        # action1_data = eeg_data_process(action1_data, sampling_rate, normalize, use_ea, combined_filter)
        # action2_data = eeg_data_process(action2_data, sampling_rate, normalize, use_ea, combined_filter)

        data = np.concatenate((action1_data, action2_data, rest_data), axis=0)
        label = np.concatenate((action1_label, action2_label, rest_label), axis=0)

    return data, label