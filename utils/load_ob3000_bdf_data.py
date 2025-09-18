import mne
import numpy as np


# event_id:
# {
#   'action1_countdown': 1, 'action1_start': 2,
#   'action2_countdown': 3, 'action2_start': 4,
#   'eye_blink_left': 5,  'eye_blink_left_countdown': 6,
#   'eye_blink_right': 7, 'eye_blink_right_countdown': 8,
#   'recording_end': 9, 'recording_start': 10,
#   'relax_start': 11
# }


def get_class_epochs(class_id, tmp, events, event_id, t_min, t_max, base_line=None):
    # selected_channels = ["FZ", "C4", "CPZ", "CZ", "C3", "PZ"]
    selected_channels = None
    epochs = mne.Epochs(
        tmp,
        events,
        event_id,
        tmin=t_min,
        tmax=t_max,
        baseline=base_line,
        preload=True,
        event_repeated="drop",
        picks=selected_channels,
    )
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
    raw.rename_channels({"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"})  # 更新通道
    if bandpass_filter:
        tmp = raw.copy().filter(
            l_freq=bandpass_filter[0],
            h_freq=bandpass_filter[1],
            fir_design="firwin",
            skip_by_annotation="edge",
        )
    else:
        tmp = raw
    sampling_rate = tmp.info["sfreq"]
    events, event_id = mne.events_from_annotations(tmp)

    rest_id = "relax_start"
    rest_data = get_class_epochs(
        rest_id, tmp, events, event_id, t_min=0, t_max=60 - (1 / sampling_rate)
    )

    window_size = int((window_l[1] - window_l[0]) * sampling_rate)
    step_size = int((rest_data.shape[-1] - window_size) / 24)

    rest_data_win = []
    for start in range(0, rest_data.shape[-1] - window_size + 1, step_size):
        rest_data_win.append(rest_data[0, :, start : start + window_size])

    rest_data = np.stack(rest_data_win)
    rest_label = np.zeros(rest_data.shape[0], dtype=int)
    # rest_data = eeg_data_process(rest_data, sampling_rate, normalize, use_ea, combined_filter)

    if num_class == 2:
        action_id = ["action1_start", "action2_start"]
        action_data = get_class_epochs(
            action_id,
            tmp,
            events,
            event_id,
            t_min=window_l[0],
            t_max=window_l[1] - (1 / sampling_rate),
        )

        action_label = np.ones(action_data.shape[0], dtype=int)

        # action_data = eeg_data_process(action_data, sampling_rate, normalize, use_ea, combined_filter)

        data = np.concatenate((action_data, rest_data), axis=0)
        label = np.concatenate((action_label, rest_label), axis=0)

    elif num_class == 3:
        action1_id = "action1_start"
        action1_data = get_class_epochs(
            action1_id,
            tmp,
            events,
            event_id,
            t_min=window_l[0],
            t_max=window_l[1] - (1 / sampling_rate),
        )
        action2_id = "action2_start"
        action2_data = get_class_epochs(
            action2_id,
            tmp,
            events,
            event_id,
            t_min=window_l[0],
            t_max=window_l[1] - (1 / sampling_rate),
        )

        action1_label = np.ones(action1_data.shape[0], dtype=int) * 1
        action2_label = np.ones(action2_data.shape[0], dtype=int) * 2

        # action1_data = eeg_data_process(action1_data, sampling_rate, normalize, use_ea, combined_filter)
        # action2_data = eeg_data_process(action2_data, sampling_rate, normalize, use_ea, combined_filter)

        data = np.concatenate((action1_data, action2_data, rest_data), axis=0)
        label = np.concatenate((action1_label, action2_label, rest_label), axis=0)

    return data, label


if __name__ == "__main__":
    data_path = r"/data/raw_data/eegmi_OB3000/Adila/session1/group1_data.bdf"

    d, l = get_ob3000_eeg_data(data_path, 3)
    print(d)

    print("success")
