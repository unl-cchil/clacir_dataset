import os
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
from flirt.eda.feature_calculation import __cvx_eda
from flirt.stats.common import get_stats


def split_set(signal, split_size, jump_size):
    """Split a signal into specified number of chunks.  Handles uneven splits. Handled by a generator.
    Very useful for pulling apart a dataaset to calculate features since you don't need the separated dataset.
    Encapsulate result in list: list(split_set(signal, split_size)) if you don't want to use the generator
    Source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Parameters
    :param signal: ndarray
        Signal to be split
    :param split_size: int
        Size of chunks
    :return: generator
        Returns an iterable object to perform this function real-time
    """
    for i in range(0, len(signal), int(jump_size)):
        yield signal[i:i + int(split_size)]


def get_e4_features(signal, signal_type):
    """Processing pipeline per signal, pass in a slice of the signal that corresponds to the window.
    Specify the signal that is passed in.
    Parameters
    :param signal: ndarray
        Windowed slice of the original signal
    :param signal_type: str
        Signal type passed in.  BVP, Accelerometer, Temperature, and EDA supported
    :return: list
        List of the features extracted
    """
    features = []
    try:
        if signal_type == 'BVP':
            features.extend(float(v) for v in get_stats(signal).values())
        elif signal_type == 'ACC':
            features.extend(float(v) for v in get_stats(signal).values())
        elif signal_type == 'TMP':
            features.extend(float(v) for v in get_stats(signal).values())
        elif signal_type == 'HRV':
            features.extend(float(v) for v in get_stats(signal).values())
        elif signal_type == 'EDA':
            r, t = __cvx_eda(signal, 1/4)
            features.extend(float(v) for v in get_stats(np.ravel(t)).values())
            features.extend(float(v) for v in get_stats(np.ravel(r)).values())
        return features
    except Exception:
        return features


def remove_nan(dataset):
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    for i in range(0, len(dataset)):
        inf_indx = np.isinf(dataset[i])
        dataset[i][inf_indx] = np.nan
        dataset[i] = imp.fit_transform(dataset[i])


def pose_control_v_condition_binary(data, labels):
    # 0 = N/A | 1 = Precondition | 2 = HAI | 3 = Control | 4 = Postcondition
    # Control is considered neutral, precondition is cognitive load/stress, HAI is amusement
    # Pose the problem of differentiating postcondition with label 0 being control and label 1 is HAI
    trimmed_dataset, trimmed_labels = [], []
    for x, y in zip(data, labels):
        sel_indxs = [i for i, n in enumerate(y) if n in [2, 3]]
        trimmed_dataset.append(x[sel_indxs])
        control = np.where(y == 3)[0]
        if len(control) > 0:
            trimmed_labels.append(np.array([[0]] * len(trimmed_dataset[-1])))
        else:
            trimmed_labels.append(np.array([[1]] * len(trimmed_dataset[-1])))
    return trimmed_dataset, trimmed_labels


def pose_intervention_binary(data, labels):
    # 0 = N/A | 1 = Precondition | 2 = HAI | 3 = Control | 4 = Postcondition | 5 = Debrief
    # Control is considered neutral, precondition is cognitive load/stress, HAI is amusement
    # Pose the problem of differentiating postcondition with label 0 being control and label 1 is HAI
    trimmed_dataset, trimmed_labels = [], []
    for x, y in zip(data, labels):
        del_indxs = [i for i, n in enumerate(y) if n not in [4]]
        trimmed_dataset.append(np.delete(x, del_indxs, 0))
        control = np.where(y == 3)[0]
        if len(control) > 0:
            trimmed_labels.append(np.array([[0]] * len(trimmed_dataset[-1])))
        else:
            trimmed_labels.append(np.array([[1]] * len(trimmed_dataset[-1])))
    return trimmed_dataset, trimmed_labels


def read_tag_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        readline = f.readline().strip().split(',')[0]
        if readline:
            data.append(float(readline))
    return data


def read_e4_file(filepath):
    read_timestamp = False
    read_sample_rate = False
    data, timestamp, sample_rates = [[], [], []], [0, 0, 0], [0, 0, 0]
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not read_timestamp:
                lines = line.strip().split(',')
                for index, l in enumerate(lines):
                    timestamp[index] = float(l.strip()) if l.strip().split('.')[0].isnumeric() else l
                read_timestamp = True
            elif not read_sample_rate:
                lines = line.strip().split(',')
                for index, l in enumerate(lines):
                    sample_rates[index] = float(l.strip()) if l.strip().split('.')[0].isnumeric() else l
                read_sample_rate = True
            else:
                lines = line.strip().split(',')
                for index, l in enumerate(lines):
                    data[index].append(float(l))
    return data, timestamp, sample_rates


def load_clacir_dataset(filepath):
    pd.options.mode.chained_assignment = None
    affect_data = pd.read_csv(os.path.join(filepath, "affect_data.csv"))
    survey_data = pd.read_csv(os.path.join(filepath, "thayer_stevens_2021_data1.csv"))
    protocol_times = pd.read_csv(os.path.join(filepath, "hr_data_times.csv"))
    expt1_subjects = [f for f in os.listdir(os.path.join(filepath, "E4_expt1")) if
                      not os.path.exists(os.path.join(filepath, "E4_expt1", f, "NA.txt"))]
    expt2_subjects = [f for f in os.listdir(os.path.join(filepath, "E4_expt2")) if
                      not os.path.exists(os.path.join(filepath, "E4_expt2", f, "NA.txt"))]
    expt_subject_data = [os.path.join(filepath, "E4_expt1", f) for f in expt1_subjects if f.isnumeric()] + \
                        [os.path.join(filepath, "E4_expt2", f) for f in expt2_subjects if f.isnumeric()]

    expt1_subjects = [int(f) for f in expt1_subjects if f.isnumeric()]
    expt2_subjects = [int(f) for f in expt2_subjects if f.isnumeric()]

    clacir_subject_data = pd.DataFrame()
    for subject in expt1_subjects + expt2_subjects:
        subject_affect = affect_data.loc[affect_data['participant'] == subject]
        subject_times = protocol_times.loc[protocol_times['participant'] == subject]
        subject_survey = survey_data.loc[survey_data['participant'] == subject]
        subject_times = subject_times.drop(['experiment', 'participant', 'condition', 'date'], axis=1)
        subject_affect = subject_affect.drop(['experiment', 'participant', 'condition'], axis=1)

        if subject_affect.empty or subject_times.empty or subject_survey.empty:
            continue
        else:
            for k, v in zip(subject_times.keys(), subject_times.values[0]):
                subject_survey[k] = v
            for k, v in zip(subject_affect.keys(), subject_affect.values[0]):
                subject_survey[k] = v
            clacir_subject_data = pd.concat([clacir_subject_data, subject_survey])
    clacir_subject_data = clacir_subject_data.set_index("experiment")
    clacir_subject_data.to_excel(os.path.join(filepath, "clacir_data.xlsx"))
    clacir_subject_data.reset_index(drop=False, inplace=True)

    clacir_dataset = {}
    for index, row in clacir_subject_data.iterrows():
        subject = os.path.join(filepath, f"E4_expt{row['experiment']}", f"{row['participant']:03d}")
        print(f"Processing participant {row['participant']} in {subject}...")
        if not os.path.exists(os.path.join(subject, "data.pkl")):
            acc_data, acc_timestamp, acc_sample_rate = read_e4_file(os.path.join(subject, "ACC.csv"))
            bvp_data, bvp_timestamp, bvp_sample_rate = read_e4_file(os.path.join(subject, "BVP.csv"))
            eda_data, eda_timestamp, eda_sample_rate = read_e4_file(os.path.join(subject, "EDA.csv"))
            hrv_data, hrv_timestamp, hrv_sample_rate = read_e4_file(os.path.join(subject, "HR.csv"))
            hrv_data[0] = [0] * int(hrv_timestamp[0] - eda_timestamp[0]) + hrv_data[0]
            tmp_data, tmp_timestamp, tmp_sample_rate = read_e4_file(os.path.join(subject, "TEMP.csv"))
            tag_data = read_tag_file(os.path.join(subject, "tags.csv"))

            e4_start_time = datetime.utcfromtimestamp(eda_timestamp[0])
            if datetime.strptime("2019-03-10", "%Y-%m-%d") > e4_start_time > datetime.strptime("2018-11-4", "%Y-%m-%d"):
                central = timedelta(hours=6)
            else:
                central = timedelta(hours=5)
            e4_start_time = datetime.utcfromtimestamp(eda_timestamp[0]) - central
            precondition_start_time = datetime.strptime(row['date'] + " " + row['t_1'], "%Y-%m-%dT00:00:00Z %H:%M:%S")
            intervention_start_time = datetime.strptime(row['date'] + " " + row['t_2'], "%Y-%m-%dT00:00:00Z %H:%M:%S")
            postcondition_start_time = datetime.strptime(row['date'] + " " + row['t_3'], "%Y-%m-%dT00:00:00Z %H:%M:%S")
            debrief_start_time = datetime.strptime(row['date'] + " " + row['t_4'], "%Y-%m-%dT00:00:00Z %H:%M:%S")
            end_time = datetime.strptime(row['date'] + " " + row['t_5'], "%Y-%m-%dT00:00:00Z %H:%M:%S")

            protocol_time = (end_time - e4_start_time).seconds
            labels = [0] * protocol_time
            label_time = e4_start_time
            label_step = timedelta(seconds=1)

            for i in range(0, len(labels)):
                # Before precondition
                if label_time <= precondition_start_time:
                    labels[i] = 0
                # During precondition
                elif precondition_start_time < label_time <= intervention_start_time:
                    labels[i] = 1
                # During condition
                elif intervention_start_time < label_time <= postcondition_start_time:
                    if row['condition'] == "hai":
                        labels[i] = 2
                    else:
                        labels[i] = 3
                # During post condition
                elif postcondition_start_time < label_time <= debrief_start_time:
                    labels[i] = 4
                # After post condition
                elif debrief_start_time < label_time:
                    labels[i] = 5
                label_time += label_step
            print(f"\tLabels: {np.unique(labels)}")
            print(f"\t{protocol_time} seconds")
            subject_data = {
                "ACX": acc_data[0],
                "ACX SR": acc_sample_rate[0],
                "ACY": acc_data[1],
                "ACY SR": acc_sample_rate[1],
                "ACZ": acc_data[2],
                "ACZ SR": acc_sample_rate[2],
                "BVP": bvp_data[0],
                "BVP SR": bvp_sample_rate[0],
                "EDA": eda_data[0],
                "EDA SR": eda_sample_rate[0],
                "HRV": hrv_data[0],
                "HRV SR": hrv_sample_rate[0],
                "TMP": tmp_data[0],
                "TMP SR": tmp_sample_rate[0],
                "TAG": tag_data,
                "TSS": acc_timestamp + bvp_timestamp + eda_timestamp + hrv_timestamp + tmp_timestamp,
                "LBL": labels,
                "LBL SR": 1 / 60
            }
            for k in row.keys():
                subject_data[k] = row[k]
            with open(os.path.join(subject, "data.pkl"), 'wb') as f:
                pickle.dump(subject_data, f)
        else:
            with open(os.path.join(subject, "data.pkl"), 'rb') as f:
                subject_data = pickle.load(f)
        clacir_dataset[row['participant']] = subject_data

    # Window size in seconds
    window_size = 60
    for participant in clacir_dataset:
        bvp_features, eda_features, acx_features, acy_features, acz_features, tmp_features, hrv_features, lbl_features = [], [], [], [], [], [], [], []
        print("Processing subject number:", clacir_dataset[participant]['participant'])
        if len(np.unique(clacir_dataset[participant]['LBL'])) == 1:
            raise Exception()
        # Signal window sizes
        bvp_window_size = int(clacir_dataset[participant]['BVP SR'] * window_size)
        acc_window_size = int(clacir_dataset[participant]['ACX SR'] * window_size)
        eda_window_size = int(clacir_dataset[participant]['EDA SR'] * window_size)
        tmp_window_size = int(clacir_dataset[participant]['TMP SR'] * window_size)
        hrv_window_size = int(clacir_dataset[participant]['HRV SR'] * window_size)
        lbl_window_size = int(clacir_dataset[participant]['LBL SR'] * window_size)
        # Iterate through the dataset types
        bvp_generator = split_set(clacir_dataset[participant]['BVP'], bvp_window_size, 64.0)
        eda_generator = split_set(clacir_dataset[participant]['EDA'], eda_window_size, 4.0)
        acx_generator = split_set(clacir_dataset[participant]['ACX'], acc_window_size, 32.0)
        acy_generator = split_set(clacir_dataset[participant]['ACY'], acc_window_size, 32.0)
        acz_generator = split_set(clacir_dataset[participant]['ACZ'], acc_window_size, 32.0)
        tmp_generator = split_set(clacir_dataset[participant]['TMP'], tmp_window_size, 4.0)
        hrv_generator = split_set(clacir_dataset[participant]['HRV'], hrv_window_size, 1.0)
        lbl_generator = split_set(clacir_dataset[participant]['LBL'], lbl_window_size, 1.0)
        # Split windows of the datastreams
        for bvp, eda, acx, acy, acz, tmp, hrv, lbl in zip(bvp_generator, eda_generator, acx_generator, acy_generator,
                                                          acz_generator, tmp_generator, hrv_generator, lbl_generator):
            if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(tmp) < tmp_window_size:
                continue
            else:
                features = get_e4_features(bvp, 'BVP')
                if features:
                    bvp_features.append(features)
                else:
                    continue
                eda_features.append(get_e4_features(eda, 'EDA'))
                acx_features.append(get_e4_features(acx, 'ACC'))
                acy_features.append(get_e4_features(acy, 'ACC'))
                acz_features.append(get_e4_features(acz, 'ACC'))
                tmp_features.append(get_e4_features(tmp, 'TMP'))
                hrv_features.append(get_e4_features(hrv, 'HRV'))
                lbl_features.append(lbl)
        clacir_dataset[participant]['BVP Features'] = bvp_features
        clacir_dataset[participant]['EDA Features'] = eda_features
        clacir_dataset[participant]['ACX Features'] = acx_features
        clacir_dataset[participant]['ACY Features'] = acy_features
        clacir_dataset[participant]['ACZ Features'] = acz_features
        clacir_dataset[participant]['TMP Features'] = tmp_features
        clacir_dataset[participant]['HRV Features'] = hrv_features
        clacir_dataset[participant]['LBL Features'] = lbl_features
        print(f"\t{np.unique(clacir_dataset[participant]['LBL'])} -> {np.unique(lbl_features)}")
        print(f"\t{len(clacir_dataset[participant]['LBL'])} -> {len(lbl_features)} seconds")
    with open(os.path.join('datasets', 'clacir_processed', 'clacir.pkl'), 'wb') as f:
        pickle.dump(clacir_dataset, f)
    return clacir_dataset


def generate_dataset(datastreams=None, dataset_name="clacir", panas_threshold=None):
    if datastreams is None:
        datastreams = [True, True, True, True, True]
    print("Collecting cLACIr dataset...")
    dataset_path = f'datasets/clacir_processed/{dataset_name}.pkl'
    if os.path.exists(dataset_path):
        print("Pickled cLACIr dataset exists...\n")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = load_clacir_dataset('datasets/clacir_raw')

    subjects, labels = [], []
    for participant in dataset:
        subject_data, subject_labels = [], []
        if panas_threshold is not None:
            if dataset[participant]['panas_pos_diff'] < panas_threshold and dataset[participant]['condition'] == 'hai':
                print(f"Skipping {participant} with PANAS Pos diff {dataset[participant]['panas_pos_diff']}")
                continue
        for bvp, eda, acx, acy, acz, tmp, hrv, lbl in zip(dataset[participant]['BVP Features'],
                                                          dataset[participant]['EDA Features'],
                                                          dataset[participant]['ACX Features'],
                                                          dataset[participant]['ACY Features'],
                                                          dataset[participant]['ACZ Features'],
                                                          dataset[participant]['TMP Features'],
                                                          dataset[participant]['HRV Features'],
                                                          dataset[participant]['LBL Features']):
            subject_labels.append([lbl])
            feature_list = []
            if datastreams[0]:
                feature_list.extend(bvp)
            if datastreams[1]:
                feature_list.extend(eda)
            if datastreams[2]:
                feature_list.extend(acx)
                feature_list.extend(acy)
                feature_list.extend(acz)
            if datastreams[3]:
                feature_list.extend(tmp)
            if datastreams[4]:
                feature_list.extend(hrv)
            subject_data.append(np.array(feature_list))
        if len(np.unique(subject_labels)) > 3:
            subjects.append(np.array(subject_data))
            labels.append(np.array(subject_labels))
    remove_nan(subjects)
    ap_dataset, ap_labels = pose_intervention_binary(subjects, labels)
    int_dataset, int_labels = pose_control_v_condition_binary(subjects, labels)
    return (ap_dataset, ap_labels), (int_dataset, int_labels)
