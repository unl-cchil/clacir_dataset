import copy
import csv
import glob
import os
import pickle
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, normalize
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pytz
import signal_processing as sp
from inferencing import get_e4_features


def get_e4_labels(subject_labels):
    """Averages the labels of the window
    Parameters
    :param subject_labels: ndarray
        Labels to be averaged
    :return: float
        Averaged label value
    """
    return np.around(np.average(subject_labels))


def load_csv_dataset(parent_dir):
    datasets_from_dir = []
    data = []
    labels = []

    for filename in glob.iglob(parent_dir + '**/*', recursive=True):
        if filename.endswith(".csv"):
            datasets_from_dir.append(filename)

    if datasets_from_dir:
        for filename in datasets_from_dir:
            print(f"Processing {filename}...")
            rows = []
            with open(filename, "r") as f:
                reader = csv.reader(f, delimiter=",")
                for row in reader:
                    rows.append(np.array(list(map(float, row))))
                labels.append(rows[-1])
                data.append([
                    np.column_stack((rows[1][1:], rows[2][1:], rows[3][1:])),
                    rows[4][1:],
                    rows[5][1:],
                    rows[6][1:]
                ])
    else:
        raise ValueError("No cLASIr datasets found!")
    return data, labels


def normalize_dataset(dataset):
    normalized_dataset = []
    for x in dataset:
        normalized_dataset.append(normalize(x))
    return normalized_dataset


def remove_nan(dataset):
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    for i in range(0, len(dataset)):
        inf_indx = np.isinf(dataset[i])
        dataset[i][inf_indx] = np.nan
        dataset[i] = imp.fit_transform(dataset[i])


def trim_data(dataset, labels):
    # 0 = N/A | 1 = Precondition | 2 = HAI | 3 = Control | 4 = Postcondition
    # Control is considered neutral, precondition is cognitive load/stress, HAI is amusement
    trimmed_dataset, trimmed_labels = [], []
    for x, y in zip(dataset, labels):
        del_indxs = np.hstack((np.where(y == 0.0)[0], np.where(y >= 4)[0]))
        trimmed_labels.append(np.delete(y, del_indxs, 0))
        trimmed_dataset.append(np.delete(x, del_indxs, 0))
        trimmed_labels[-1][trimmed_labels[-1] == 3] = 0
    return trimmed_dataset, trimmed_labels


def binarize_dataset(dataset, labels):
    # 0 = N/A | 1 = Precondition | 2 = HAI | 3 = Control | 4 = Postcondition
    trimmed_dataset, trimmed_labels = [], []
    for x, y in zip(dataset, labels):
        del_indxs = [i for i, n in enumerate(y) if n not in [1, 4]]
        trimmed_dataset.append(np.delete(x, del_indxs, 0))
        trimmed_labels.append(np.delete(y, del_indxs, 0))
        trimmed_labels[-1][trimmed_labels[-1] == 4] = 0
    return trimmed_dataset, trimmed_labels


def pose_active_passive(dataset, labels):
    # 0 = Control | 1 = Precondition | 2 = HAI
    trimmed_dataset, trimmed_labels = [], []
    for x, y in zip(dataset, labels):
        del_indxs = np.where(y == 1)[0]
        trimmed_labels.append(np.delete(y, del_indxs, 0))
        trimmed_dataset.append(np.delete(x, del_indxs, 0))
        trimmed_labels[-1][trimmed_labels[-1] == 2] = 1
    return trimmed_dataset, trimmed_labels


def pose_control_v_condition_binary(dataset, labels):
    # 0 = N/A | 1 = Precondition | 2 = HAI | 3 = Control | 4 = Postcondition
    # Control is considered neutral, precondition is cognitive load/stress, HAI is amusement
    # Pose the problem of differentiating postcondition with label 0 being control and label 1 is HAI
    trimmed_dataset, trimmed_labels = [], []
    for x, y in zip(dataset, labels):
        sel_indxs = [i for i, n in enumerate(y) if n in [2, 3]]
        trimmed_dataset.append(x[sel_indxs])
        control = np.where(y == 3)[0]
        if len(control) > 0:
            trimmed_labels.append(np.array([[0]] * len(trimmed_dataset[-1])))
        else:
            trimmed_labels.append(np.array([[1]] * len(trimmed_dataset[-1])))
    return trimmed_dataset, trimmed_labels


def pose_intervention_binary(dataset, labels):
    # 0 = N/A | 1 = Precondition | 2 = HAI | 3 = Control | 4 = Postcondition
    # Control is considered neutral, precondition is cognitive load/stress, HAI is amusement
    # Pose the problem of differentiating postcondition with label 0 being control and label 1 is HAI
    trimmed_dataset, trimmed_labels = [], []
    for x, y in zip(dataset, labels):
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
    data, timestamp = [[], [], []], [[], [], []]
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not read_timestamp:
                lines = line.strip().split(',')
                for index, l in enumerate(lines):
                    timestamp[index] = float(l) if l.split('.')[0].isnumeric() else l
                read_timestamp = True
            else:
                lines = line.strip().split(',')
                for index, l in enumerate(lines):
                    data[index].append(float(l))
    return data, timestamp


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
        print(f"Processing participant {row['participant']} in {index}...")
        subject = expt_subject_data[index]
        acc_data, acc_timestamp = read_e4_file(os.path.join(subject, "ACC.csv"))
        bvp_data, bvp_timestamp = read_e4_file(os.path.join(subject, "BVP.csv"))
        eda_data, eda_timestamp = read_e4_file(os.path.join(subject, "EDA.csv"))
        hrv_data, hrv_timestamp = read_e4_file(os.path.join(subject, "HR.csv"))
        hrv_data[0] = [0] * int(hrv_timestamp[0] - eda_timestamp[0]) + hrv_data[0]
        tmp_data, tmp_timestamp = read_e4_file(os.path.join(subject, "TEMP.csv"))
        tag_data = read_tag_file(os.path.join(subject, "tags.csv"))

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

        subject_data = {
            "ACC X": acc_data[0],
            "ACC Y": acc_data[1],
            "ACC Z": acc_data[2],
            "BVP": bvp_data[0],
            "EDA": eda_data[0],
            "HRV": hrv_data[0],
            "TMP": tmp_data[0],
            "TAG": tag_data,
            "Timestamps": acc_timestamp + bvp_timestamp + eda_timestamp + hrv_timestamp + tmp_timestamp,
            "Labels": labels
        }
        for k in row.keys():
            subject_data[k] = row[k]
        with open(os.path.join(subject, "data.pkl"), 'wb') as f:
            pickle.dump(subject_data, f)
        clacir_dataset[row['participant']] = subject_data

    window_size = 60
    for participant in clacir_dataset:
        # Signal window sizes
        bvp_window_size = 64.0 * window_size
        acc_window_size = 32.0 * window_size
        eda_window_size = 4.0 * window_size
        temp_window_size = 4.0 * window_size
        label_window_size = window_size
        # Iterate through the dataset types
        print("Beginning sample processing...")
        for subject in range(0, len(data)):
            print("Processing subject number:", subject)
            bvp_generator = sp.split_set(data[subject][1], bvp_window_size, 64.0 / 4)
            eda_generator = sp.split_set(data[subject][2], eda_window_size, 4.0 / 4)
            acc_generator = sp.split_set(data[subject][0], acc_window_size, 32.0 / 4)
            temp_generator = sp.split_set(data[subject][3], temp_window_size, 4.0 / 4)
            label_generator = sp.split_set(labels[subject], label_window_size, 1)
            for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator,
                                                  label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    if datastreams[0]:
                        features = get_e4_features(bvp, 'BVP')
                        if features:
                            window_data.extend(features)
                        else:
                            continue
                    if datastreams[1]:
                        window_data.extend(get_e4_features(eda, 'EDA'))
                    if datastreams[2]:
                        window_data.extend(get_e4_features(acc, 'ACC'))
                    if datastreams[3]:
                        window_data.extend(get_e4_features(temp, 'TEMP'))
                    window_label = label[int((len(label) / 2.0) + 0.5) - 1]

    with open(os.path.join('datasets', 'clacir_raw', 'clacir_dataset.pkl'), 'wb') as f:
        pickle.dump(clacir_dataset, f)
    return clacir_dataset


def generate_dataset(window_size, write_pickle=True, datastreams=None, dataset_name="clacir"):
    if datastreams is None:
        datastreams = [True, True, True, True]
    print("Collecting cLACIr dataset...")
    dataset_path = f'datasets/clacir_processed/{dataset_name}.pkl'
    if os.path.exists(dataset_path):
        print("Pickled cLACIr dataset exists...\n")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        datasets_array, labels_array = dataset['features'], dataset['labels']
    else:
        data = load_clacir_dataset('datasets/clacir_raw')


def windowed_feature_extraction(window_size, write_pickle=True, datastreams=None, dataset_name="clasir"):
    if datastreams is None:
        datastreams = [True, True, True, True]
    print("Collecting cLASIr dataset...")
    dataset_path = f'datasets/clasir_processed/{dataset_name + "_".join([str(int(i)) for i in datastreams])}.pkl'
    if os.path.exists(dataset_path):
        print("Pickled cLASIr dataset exists...\n")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        datasets_array, labels_array = dataset['features'], dataset['labels']
    else:
        data, labels = load_csv_dataset('datasets/clasir_raw')
        # Initialize return lists
        windowed_data = []
        windowed_labels = []
        # Initialize subject lists
        subject_data_list = None
        subject_label_list = None
        window_data = []
        # Signal window sizes
        bvp_window_size = 64.0 * window_size
        acc_window_size = 32.0 * window_size
        eda_window_size = 4.0 * window_size
        temp_window_size = 4.0 * window_size
        label_window_size = window_size
        # Iterate through the dataset types
        print("Beginning sample processing...")
        for subject in range(0, len(data)):
            print("Processing subject number:", subject)
            bvp_generator = sp.split_set(data[subject][1], bvp_window_size, 64.0 / 4)
            eda_generator = sp.split_set(data[subject][2], eda_window_size, 4.0 / 4)
            acc_generator = sp.split_set(data[subject][0], acc_window_size, 32.0 / 4)
            temp_generator = sp.split_set(data[subject][3], temp_window_size, 4.0 / 4)
            label_generator = sp.split_set(labels[subject], label_window_size, 1)
            for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator,
                                                  label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    if datastreams[0]:
                        features = get_e4_features(bvp, 'BVP')
                        if features:
                            window_data.extend(features)
                        else:
                            continue
                    if datastreams[1]:
                        window_data.extend(get_e4_features(eda, 'EDA'))
                    if datastreams[2]:
                        window_data.extend(get_e4_features(acc, 'ACC'))
                    if datastreams[3]:
                        window_data.extend(get_e4_features(temp, 'TEMP'))
                    window_label = label[int((len(label) / 2.0) + 0.5) - 1]
                    if subject_data_list is None:
                        subject_data_list = np.array(window_data)
                    else:
                        subject_data_list = np.vstack((subject_data_list, np.array(window_data)))
                    if subject_label_list is None:
                        subject_label_list = np.array(window_label)
                    else:
                        subject_label_list = np.vstack((subject_label_list, np.array(window_label)))
                    window_data = []
            windowed_data.append(subject_data_list)
            windowed_labels.append(subject_label_list)
            subject_data_list = None
            subject_label_list = None

        datasets_array = windowed_data
        labels_array = windowed_labels

        if write_pickle:
            print("Currently pickling cLASIr dataset...\n")
            with open(dataset_path, 'wb') as f:
                pickle.dump({"features": datasets_array,
                             "labels": labels_array}, f)

    remove_nan(datasets_array)
    # datasets_array = normalize_dataset(datasets_array)
    ap_dataset, ap_labels = pose_intervention_binary(datasets_array, labels_array)
    int_dataset, int_labels = pose_control_v_condition_binary(datasets_array, labels_array)
    datasets_array, labels_array = datasets_array, labels_array
    binary_dataset, binary_labels = binarize_dataset(datasets_array, labels_array)
    return (datasets_array, labels_array), (binary_dataset, binary_labels), (ap_dataset, ap_labels), (
    int_dataset, int_labels)
