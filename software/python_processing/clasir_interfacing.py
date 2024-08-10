import copy
import csv
import glob
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

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
    scaler = MinMaxScaler(feature_range=(0, 1))
    for x in dataset:
        normalized_dataset.append(scaler.fit_transform(x))
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
    # 0 = Control | 1 = Precondition | 2 = HAI
    binary_dataset, binary_labels = [], []
    for x, y in zip(dataset, labels):
        binary_dataset.append(copy.deepcopy(x))
        binary_labels.append(copy.deepcopy(y))
        binary_labels[-1][binary_labels[-1] == 2] = 0
    return binary_dataset, binary_labels


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


def load_clacir_dataset(filepath):
    pd.options.mode.chained_assignment = None
    affect_data = pd.read_csv(os.path.join(filepath, "affect_data.csv"))
    survey_data = pd.read_csv(os.path.join(filepath, "thayer_stevens_2021_data1.csv"))
    protocol_times = pd.read_csv(os.path.join(filepath, "hr_data_times.csv"))
    expt1_subjects = [f for f in os.listdir(os.path.join(filepath, "E4_expt1")) if
                      not os.path.exists(os.path.join(filepath, "E4_expt1", f, "NA.txt"))]
    expt2_subjects = [f for f in os.listdir(os.path.join(filepath, "E4_expt2")) if
                      not os.path.exists(os.path.join(filepath, "E4_expt2", f, "NA.txt"))]
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


def generate_dataset(window_size, write_pickle=True, datastreams=None, dataset_name="clacir"):
    if datastreams is None:
        datastreams = [True, True, True, True]
    print("Collecting cLACIr dataset...")
    dataset_path = f'datasets/clacir_processed/{dataset_name + "_".join([str(int(i)) for i in datastreams])}.pkl'
    if os.path.exists(dataset_path):
        print("Pickled cLACIr dataset exists...\n")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        datasets_array, labels_array = dataset['features'], dataset['labels']
    else:
        data, labels = load_clacir_dataset('datasets/clacir_raw')


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
            for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator, label_generator):
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
    ap_dataset, ap_labels = pose_intervention_binary(datasets_array, labels_array)
    int_dataset, int_labels = pose_control_v_condition_binary(datasets_array, labels_array)
    datasets_array, labels_array = trim_data(datasets_array, labels_array)
    binary_dataset, binary_labels = binarize_dataset(datasets_array, labels_array)
    return (datasets_array, labels_array), (binary_dataset, binary_labels), (ap_dataset, ap_labels), (int_dataset, int_labels)
