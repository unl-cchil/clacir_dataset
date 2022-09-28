import glob
import os
import pickle

import numpy as np
import signal_processing as sp
import csv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
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


def trim_data(dataset, labels):
    # 0 = N/A | 1 = Precondition | 2 = HAI | 3 = Control | 4 = Postcondition
    trimmed_dataset, trimmed_labels = [], []
    for x, y in zip(dataset, labels):
        del_indxs = np.hstack((np.where(y == 0.0)[0], np.where(y >= 4)[0]))
        trimmed_labels.append(np.delete(y, del_indxs, 0))
        trimmed_dataset.append(np.delete(x, del_indxs, 0))
        trimmed_labels[-1][trimmed_labels[-1] == 3] = 0
    return trimmed_dataset, trimmed_labels


def binarize_dataset(dataset, labels):
    # 1 = Precondition | 2 = HAI | 3 = Control
    binary_dataset, binary_labels = [], []
    for x, y in zip(dataset, labels):
        binary_dataset.append(x)
        binary_labels.append(y)
        binary_labels[-1][binary_labels[-1] == 2] = 0
    return binary_dataset, binary_labels


def windowed_feature_extraction(window_size, train_portion=0.7, test_portion=0.2, dev_portion=0.1,
                                write_csv=True, write_pickle=True, exclude_acc=False, dataset_name="clasir"):
    print("Collecting cLASIr dataset...\n")
    if os.path.exists(f'datasets/clasir_processed/{dataset_name}.pkl'):
        print("Pickled cLASIr dataset exists...")
        with open(f'datasets/clasir_processed/{dataset_name}.pkl', 'rb') as f:
            dataset = pickle.load(f)
        datasets_array, labels_array = dataset['features'], dataset['labels']
    else:
        data, labels = load_csv_dataset('datasets/clasir_raw')
        # Initialize return lists
        windowed_train_data = []
        windowed_train_labels = []
        windowed_test_data = []
        windowed_test_labels = []
        windowed_dev_data = []
        windowed_dev_labels = []
        # Initialize subject lists
        subject_data_list = []
        subject_label_list = []
        window_data = []
        window_label = []
        # Signal window sizes
        bvp_window_size = 64.0 * window_size
        acc_window_size = 32.0 * window_size
        eda_window_size = 4.0 * window_size
        temp_window_size = 4.0 * window_size
        label_window_size = window_size
        feature_count = 0
        # Segment dataset types
        train_samples = int(np.round(len(data) * train_portion))
        test_samples = int(np.round(len(data) * test_portion))
        dev_samples = int(np.round(len(data) * dev_portion))
        print("Train Samples:", train_samples, "| Test Samples:", test_samples, "| Dev Samples:", dev_samples)
        # Iterate through the dataset types
        print("Beginning train sample processing...")
        for train in range(0, train_samples):
            print("Processing subject number:", train)
            bvp_generator = sp.split_set(data[train][1], bvp_window_size, 64.0 / 4)
            eda_generator = sp.split_set(data[train][2], eda_window_size, 4.0 / 4)
            acc_generator = sp.split_set(data[train][0], acc_window_size, 32.0 / 4)
            temp_generator = sp.split_set(data[train][3], temp_window_size, 4.0 / 4)
            label_generator = sp.split_set(labels[train], label_window_size, 1)
            for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator, label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    window_data.extend(get_e4_features(bvp, 'BVP'))
                    window_data.extend(get_e4_features(eda, 'EDA'))
                    if not exclude_acc:
                        window_data.extend(get_e4_features(acc, 'ACC'))
                    window_data.extend(get_e4_features(temp, 'TEMP'))
                    feature_count = len(window_data)
                    window_label.append(label[int((len(label) / 2.0) + 0.5) - 1])
                    subject_data_list.append(np.array(window_data))
                    subject_label_list.append(window_label)
                    window_data = []
                    window_label = []
            windowed_train_data.append(subject_data_list)
            windowed_train_labels.append(subject_label_list)
            subject_data_list = []
            subject_label_list = []
        for test in range(train_samples, train_samples + test_samples):
            print("Processing subject number:", test)
            bvp_generator = sp.split_set(data[test][1], bvp_window_size, 64.0 / 4)
            eda_generator = sp.split_set(data[test][2], eda_window_size, 4.0 / 4)
            acc_generator = sp.split_set(data[test][0], acc_window_size, 32.0 / 4)
            temp_generator = sp.split_set(data[test][3], temp_window_size, 4.0 / 4)
            label_generator = sp.split_set(labels[test], label_window_size, window_size / 4)
            for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator, label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    window_data.extend(get_e4_features(bvp, 'BVP'))
                    window_data.extend(get_e4_features(eda, 'EDA'))
                    if not exclude_acc:
                        window_data.extend(get_e4_features(acc, 'ACC'))
                    window_data.extend(get_e4_features(temp, 'TEMP'))
                    feature_count = len(window_data)
                    window_label.append(label[int((len(label) / 2.0) + 0.5) - 1])
                    subject_data_list.append(np.array(window_data))
                    subject_label_list.append(window_label)
                    window_data = []
                    window_label = []
            windowed_test_data.append(subject_data_list)
            windowed_test_labels.append(subject_label_list)
            subject_data_list = []
            subject_label_list = []
        for dev in range(train_samples + test_samples, train_samples + test_samples + dev_samples):
            print("Processing subject number:", dev)
            bvp_generator = sp.split_set(data[dev][1], bvp_window_size, 64.0 / 4)
            eda_generator = sp.split_set(data[dev][2], eda_window_size, 4.0 / 4)
            acc_generator = sp.split_set(data[dev][0], acc_window_size, 32.0 / 4)
            temp_generator = sp.split_set(data[dev][3], temp_window_size, 4.0 / 4)
            label_generator = sp.split_set(labels[dev], label_window_size, 1)
            for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator, label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    window_data.extend(get_e4_features(bvp, 'BVP'))
                    window_data.extend(get_e4_features(eda, 'EDA'))
                    if not exclude_acc:
                        window_data.extend(get_e4_features(acc, 'ACC'))
                    window_data.extend(get_e4_features(temp, 'TEMP'))
                    feature_count = len(window_data)
                    window_label.append(label[int((len(label) / 2.0) + 0.5) - 1])
                    subject_data_list.append(np.array(window_data))
                    subject_label_list.append(window_label)
                    window_data = []
                    window_label = []
            windowed_dev_data.append(subject_data_list)
            windowed_dev_labels.append(subject_label_list)
            subject_data_list = []
            subject_label_list = []

        print("Converting lists to arrays...")
        datasets = [windowed_train_data, windowed_test_data, windowed_dev_data]
        labels = [windowed_train_labels, windowed_test_labels, windowed_dev_labels]

        datasets_array = []
        labels_array = []
        for dataset, label in zip(datasets, labels):
            x_train = np.empty((feature_count,))
            y_train = np.empty((1,))
            for window, lbl in zip(dataset, label):
                for x, y in zip(window, lbl):
                    data_window = x
                    data_window[np.isnan(data_window)] = 0
                    data_window[np.isinf(data_window)] = 0
                    data_label = np.array(y[0])
                    data_label[np.isnan(data_label)] = 0
                    data_label[np.isinf(data_label)] = 0
                    x_train = np.vstack((x_train, data_window))
                    y_train = np.vstack((y_train, data_label))
            datasets_array.append(x_train)
            labels_array.append(y_train)

        print("Imputing missing features...")
        labels_array[0][0] = 0.0
        datasets_array[0][0] = 0.0
        nan_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        inf_imp = SimpleImputer(missing_values=np.inf, strategy='mean')
        for i in range(0, len(datasets_array)):
            nan_imp.fit(datasets_array[i])
            nan_imp.transform(datasets_array[i])
            inf_imp.fit(datasets_array[i])
            inf_imp.transform(datasets_array[i])
        if write_pickle:
            print("Currently pickling cLASIr dataset...")
            with open(f'datasets/clasir_processed/{dataset_name}.pkl', 'wb') as f:
                pickle.dump({"features": datasets_array,
                             "labels": labels_array}, f)
    datasets_array, labels_array = trim_data(datasets_array, labels_array)
    datasets_array = normalize_dataset(datasets_array)
    binary_dataset, binary_labels = binarize_dataset(datasets_array, labels_array)
    return (datasets_array, labels_array), (binary_dataset, binary_labels)
