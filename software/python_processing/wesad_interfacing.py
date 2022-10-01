import csv
import glob
import pickle
import numpy as np
import scipy.signal as signal
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import signal_processing as sp
import os
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


def normalize_dataset(dataset):
    normalized_dataset = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    for x in dataset:
        normalized_dataset.append(scaler.fit_transform(x))
    return normalized_dataset


def trim_data(subject_data, subject_labels):
    """Trims the dataset of the unused labels and windows, prepares it to be converted to a binary and multi-class set
    Parameters
    :param subject_labels: ndarray
        Array of labels for the subject
    :param subject_data: ndarray
        Array for daya for the subject aligned with the labels
    :return: ndarray, ndarray
        Trimmed data and labels
    """
    del_data, del_labels = [], []
    for x, y in zip(subject_data, subject_labels):
        del_indxs = np.hstack((np.where(y >= 4.0)[0], np.where(y == 0.0)[0]))
        del_labels.append(np.delete(y, del_indxs, 0))
        del_data.append(np.delete(x, del_indxs, 0))
    return del_data, del_labels


def binarize_dataset(dataset, labels):
    # Merge amusement into neutral class
    binary_dataset, binary_labels = [], []
    for x, y in zip(dataset, labels):
        binary_dataset.append(x)
        binary_labels.append(y)
        binary_labels[-1][binary_labels[-1] == 3] = 0
        binary_labels[-1][binary_labels[-1] == 1] = 0
        binary_labels[-1][binary_labels[-1] == 2] = 1
    return binary_dataset, binary_labels


def remove_nan(dataset):
    for x in dataset:
        np.nan_to_num(x, nan=0, posinf=0, neginf=0, copy=False)


def windowed_feature_extraction(window_size, train_portion=0.7, test_portion=0.2, dev_portion=0.1,
                                write_pickle=True, exclude_acc=False, dataset_name="wesad"):
    """Function to run through the entire WESAD dataset and extract the features.
    Parameters
    :param window_size: float
        Window size in seconds, such as one second = 1.0
    :param train_portion: float
        Portion of dataset to be used for training, default = 0.7
    :param test_portion: float
        Portion of dataset to be used for testing, default = 0.2
    :param dev_portion: float
        Portion of dataset to be used for development, default = 0.1
    :param write_csv: bool
        Indicate if features should be written out to CSV file
    :param write_pickle: bool
        Indicate if features should be written out to pickle file
    :return: list
        List of features
    """
    print("Collecting WESAD dataset...\n")
    if os.path.exists(f'datasets/wesad_processed/{dataset_name}.pkl'):
        print("Pickled WESAD dataset exists...")
        with open(f'datasets/wesad_processed/{dataset_name}.pkl', 'rb') as f:
            dataset = pickle.load(f)
        datasets_array, labels_array = dataset['features'], dataset['labels']
    else:
        subject_data = load_wesad_dataset(r"E:\Datasets\WESAD/")
        # Initialize return lists
        windowed_train_data = None
        windowed_train_labels = None
        windowed_test_data = None
        windowed_test_labels = None
        windowed_dev_data = None
        windowed_dev_labels = None
        # Initialize subject lists
        subject_data_list = None
        subject_label_list = None
        window_data = []
        # Signal window sizes
        bvp_window_size = 64.0 * window_size
        acc_window_size = 32.0 * window_size
        eda_window_size = 4.0 * window_size
        temp_window_size = 4.0 * window_size
        label_window_size = 700.0 * window_size
        feature_count = 0
        # Segment dataset types
        train_samples = int(np.round(len(subject_data) * train_portion))
        test_samples = int(np.round(len(subject_data) * test_portion))
        dev_samples = int(np.round(len(subject_data) * dev_portion))
        print("Train Samples:", train_samples, "| Test Samples:", test_samples, "| Dev Samples:", dev_samples)
        # Iterate through the dataset types
        print("Beginning train sample processing...")
        for train in range(0, train_samples):
            print("Processing subject number:", train)
            bvp_generator = sp.split_set(subject_data[train]['signal']['wrist']['BVP'], bvp_window_size, 64.0 / 4)
            eda_generator = sp.split_set(subject_data[train]['signal']['wrist']['EDA'], eda_window_size, 4.0 / 4)
            acc_generator = sp.split_set(subject_data[train]['signal']['wrist']['ACC'], acc_window_size, 32.0 / 4)
            temp_generator = sp.split_set(subject_data[train]['signal']['wrist']['TEMP'], temp_window_size, 4.0 / 4)
            label_generator = sp.split_set(subject_data[train]['label'], label_window_size, 700.0 / 4)
            for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator, label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    window_data.extend(get_e4_features(bvp, 'BVP'))
                    window_data.extend(get_e4_features(eda, 'EDA'))
                    if not exclude_acc:
                        window_data.extend(get_e4_features(acc, 'ACC'))
                    window_data.extend(get_e4_features(temp, 'TEMP'))
                    window_label = get_e4_labels(label)
                    if subject_data_list is None:
                        subject_data_list = np.array(window_data)
                    else:
                        subject_data_list = np.vstack((subject_data_list, np.array(window_data)))
                    if subject_label_list is None:
                        subject_label_list = np.array(window_label)
                    else:
                        subject_label_list = np.vstack((subject_label_list, np.array(window_label)))
                    window_data = []
            if windowed_train_data is None:
                windowed_train_data = subject_data_list
            else:
                windowed_train_data = np.vstack((windowed_train_data, subject_data_list))
            if windowed_train_labels is None:
                windowed_train_labels = subject_label_list
            else:
                windowed_train_labels = np.vstack((windowed_train_labels, subject_label_list))
            subject_data_list = None
            subject_label_list = None
        for test in range(train_samples, train_samples + test_samples):
            print("Processing subject number:", test)
            bvp_generator = sp.split_set(subject_data[test]['signal']['wrist']['BVP'], bvp_window_size, 64.0 / 4)
            eda_generator = sp.split_set(subject_data[test]['signal']['wrist']['EDA'], eda_window_size, 4.0 / 4)
            acc_generator = sp.split_set(subject_data[test]['signal']['wrist']['ACC'], acc_window_size, 32.0 / 4)
            temp_generator = sp.split_set(subject_data[test]['signal']['wrist']['TEMP'], temp_window_size, 4.0 / 4)
            label_generator = sp.split_set(subject_data[test]['label'], label_window_size, 700.0 / 4)
            for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator, label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    window_data.extend(get_e4_features(bvp, 'BVP'))
                    window_data.extend(get_e4_features(eda, 'EDA'))
                    if not exclude_acc:
                        window_data.extend(get_e4_features(acc, 'ACC'))
                    window_data.extend(get_e4_features(temp, 'TEMP'))
                    window_label = get_e4_labels(label)
                    if subject_data_list is None:
                        subject_data_list = np.array(window_data)
                    else:
                        subject_data_list = np.vstack((subject_data_list, np.array(window_data)))
                    if subject_label_list is None:
                        subject_label_list = np.array(window_label)
                    else:
                        subject_label_list = np.vstack((subject_label_list, np.array(window_label)))
                    window_data = []
            if windowed_test_data is None:
                windowed_test_data = subject_data_list
            else:
                windowed_test_data = np.vstack((windowed_test_data, subject_data_list))
            if windowed_test_labels is None:
                windowed_test_labels = subject_label_list
            else:
                windowed_test_labels = np.vstack((windowed_test_labels, subject_label_list))
            subject_data_list = None
            subject_label_list = None
        for dev in range(train_samples + test_samples, train_samples + test_samples + dev_samples):
            print("Processing subject number:", dev)
            bvp_generator = sp.split_set(subject_data[dev]['signal']['wrist']['BVP'], bvp_window_size, 64.0 / 4)
            eda_generator = sp.split_set(subject_data[dev]['signal']['wrist']['EDA'], eda_window_size, 4.0 / 4)
            acc_generator = sp.split_set(subject_data[dev]['signal']['wrist']['ACC'], acc_window_size, 32.0 / 4)
            temp_generator = sp.split_set(subject_data[dev]['signal']['wrist']['TEMP'], temp_window_size, 4.0 / 4)
            label_generator = sp.split_set(subject_data[dev]['label'], label_window_size, 700.0 / 4)
            for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator, label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    window_data.extend(get_e4_features(bvp, 'BVP'))
                    window_data.extend(get_e4_features(eda, 'EDA'))
                    if not exclude_acc:
                        window_data.extend(get_e4_features(acc, 'ACC'))
                    window_data.extend(get_e4_features(temp, 'TEMP'))
                    window_label = get_e4_labels(label)
                    if subject_data_list is None:
                        subject_data_list = np.array(window_data)
                    else:
                        subject_data_list = np.vstack((subject_data_list, np.array(window_data)))
                    if subject_label_list is None:
                        subject_label_list = np.array(window_label)
                    else:
                        subject_label_list = np.vstack((subject_label_list, np.array(window_label)))
                    window_data = []
            if windowed_dev_data is None:
                windowed_dev_data = subject_data_list
            else:
                windowed_dev_data = np.vstack((windowed_dev_data, subject_data_list))
            if windowed_dev_labels is None:
                windowed_dev_labels = subject_label_list
            else:
                windowed_dev_labels = np.vstack((windowed_dev_labels, subject_label_list))
            subject_data_list = None
            subject_label_list = None

        datasets_array = [windowed_train_data, windowed_test_data, windowed_dev_data]
        labels_array = [windowed_train_labels, windowed_test_labels, windowed_dev_labels]

        labels_array[0][0] = 0.0
        datasets_array[0][0] = 0.0

        if write_pickle:
            print("Currently pickling WESAD dataset...")
            with open(f'datasets/wesad_processed/{dataset_name}.pkl', 'wb') as f:
                pickle.dump({"features": datasets_array,
                             "labels": labels_array}, f)

    remove_nan(datasets_array)
    datasets_array, labels_array = trim_data(datasets_array, labels_array)
    datasets_array = normalize_dataset(datasets_array)
    binary_dataset, binary_labels = binarize_dataset(datasets_array, labels_array)
    return (datasets_array, labels_array), (binary_dataset, binary_labels)


def save_dataset(subject_data):
    data_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
    for sub in range(len(subject_data)):
        with open(subject_data[sub]['subject'] + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for data in data_keys:
                print("Writing line: ", data)
                writer.writerows([np.asarray(subject_data[sub]['signal']['wrist'][data].flatten())])
            writer.writerows([np.asarray(subject_data[sub]['label'].flatten())])


def load_csv_dataset(parent_dir):
    datasets_from_dir = []
    unpickled_datasets = []

    for filename in glob.iglob(parent_dir + '**/*', recursive=True):
        if filename.endswith(".csv"):
            datasets_from_dir.append(filename)

    for filename in datasets_from_dir:
        unpickled_datasets.append([])
        with open(filename, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                unpickled_datasets[-1].append(list(map(float, row)))
    return unpickled_datasets


def load_features():
    with open('extracted_features.pkl', 'rb') as f:
        dataset, labels = pickle.load(f)
    return dataset, labels


def load_wesad_dataset(parent_dir):
    """Recursive function to load pickled WESAD dataset into a dictionary
    Parameters
    :param parent_dir:
    :return:
    """
    datasets_from_dir = []
    unpickled_datasets = []

    for filename in glob.iglob(parent_dir + '**/*', recursive=True):
        if filename.endswith(".pkl"):
            datasets_from_dir.append(filename)
    if datasets_from_dir:
        for filename in datasets_from_dir:
            print("Processing file: " + filename + "...")
            unpickled_datasets.append(pickle.load(open(filename, mode='rb'), encoding='latin1'))
    else:
        raise ValueError("No WESAD datasets found!")

    return unpickled_datasets


def resample_data(ACC, EDA, labels, new_length):
    """Resamples the passed signals to the specified length using signal.resample
    TODO: Should be generalized to use a list of signals
    Parameters
    :param ACC:
    :param EDA:
    :param labels:
    :param new_length:
    :return:
    """
    new_ACC = signal.resample(ACC, new_length)
    new_EDA = signal.resample(EDA, new_length)
    new_label = np.around(signal.resample(labels, new_length)).clip(min=0, max=7)
    return new_ACC, new_EDA, new_label
