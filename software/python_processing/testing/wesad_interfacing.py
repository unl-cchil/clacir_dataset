import copy
import csv
import glob
import os
import pickle

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
        y -= 1
        binary_dataset.append(copy.deepcopy(x))
        binary_labels.append(copy.deepcopy(y))
        binary_labels[-1][binary_labels[-1] == 3] = 0
        binary_labels[-1][binary_labels[-1] == 1] = 0
        binary_labels[-1][binary_labels[-1] == 2] = 1
    return binary_dataset, binary_labels


def remove_nan(dataset):
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    for i in range(0, len(dataset)):
        inf_indx = np.isinf(dataset[i])
        dataset[i][inf_indx] = np.nan
        dataset[i] = imp.fit_transform(dataset[i])


def get_raw_respiban_dataset():
    if os.path.exists(f"datasets/wesad_processed/raw_dataset.pkl"):
        with open(f"datasets/wesad_processed/raw_dataset.pkl", 'rb') as f:
            dataset = pickle.load(f)
        data, labels = dataset['data'], dataset['labels']
    else:
        subject_data = load_wesad_dataset(r"E:\Datasets\WESAD/")
        data, labels = [], []
        for subject in range(0, len(subject_data)):
            data.append([subject_data[subject]['signal']['chest']['ACC'],
                         subject_data[subject]['signal']['chest']['ECG'],
                         subject_data[subject]['signal']['chest']['EDA'],
                         subject_data[subject]['signal']['chest']['Temp']])
            labels.append(subject_data[subject]['label'])
        with open(f'datasets/wesad_processed/raw_dataset.pkl', 'wb') as f:
            pickle.dump({"data": data,
                         "labels": labels}, f)
    return data, labels


def get_raw_e4_dataset():
    if os.path.exists(f"datasets/wesad_processed/raw_dataset.pkl"):
        with open(f"datasets/wesad_processed/raw_dataset.pkl", 'rb') as f:
            dataset = pickle.load(f)
        data, labels = dataset['data'], dataset['labels']
    else:
        subject_data = load_wesad_dataset(r"E:\Datasets\WESAD/")
        data, labels = [], []
        for subject in range(0, len(subject_data)):
            data.append([subject_data[subject]['signal']['wrist']['ACC'],
                         subject_data[subject]['signal']['wrist']['BVP'],
                         subject_data[subject]['signal']['wrist']['EDA'],
                         subject_data[subject]['signal']['wrist']['TEMP']])
            labels.append(subject_data[subject]['label'])
        with open(f'datasets/wesad_processed/raw_dataset.pkl', 'wb') as f:
            pickle.dump({"data": data,
                         "labels": labels}, f)
    return data, labels


def respiban_windowed_feature_extraction(window_size, write_pickle=True, datastreams=None,
                                         dataset_name="wesad_respiban"):
    if datastreams is None:
        datastreams = [True, True, True, True]
    print("Collecting WESAD dataset...")
    dataset_path = f'datasets/wesad_processed/{dataset_name + "_".join([str(int(i)) for i in datastreams])}.pkl'
    if os.path.exists(dataset_path):
        print("Pickled WESAD dataset exists...\n")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        datasets_array, labels_array = dataset['features'], dataset['labels']
    else:
        subject_data = load_wesad_dataset(r"E:\Thesis Results\Datasets\WESAD/")
        # Initialize return lists
        windowed_data = []
        windowed_labels = []
        # Initialize subject lists
        subject_data_list = None
        subject_label_list = None
        window_data = []
        # Signal window sizes
        bvp_window_size = 700.0 * window_size
        acc_window_size = 700.0 * window_size
        eda_window_size = 700.0 * window_size
        temp_window_size = 700.0 * window_size
        label_window_size = 700.0 * window_size
        # Iterate through the dataset types
        print("Beginning sample processing...")
        for subject in range(0, len(subject_data)):
            print("Processing subject number:", subject)
            bvp_generator = sp.split_set(subject_data[subject]['signal']['chest']['ECG'], bvp_window_size, 700.0 / 4)
            eda_generator = sp.split_set(subject_data[subject]['signal']['chest']['EDA'], eda_window_size, 700.0 / 4)
            acc_generator = sp.split_set(subject_data[subject]['signal']['chest']['ACC'], acc_window_size, 700.0 / 4)
            temp_generator = sp.split_set(subject_data[subject]['signal']['chest']['Temp'], temp_window_size, 700.0 / 4)
            label_generator = sp.split_set(subject_data[subject]['label'], label_window_size, 700.0 / 4)
            for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator,
                                                  label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    if datastreams[0]:
                        features = get_e4_features(bvp, 'BVP', sample_rate=700.0)
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
                    window_label = np.around(np.average(label))
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
            print("Currently pickling WESAD dataset...\n")
            with open(dataset_path, 'wb') as f:
                pickle.dump({"features": datasets_array,
                             "labels": labels_array}, f)

    remove_nan(datasets_array)
    datasets_array, labels_array = trim_data(datasets_array, labels_array)
    binary_dataset, binary_labels = binarize_dataset(datasets_array, labels_array)
    return (datasets_array, labels_array), (binary_dataset, binary_labels)


def e4_windowed_feature_extraction(window_size, write_pickle=True, datastreams=None, dataset_name="wesad"):
    if datastreams is None:
        datastreams = [True, True, True, True]
    print("Collecting WESAD dataset...")
    dataset_path = f'datasets/wesad_processed/{dataset_name + "_".join([str(int(i)) for i in datastreams])}.pkl'
    if os.path.exists(dataset_path):
        print("Pickled WESAD dataset exists...\n")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        datasets_array, labels_array = dataset['features'], dataset['labels']
    else:
        subject_data = load_wesad_dataset(r"E:\Thesis Results\Datasets\WESAD/")
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
        label_window_size = 700.0 * window_size
        # Iterate through the dataset types
        print("Beginning sample processing...")
        for subject in range(0, len(subject_data)):
            print("Processing subject number:", subject)
            bvp_generator = sp.split_set(subject_data[subject]['signal']['wrist']['BVP'], bvp_window_size, 64.0)
            eda_generator = sp.split_set(subject_data[subject]['signal']['wrist']['EDA'], eda_window_size, 4.0)
            acc_generator = sp.split_set(subject_data[subject]['signal']['wrist']['ACC'], acc_window_size, 32.0)
            temp_generator = sp.split_set(subject_data[subject]['signal']['wrist']['TEMP'], temp_window_size, 4.0)
            label_generator = sp.split_set(subject_data[subject]['label'], label_window_size, 700.0)
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
            windowed_data.append(subject_data_list)
            windowed_labels.append(subject_label_list)
            subject_data_list = None
            subject_label_list = None

        datasets_array = windowed_data
        labels_array = windowed_labels

        if write_pickle:
            print("Currently pickling WESAD dataset...\n")
            with open(dataset_path, 'wb') as f:
                pickle.dump({"features": datasets_array,
                             "labels": labels_array}, f)

    remove_nan(datasets_array)
    datasets_array, labels_array = trim_data(datasets_array, labels_array)
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
