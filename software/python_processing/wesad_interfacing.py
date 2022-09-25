import csv
import glob
import pickle
import numpy as np
import scipy.signal as signal
from sklearn.impute import SimpleImputer
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


def windowed_feature_extraction(window_size, train_portion=0.7, test_portion=0.2, dev_portion=0.1,
                                write_csv=True, write_pickle=True, exclude_acc=False, dataset_name="wesad"):
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
            datasets_array, labels_array = pickle.load(f)
    else:
        subject_data = load_wesad_dataset(r"E:\Datasets\WESAD/")
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
            label_generator = sp.split_set(subject_data[train]['label'], label_window_size, 1)
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
                    window_label.append(get_e4_labels(label))
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
            bvp_generator = sp.split_set(subject_data[test]['signal']['wrist']['BVP'], bvp_window_size, 64.0 / 4)
            eda_generator = sp.split_set(subject_data[test]['signal']['wrist']['EDA'], eda_window_size, 4.0 / 4)
            acc_generator = sp.split_set(subject_data[test]['signal']['wrist']['ACC'], acc_window_size, 32.0 / 4)
            temp_generator = sp.split_set(subject_data[test]['signal']['wrist']['TEMP'], temp_window_size, 4.0 / 4)
            label_generator = sp.split_set(subject_data[test]['label'], label_window_size, 1)
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
                    window_label.append(get_e4_labels(label))
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
            bvp_generator = sp.split_set(subject_data[dev]['signal']['wrist']['BVP'], bvp_window_size, 64.0 / 4)
            eda_generator = sp.split_set(subject_data[dev]['signal']['wrist']['EDA'], eda_window_size, 4.0 / 4)
            acc_generator = sp.split_set(subject_data[dev]['signal']['wrist']['ACC'], acc_window_size, 32.0 / 4)
            temp_generator = sp.split_set(subject_data[dev]['signal']['wrist']['TEMP'], temp_window_size, 4.0 / 4)
            label_generator = sp.split_set(subject_data[dev]['label'], label_window_size, 1)
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
                    window_label.append(get_e4_labels(label))
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
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(datasets_array[0])
        imp.transform(datasets_array[0])
        imp.fit(labels_array[0])
        imp.transform(labels_array[0])
        imp = SimpleImputer(missing_values=np.inf, strategy='mean')
        imp.fit(datasets_array[0])
        imp.transform(datasets_array[0])
        imp.fit(labels_array[0])
        imp.transform(labels_array[0])

        if write_pickle:
            print("Currently pickling WESAD dataset...")
            with open(f'datasets/wesad_processed/{dataset_name}.pkl', 'wb') as f:
                pickle.dump({"features": datasets_array,
                             "labels": labels_array}, f)
    return datasets_array, labels_array


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
