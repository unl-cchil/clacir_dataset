import glob
import numpy as np
import signal_processing as sp
import csv
from sklearn.impute import SimpleImputer
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
    return data, labels


def windowed_feature_extraction(window_size, train_portion=0.7, test_portion=0.2, dev_portion=0.1,
                                write_csv=True, write_pickle=True, exclude_acc=False):
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
    # Segment dataset types
    train_samples = int(np.round(len(data) * train_portion))
    test_samples = int(np.round(len(data) * test_portion))
    dev_samples = int(np.round(len(data) * dev_portion))
    print("Train Samples:", train_samples, "| Test Samples:", test_samples, "| Dev Samples:", dev_samples)
    # Iterate through the dataset types
    print("Beginning train sample processing...")
    for train in range(0, train_samples):
        print("Processing subject number:", train)
        bvp_generator = sp.split_set(data[train][1], bvp_window_size)
        eda_generator = sp.split_set(data[train][2], eda_window_size)
        acc_generator = sp.split_set(data[train][0], acc_window_size)
        temp_generator = sp.split_set(data[train][3], temp_window_size)
        label_generator = sp.split_set(labels[train], label_window_size)
        for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator,
                                              label_generator):
            window_data.append(get_e4_features(bvp, 'BVP'))
            window_data.append(get_e4_features(eda, 'EDA'))
            if not exclude_acc:
                window_data.append(get_e4_features(acc, 'ACC'))
                window_data.append(get_e4_features(temp, 'TEMP'))
            window_label.append(get_e4_labels(label))
            subject_data_list.append(window_data)
            subject_label_list.append(window_label)
            window_data = []
            window_label = []
        windowed_train_data.append(subject_data_list)
        windowed_train_labels.append(subject_label_list)
        subject_data_list = []
        subject_label_list = []
    for test in range(train_samples, train_samples + test_samples):
        print("Processing subject number:", test)
        bvp_generator = sp.split_set(data[test][1], bvp_window_size)
        eda_generator = sp.split_set(data[test][2], eda_window_size)
        acc_generator = sp.split_set(data[test][0], acc_window_size)
        temp_generator = sp.split_set(data[test][3], temp_window_size)
        label_generator = sp.split_set(labels[test], label_window_size)
        for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator,
                                              label_generator):
            window_data.append(get_e4_features(bvp, 'BVP'))
            window_data.append(get_e4_features(eda, 'EDA'))
            if not exclude_acc:
                window_data.append(get_e4_features(acc, 'ACC'))
                window_data.append(get_e4_features(temp, 'TEMP'))
            window_label.append(get_e4_labels(label))
            subject_data_list.append(window_data)
            subject_label_list.append(window_label)
            window_data = []
            window_label = []
        windowed_test_data.append(subject_data_list)
        windowed_test_labels.append(subject_label_list)
        subject_data_list = []
        subject_label_list = []
    for dev in range(train_samples + test_samples, train_samples + test_samples + dev_samples):
        print("Processing subject number:", dev)
        bvp_generator = sp.split_set(data[dev][1], bvp_window_size)
        eda_generator = sp.split_set(data[dev][2], eda_window_size)
        acc_generator = sp.split_set(data[dev][0], acc_window_size)
        temp_generator = sp.split_set(data[dev][3], temp_window_size)
        label_generator = sp.split_set(labels[dev], label_window_size)
        for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator,
                                              label_generator):
            window_data.append(get_e4_features(bvp, 'BVP'))
            window_data.append(get_e4_features(eda, 'EDA'))
            if not exclude_acc:
                window_data.append(get_e4_features(acc, 'ACC'))
                window_data.append(get_e4_features(temp, 'TEMP'))
            window_label.append(get_e4_labels(label))
            subject_data_list.append(window_data)
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
        x_train = np.empty((58,))
        y_train = np.empty((1,))
        for window, lbl in zip(dataset, label):
            for x, y in zip(window, lbl):
                data_window = np.hstack((np.array(x[0]), np.array(x[1]), np.array(x[2]), np.array(x[3])))
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
    # TODO: Fix formatting for CSV output. Currently outputs an unintelligible mess.
    if write_csv:
        print("Writing to CSV...")
        with open('extracted_features.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for data in range(len(datasets_array)):
                for i in range(len(datasets_array[data])):
                    print("Writing line: ", i)
                    writer.writerows([np.append(datasets_array[data][i], [labels_array[data][i]])])
    # if write_pickle:
    #     print("Currently pickling...")
    #     inf.save_features([datasets_array, labels_array])
    return datasets_array, labels_array


windowed_feature_extraction(5.0)
