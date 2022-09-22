import glob
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
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


def windowed_feature_extraction(window_size, train_portion=0.7, test_portion=0.2, dev_portion=0.1,
                                      write_csv=True, write_pickle=True, dataset_name="case"):
    print("Collecting CASE dataset...\n")
    if os.path.exists(f'datasets/case_processed/{dataset_name}.pkl'):
        print("Pickled CASE dataset exists...")
        with open(f'datasets/case_processed/{dataset_name}.pkl', 'rb') as f:
            datasets_array, labels_array = pickle.load(f)
    else:
        subject_data, annotations = load_csv_dataset('datasets/case_raw')
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
        bvp_window_size = 1000.0 * window_size
        eda_window_size = 1000.0 * window_size
        temp_window_size = 1000.0 * window_size
        label_window_size = 250.0 * window_size
        # Segment dataset types
        train_samples = int(np.round(len(subject_data) * train_portion))
        test_samples = int(np.round(len(subject_data) * test_portion))
        dev_samples = int(np.round(len(subject_data) * dev_portion))
        print("Train Samples:", train_samples, "| Test Samples:", test_samples, "| Dev Samples:", dev_samples)
        # Iterate through the dataset types
        print("Beginning train sample processing...")
        for train in range(0, train_samples):
            print("Processing subject number:", train)
            bvp_generator = sp.split_set(subject_data[train][1], bvp_window_size, 250)
            eda_generator = sp.split_set(subject_data[train][2], eda_window_size, 250)
            temp_generator = sp.split_set(subject_data[train][3], temp_window_size, 250)
            label_generator = sp.split_set(annotations[train][4], label_window_size, 75)
            for bvp, eda, temp, label in zip(bvp_generator, eda_generator, temp_generator, label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    window_data.append(get_e4_features(bvp, 'BVP'))
                    window_data.append(get_e4_features(eda, 'EDA'))
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
            bvp_generator = sp.split_set(subject_data[test][1], bvp_window_size, 250)
            eda_generator = sp.split_set(subject_data[test][2], eda_window_size, 250)
            temp_generator = sp.split_set(subject_data[test][3], temp_window_size, 250)
            label_generator = sp.split_set(annotations[test][4], label_window_size, 75)
            for bvp, eda, temp, label in zip(bvp_generator, eda_generator, temp_generator, label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    window_data.append(get_e4_features(bvp, 'BVP'))
                    window_data.append(get_e4_features(eda, 'EDA'))
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
            bvp_generator = sp.split_set(subject_data[dev][1], bvp_window_size, 250)
            eda_generator = sp.split_set(subject_data[dev][2], eda_window_size, 250)
            temp_generator = sp.split_set(subject_data[dev][3], temp_window_size, 250)
            label_generator = sp.split_set(annotations[dev][4], label_window_size, 75)
            for bvp, eda, temp, label in zip(bvp_generator, eda_generator, temp_generator, label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    window_data.append(get_e4_features(bvp, 'BVP'))
                    window_data.append(get_e4_features(eda, 'EDA'))
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
        if write_pickle:
            print("Currently pickling cLASIr dataset...")
            with open(f'datasets/case_processed/{dataset_name}.pkl', 'wb') as f:
                pickle.dump({"features": datasets_array,
                             "labels": labels_array}, f)
    return datasets_array, labels_array


def load_csv_dataset(parent_dir):
    datasets_from_dir = []
    annotations_from_dir = []
    datasets = []
    annotations = []

    for filename in glob.iglob(parent_dir + r'\physiological\*', recursive=True):
        if filename.endswith(".csv"):
            datasets_from_dir.append(filename)
    for filename in glob.iglob(parent_dir + r'\annotations\*', recursive=True):
        if filename.endswith(".csv"):
            annotations_from_dir.append(filename)

    if datasets_from_dir and annotations_from_dir:
        clustered = True
        for dataset, annotation in zip(datasets_from_dir, annotations_from_dir):
            print(f"Processing {dataset}...")
            datasets.append([])
            annotations.append([])
            d_df = pd.read_csv(dataset)
            for col in ['daqtime', 'bvp', 'gsr', 'skt']:
                datasets[-1].append(np.array(d_df[col].to_list()))
            a_df = pd.read_csv(annotation)
            for col in ['jstime', 'valence', 'arousal', 'video', 'cluster']:
                try:
                    annotations[-1].append(np.array(a_df[col].to_list()))
                except KeyError:
                    clustered = False
    else:
        raise ValueError("No CASE dataset found!")

    if not clustered:
        k_data_x, k_data_y = [], []
        for a in annotations:
            k_data_x.extend(a[1])
            k_data_y.extend(a[2])

        print(f"Clustering with k-means...")
        k_data = np.column_stack((k_data_x, k_data_y))
        kmean = KMeans(n_clusters=4)
        kmean.fit(k_data)

        index_count = 0
        for a in annotations:
            a.append(kmean.labels_[index_count:index_count+len(a[1])])
            index_count += len(a[1])

        y_pred = kmean.predict(k_data)
        plt.scatter(k_data_x, k_data_y, c=y_pred, s=50)
        centers = kmean.cluster_centers_
        print("Calculating Silhouette score...")
        s_score = metrics.silhouette_score(k_data, kmean.labels_, metric='euclidean', sample_size=5000)
        print(f"Silhouette Score: {s_score}")
        center = 0
        for c in centers:
            plt.text(c[0], c[1], center)
            print(f"Center {center}: {c}")
            center += 1
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.title('CASE Dataset Clustering')
        plt.xlabel('Valence')
        plt.ylabel('Arousal')
        plt.savefig(parent_dir + r'\clustered_case.png')

        for annotation, new_a in zip(annotations_from_dir, annotations):
            a_df = pd.read_csv(annotation)
            a_df['cluster'] = new_a[-1]
            a_df.to_csv(annotation, index=False)
    else:
        print("CASE dataset already clustered")
    return datasets, annotations
