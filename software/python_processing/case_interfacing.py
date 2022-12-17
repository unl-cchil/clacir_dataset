import copy
import glob
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
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


def trim_dataset(dataset, labels):
    trimmed_dataset, trimmed_labels = [], []
    for x, y in zip(dataset, labels):
        del_labels = np.where(y == 3)
        trimmed_dataset.append(np.delete(x, del_labels, 0))
        trimmed_labels.append(np.delete(y, del_labels, 0))
    return trimmed_dataset, trimmed_labels


def binarize_dataset(dataset, labels):
    # Pose stress vs. non-stress problem, merge boredom and amusement into neutral
    # 0 = neutral | 1 = stress/fear | 2 = amusement
    binary_dataset, binary_labels = [], []
    for x, y in zip(dataset, labels):
        binary_dataset.append(copy.deepcopy(x))
        binary_labels.append(copy.deepcopy(y))
        binary_labels[-1][binary_labels[-1] > 1] = 0
    return binary_dataset, binary_labels


def remove_nan(dataset):
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    for i in range(0, len(dataset)):
        inf_indx = np.isinf(dataset[i])
        dataset[i][inf_indx] = np.nan
        dataset[i] = imp.fit_transform(dataset[i])


def windowed_feature_extraction(window_size, write_pickle=True, datastreams=None, dataset_name="case"):
    if datastreams is None:
        datastreams = [True, True, True, True]
    print("Collecting CASE dataset...")
    dataset_path = f'datasets/case_processed/{dataset_name + "_".join([str(int(i)) for i in datastreams])}.pkl'
    if os.path.exists(dataset_path):
        print("Pickled CASE dataset exists...\n")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        datasets_array, labels_array = dataset['features'], dataset['labels']
    else:
        subject_data, annotations = load_csv_dataset('datasets/case_raw')
        # Initialize return lists
        windowed_data = []
        windowed_labels = []
        # Initialize subject lists
        subject_data_list = None
        subject_label_list = None
        window_data = []
        # Signal window sizes
        bvp_window_size = 1000.0 * window_size
        eda_window_size = 1000.0 * window_size
        temp_window_size = 1000.0 * window_size
        label_window_size = 20.0 * window_size
        # Iterate through the dataset types
        print("Beginning sample processing...")
        for subject in range(0, len(subject_data)):
            print("Processing subject number:", subject)
            bvp_generator = sp.split_set(subject_data[subject][1], bvp_window_size, 1000)
            eda_generator = sp.split_set(subject_data[subject][2], eda_window_size, 1000)
            temp_generator = sp.split_set(subject_data[subject][3], temp_window_size, 1000)
            label_generator = sp.split_set(annotations[subject][4], label_window_size, 20)
            for bvp, eda, temp, label in zip(bvp_generator, eda_generator, temp_generator, label_generator):
                if len(bvp) < bvp_window_size or len(eda) < eda_window_size or len(temp) < temp_window_size:
                    continue
                else:
                    if datastreams[0]:
                        window_data.extend(get_e4_features(bvp, 'BVP'))
                    if datastreams[1]:
                        window_data.extend(get_e4_features(eda, 'EDA'))
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
            print("Currently pickling CASE dataset...\n")
            with open(dataset_path, 'wb') as f:
                pickle.dump({"features": datasets_array,
                             "labels": labels_array}, f)

    remove_nan(datasets_array)
    datasets_array, labels_array = trim_dataset(datasets_array, labels_array)
    binary_dataset, binary_labels = binarize_dataset(datasets_array, labels_array)
    return (datasets_array, labels_array), (binary_dataset, binary_labels)


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
