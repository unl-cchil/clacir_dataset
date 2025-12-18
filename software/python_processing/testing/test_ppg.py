import numpy as np
import copy
import csv
import glob
import os
import pickle

import numpy as np
import scipy.signal as signal
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import signal_processing as sp
from inferencing import get_e4_features
import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from wesad_interfacing import load_wesad_dataset
import clacir_interfacing as clasir

clasir.generate_dataset()

window_size = 10
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
all_features = []
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
            features = get_e4_features(bvp, 'BVP')
            if features:
                all_features.append(features)

print()