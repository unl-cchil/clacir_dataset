from flirt.eda.feature_calculation import __cvx_eda
from flirt.hrv.feature_calculation import StatFeatures
from flirt.hrv.features.fd_features import FdFeatures
from flirt.hrv.features.nl_features import NonLinearFeatures
from flirt.hrv.features.td_features import TdFeatures

import signal_processing as sp
import feature_extraction as fe
import machine_learning_utils as mlu
import numpy as np
import flirt
from flirt.stats.common import get_stats
import flirt.hrv.features


def run_testing(dataset, labels, window_size=5.0):
    """Highest function call to run the pipeline from start to finish.
    Parameters
    :param labels:
    :param dataset:
    :param window_size:
    """
    # Machine learning labels
    ml_algo = ['Linear Discriminant Analysis', 'K-Neighbors Classification',
               'Decision Tree Classification', 'Random Forest Classification',
               'AdaBoost Classification']
    # Perform pre-processing input pipeline
    # if load_dataset:
    #     dataset, labels = wesad.load_features()
    # else:
    #     dataset, labels = wesad.windowed_feature_extraction(window_size=window_size)
    # Setup variables
    multi_class_x, multi_class_y = [], []
    # Iterate through data, remove undefined labels, and add to multi class
    for data, label in zip(dataset, labels):
        del_data, del_labels = mlu.trim_data(data, label)
        multi_class_x.append(del_data)
        multi_class_y.append(mlu.scale_labels(del_labels))
    # Create binary class dataset
    bin_class_x = multi_class_x.copy()
    bin_class_y = multi_class_y.copy()
    # Combine amusement class with baseline class
    for i in range(0, len(bin_class_y)):
        bin_class_y[i][np.where(bin_class_y[i] == 2)] = 0
    # Perform inferencing on both multi-class and binary class problems
    multi_model_metrics = mlu.get_metrics([np.vstack((multi_class_x[0], multi_class_x[1]))],
                                          [np.hstack((multi_class_y[0], multi_class_y[1]))],
                                          multi_class_x[2], multi_class_y[2])
    bin_model_metrics = mlu.get_metrics([np.vstack((bin_class_x[0], bin_class_x[1]))],
                                        [np.hstack((bin_class_y[0], bin_class_y[1]))],
                                        bin_class_x[2], bin_class_y[2])
    # Print classification results
    print("Multi-Class WESAD Dataset Results")
    for m_metrics, algo in zip(multi_model_metrics, ml_algo):
        print(algo,
              "| Accuracy: {:.2%}".format(m_metrics[0]),
              "| F1-Score: {:.2%}".format(m_metrics[3]),
              "| Precision: {:.2%}".format(m_metrics[1]),
              "| Recall: {:.2%}".format(m_metrics[2]))
        print("Confusion Matrix")
        print(np.matrix(m_metrics[4]), '\n')
    print("Binary-Class WESAD Dataset Results")
    for m_metrics, algo in zip(bin_model_metrics, ml_algo):
        print(algo,
              "| Accuracy: {:.2%}".format(m_metrics[0]),
              "| F1-Score: {:.2%}".format(m_metrics[3]),
              "| Precision: {:.2%}".format(m_metrics[1]),
              "| Recall: {:.2%}".format(m_metrics[2]))
        print("Confusion Matrix")
        print(np.matrix(m_metrics[4]), '\n')


def get_e4_features(signal, signal_type):
    """Processing pipeline per signal, pass in a slice of the signal that corresponds to the window.
    Specify the signal that is passed in.
    Parameters
    :param signal: ndarray
        Windowed slice of the original signal
    :param signal_type: str
        Signal type passed in.  BVP, Accelerometer, Temperature, and EDA supported
    :return: list
        List of the features extracted
    """
    features = []
    if signal_type == 'BVP':
        features.extend(float(v) for v in TdFeatures().__generate__(signal.reshape(-1)).values())
        features.extend(float(v) for v in NonLinearFeatures().__generate__(signal.reshape(-1)).values())
        features.extend(float(v) for v in StatFeatures().__generate__(signal.reshape(-1)).values())
        features.extend(float(v) for v in FdFeatures().__generate__(signal.reshape(-1)).values())
        # signal = signal.reshape(-1)
        # m = fe.bvp_signal_processing(signal, 64.0)
        # for key in m:
        #     features.append(m[key])
        # frequency_energies = fe.signal_frequency_band_energies(signal, [[0.01, 0.04], [0.04, 0.15], [0.15, 0.4], [0.4, 1.0]], 64.0)
        # norm_freq = fe.signal_frequency_normalize([frequency_energies[1], frequency_energies[2]])
        # freq_sums = fe.signal_frequency_summation(frequency_energies)
        # freq_powers = fe.signal_relative_power(frequency_energies, 1024 / 2)
        # for freq in freq_sums:
        #     features.append(freq)
        # for freq in freq_powers:
        #     features.append(freq)
        # features.append(np.average(norm_freq[0]))
        # features.append(np.average(norm_freq[1]))
        # features.append(fe.signal_lf_hf_ratio(freq_sums[1], freq_sums[2]))
    elif signal_type == 'ACC':
        features.extend(float(v) for v in get_stats(signal[:, 0].reshape(-1)).values())
        features.extend(float(v) for v in get_stats(signal[:, 1].reshape(-1)).values())
        features.extend(float(v) for v in get_stats(signal[:, 2].reshape(-1)).values())
        # features.append(fe.signal_mean(signal[:, 0].reshape(-1)))
        # features.append(fe.signal_mean(signal[:, 1].reshape(-1)))
        # features.append(fe.signal_mean(signal[:, 2].reshape(-1)))
        # features.append(fe.signal_mean(signal.flatten()))
        # features.append(fe.signal_standard_deviation(signal[:, 0].reshape(-1)))
        # features.append(fe.signal_standard_deviation(signal[:, 1].reshape(-1)))
        # features.append(fe.signal_standard_deviation(signal[:, 2].reshape(-1)))
        # features.append(fe.signal_standard_deviation(signal.reshape(-1)))
        # features.append(fe.signal_absolute(fe.signal_integral(signal[:, 0].reshape(-1))))
        # features.append(fe.signal_absolute(fe.signal_integral(signal[:, 1].reshape(-1))))
        # features.append(fe.signal_absolute(fe.signal_integral(signal[:, 2].reshape(-1))))
        # features.append(fe.signal_absolute(fe.signal_integral(signal.reshape(-1))))
        # features.append(fe.signal_peak_frequency(signal[:, 0].reshape(-1), 32.0))
        # features.append(fe.signal_peak_frequency(signal[:, 1].reshape(-1), 32.0))
        # features.append(fe.signal_peak_frequency(signal[:, 2].reshape(-1), 32.0))
    elif signal_type == 'TEMP':
        features.extend(float(v) for v in get_stats(signal.reshape(-1)).values())
        # signal = signal.reshape(-1)
        # features.append(fe.signal_mean(signal))
        # features.append(fe.signal_standard_deviation(signal))
        # minmax = fe.signal_min_max(signal)
        # features.append(minmax[0])
        # features.append(minmax[1])
        # features.append(fe.signal_dynamic_range(signal))
        # features.append(fe.signal_slope(signal, np.arange(0, len(signal))))
    elif signal_type == 'EDA':
        signal = signal.reshape(-1).ravel()
        r, t = __cvx_eda(signal, 1/4)
        features.extend(float(v) for v in get_stats(np.ravel(t)).values())
        features.extend(float(v) for v in get_stats(np.ravel(r)).values())
        # scr = sp.tarvainen_detrending(signal_lambda=1500, input_signal=signal)
        # scl = sp.get_residual(signal, scr)
        # features.extend(float(v) for v in get_stats(scr).values())
        # features.extend(float(v) for v in get_stats(scl).values())
        # features.append(fe.signal_mean(signal))
        # features.append(fe.signal_standard_deviation(signal))
        # features.append(fe.signal_slope(signal, np.arange(0, len(signal))))
        # features.append(fe.signal_dynamic_range(signal))
        # minmax = fe.signal_min_max(signal)
        # features.append(minmax[0])
        # features.append(minmax[1])
        # features.append(fe.signal_mean(scl))
        # features.append(fe.signal_mean(scr))
        # features.append(fe.signal_standard_deviation(scl))
        # features.append(fe.signal_standard_deviation(scr))
        # features.append(fe.signal_correlation(scl, np.arange(0, len(scl))))
        # features.append(len(fe.signal_peak_count(scr)))
        # features.append(fe.signal_integral(scr))
    return features

