import numpy as np
from flirt.eda.feature_calculation import __cvx_eda
from flirt.stats.common import get_stats
from scipy.signal import resample
import machine_learning_utils as mlu
import heartpy as hp


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


def get_e4_features(signal, signal_type, sample_rate=64.0):
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
    try:
        if signal_type == 'BVP':
            features.extend(float(v) for v in get_stats(signal).values())
        elif signal_type == 'ACC':
            features.extend(float(v) for v in get_stats(signal).values())
        elif signal_type == 'TMP':
            features.extend(float(v) for v in get_stats(signal).values())
        elif signal_type == 'HRV':
            features.extend(float(v) for v in get_stats(signal).values())
        elif signal_type == 'EDA':
            r, t = __cvx_eda(signal, 1/4)
            features.extend(float(v) for v in get_stats(np.ravel(t)).values())
            features.extend(float(v) for v in get_stats(np.ravel(r)).values())
        return features
    except Exception:
        return features

