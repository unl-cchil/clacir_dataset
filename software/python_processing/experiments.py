"""
Don't forget that line 1135 of venv\Lib\site-packages\sklearn\model_selection\_validation.py was changed from n_classes
"""
import os
import pickle
from itertools import cycle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score, recall_score, precision_score, roc_curve, roc_auc_score, average_precision_score, \
    accuracy_score, mean_absolute_error, balanced_accuracy_score, auc
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import seaborn as sns

import case_interfacing as case
import clasir_interfacing as clasir
import wesad_interfacing as wesad


# import warnings
# warnings.filterwarnings("ignore")


def domain_transfer_train():
    pass


def train_pretrained_models(clf, datasets, experiment_name, threshold=0.7, binary=True):
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
        x_train = np.vstack((datasets[0][0], datasets[0][1]))
        y_train = np.vstack((datasets[1][0], datasets[1][1])).ravel()
        x_test = datasets[0][2]
        y_test = datasets[1][2].ravel()

        if binary:
            score_metrics = (
                'accuracy', 'f1', 'f1_micro', 'f1_macro', 'roc_auc'
            )
        else:
            score_metrics = (
                'accuracy', 'f1_micro', 'f1_macro', 'roc_auc'
            )
        scores = []
        best_model_scores = []
        best_clfs = []
        best_roc_curves = []
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        print(f"Testing {str(clf)} on {experiment_name}")
        scores.append(cross_validate(clf, x_train, y_train, cv=k_fold, return_estimator=True, scoring=score_metrics,
                                     verbose=0, n_jobs=-1))
        training_df = pd.DataFrame(scores[-1])
        training_df.to_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_training_results.csv'))
        # Test best classifier from folds
        best_clf = scores[-1]['estimator'][np.argmax(scores[-1]['test_roc_auc'])]
        best_clfs.append(best_clf)
        y_pred = best_clf.predict_proba(x_test)
        y_pred_t = best_clf.predict(x_test)
        if binary:
            y_pred_thresh = y_pred_t > threshold
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred_t)
            _val_f1 = f1_score(y_test, y_pred_thresh, average='macro', zero_division=0)
            _val_recall = recall_score(y_test, y_pred_thresh, average='macro', zero_division=0)
            _val_precision = precision_score(y_test, y_pred_thresh, average='macro', zero_division=0)
            _val_roc_fpr, _val_roc_tpr, _val_roc_thresh = roc_curve(y_test, y_pred[:, 1])
            _val_roc_auc = roc_auc_score(y_test, y_pred[:, 1], average='weighted')
            _val_map = average_precision_score(y_test, y_pred[:, 1], average='weighted')
            _val_class_f1 = [None, None, None]
            _val_class_recall = [None, None, None]
            _val_class_precision = [None, None, None]
            _val_class_roc_auc = [None, None, None]
        else:
            y_pred_thresh = np.argmax(y_pred, axis=1)
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred_t)
            _val_class_f1 = f1_score(y_test, y_pred_thresh, labels=[0, 1, 2], average=None, zero_division=0)
            _val_f1 = f1_score(y_test, y_pred_thresh, labels=[0, 1, 2], average='weighted', zero_division=0)
            _val_class_recall = recall_score(y_test, y_pred_thresh, labels=[0, 1, 2], average=None, zero_division=0)
            _val_recall = recall_score(y_test, y_pred_thresh, labels=[0, 1, 2], average='weighted', zero_division=0)
            _val_class_precision = precision_score(y_test, y_pred_thresh, labels=[0, 1, 2], average=None,
                                                   zero_division=0)
            _val_precision = precision_score(y_test, y_pred_thresh, labels=[0, 1, 2], average='weighted',
                                             zero_division=0)
            _val_roc_fpr, _val_roc_tpr, _val_roc_thresh = None, None, None
            _val_roc_auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')
            _val_class_roc_auc = roc_auc_score(y_test, y_pred, average=None, multi_class='ovr', labels=[0, 1, 2])
            _val_map = None
        metrics_dict = {
            'Test Accuracy': accuracy,
            'F1 Score': _val_f1,
            'Class F1 Score': _val_class_f1,
            'Recall': _val_recall,
            'Class Recall': _val_class_recall,
            'Precision': _val_precision,
            'Class Precision': _val_class_precision,
            'ROC-AUC': _val_roc_auc,
            'Class ROC-AUC': _val_class_roc_auc,
            'mAP': _val_map,
            'y_test': y_test,
            'y_pred': y_pred,
            'fpr': _val_roc_fpr,
            'tpr': _val_roc_tpr,
            'roc_thresh': _val_roc_thresh,
            'best_clf': best_clf,
            'training_results': scores[-1]
        }
        # Write metrics to file
        results_file = os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(metrics_dict, f)
        best_roc_curves.append([_val_roc_fpr, _val_roc_tpr])
        best_model_scores.append([accuracy, _val_f1, _val_recall, _val_precision, _val_roc_auc, _val_map,
                                  _val_class_f1, _val_class_recall, _val_class_precision, _val_class_roc_auc])
        final_results = [scores, best_model_scores, best_clfs, best_roc_curves]
        overall_best_model = final_results[2][np.argmax([i[4] for i in final_results[1]])]
        write_results(final_results, experiment_name, binary)
        results_file = os.path.join('results', str(experiment_name), f'best_model_trained.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(overall_best_model, f)
    else:
        results_file = os.path.join('results', str(experiment_name), f'best_model_trained.pkl')
        with open(results_file, 'rb') as f:
            overall_best_model = pickle.load(f)
    return overall_best_model


def evaluate_pretrained_models(clf, datasets, experiment_name, threshold=0.7, binary=True):
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
        x_test = np.vstack((i for i in datasets[0]))
        y_test = np.vstack((i for i in datasets[1])).ravel()
        x_test, y_test = shuffle(x_test, y_test)

        y_pred = clf.predict_proba(x_test)
        y_pred_t = clf.predict(x_test)
        if binary:
            y_pred_thresh = y_pred_t > threshold
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred_t)
            _val_f1 = f1_score(y_test, y_pred_thresh, average='macro', zero_division=0)
            _val_recall = recall_score(y_test, y_pred_thresh, average='macro', zero_division=0)
            _val_precision = precision_score(y_test, y_pred_thresh, average='macro', zero_division=0)
            _val_roc_fpr, _val_roc_tpr, _val_roc_thresh = roc_curve(y_test, y_pred[:, 1])
            _val_roc_auc = roc_auc_score(y_test, y_pred[:, 1], average='weighted')
            _val_map = average_precision_score(y_test, y_pred[:, 1], average='weighted')
            _val_class_f1 = [None, None, None]
            _val_class_recall = [None, None, None]
            _val_class_precision = [None, None, None]
            _val_class_roc_auc = [None, None, None]
        else:
            y_pred_thresh = np.argmax(y_pred, axis=1)
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred_t)
            _val_class_f1 = f1_score(y_test, y_pred_thresh, labels=[0, 1, 2], average=None, zero_division=0)
            _val_f1 = f1_score(y_test, y_pred_thresh, labels=[0, 1, 2], average='weighted', zero_division=0)
            _val_class_recall = recall_score(y_test, y_pred_thresh, labels=[0, 1, 2], average=None, zero_division=0)
            _val_recall = recall_score(y_test, y_pred_thresh, labels=[0, 1, 2], average='weighted', zero_division=0)
            _val_class_precision = precision_score(y_test, y_pred_thresh, labels=[0, 1, 2], average=None,
                                                   zero_division=0)
            _val_precision = precision_score(y_test, y_pred_thresh, labels=[0, 1, 2], average='weighted',
                                             zero_division=0)
            _val_roc_fpr, _val_roc_tpr, _val_roc_thresh = None, None, None
            _val_roc_auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')
            _val_class_roc_auc = roc_auc_score(y_test, y_pred, average=None, multi_class='ovr', labels=[0, 1, 2])
            _val_map = None
        metrics_dict = {
            'Test Accuracy': accuracy,
            'F1 Score': _val_f1,
            'Class F1 Score': _val_class_f1,
            'Recall': _val_recall,
            'Class Recall': _val_class_recall,
            'Precision': _val_precision,
            'Class Precision': _val_class_precision,
            'ROC-AUC': _val_roc_auc,
            'Class ROC-AUC': _val_class_roc_auc,
            'mAP': _val_map,
            'y_test': y_test,
            'y_pred': y_pred,
            'fpr': _val_roc_fpr,
            'tpr': _val_roc_tpr,
            'roc_thresh': _val_roc_thresh,
            'clf': clf,
        }
        # Write metrics to file
        results_file = os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(metrics_dict, f)
        final_results = [
            [],
            [[accuracy, _val_f1, _val_recall, _val_precision, _val_roc_auc, _val_map,
              _val_class_f1, _val_class_recall, _val_class_precision, _val_class_roc_auc]],
            [clf],
            [[_val_roc_fpr, _val_roc_tpr]]
        ]
        write_results(final_results, experiment_name, binary)


def multiclass_roc(n_classes, y_test, y_score, clf, experiment_name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw = 2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic for {str(clf)}')
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.savefig(os.path.join('results', str(experiment_name), f'{experiment_name}_multi_roc.png'))
    plt.clf()
    return roc_auc


def train_fresh_models(datasets, experiment_name, binary=True):
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
        x = np.vstack((datasets[0][0], datasets[0][1], datasets[0][2]))
        y = np.vstack((datasets[1][0], datasets[1][1], datasets[1][2])).ravel()

        y_0, y_1 = len(y[y == 0]), len(y[y == 1])
        print(f"{experiment_name} Dataset | Class 0 Exemplars: {y_0} | Class 1 Exemplars: {y_1}")

        classifiers = [LinearDiscriminantAnalysis(),
                       KNeighborsClassifier(9),
                       DecisionTreeClassifier(min_samples_split=20),
                       RandomForestClassifier(min_samples_split=20, n_estimators=100),
                       AdaBoostClassifier(n_estimators=100,
                                          base_estimator=DecisionTreeClassifier(min_samples_split=20)),
                       # SVC(kernel="linear", C=0.025, probability=True),
                       # SVC(gamma=2, C=1, probability=True),
                       # GaussianProcessClassifier(1.0 * RBF(1.0)),
                       # MLPClassifier(alpha=1, max_iter=1000),
                       # GaussianNB(),
                       # QuadraticDiscriminantAnalysis()
                       ]
        k_fold = StratifiedKFold(n_splits=10, shuffle=True)
        for clf in classifiers:
            if binary:
                k_fold_df = dict((sub, []) for sub in ['F1 Score', 'Accuracy', 'AUC'])
            else:
                k_fold_df = dict((sub, []) for sub in ['F1 Score', 'Accuracy', 'AUC 0', 'AUC 1', 'AUC 2'])
            roc_curves = []
            for train_ix, test_ix in k_fold.split(x, y):
                train_x, test_x = x[train_ix], x[test_ix]
                train_y, test_y = y[train_ix], y[test_ix]
                if binary:
                    y_score = label_binarize(test_y, classes=[0, 1])
                else:
                    y_score = label_binarize(test_y, classes=[0, 1, 2])
                model = clf.fit(train_x, train_y)
                y_hat_probs = model.predict_proba(test_x)
                if binary:
                    y_hat = model.predict(train_x)
                else:
                    y_hat = np.argmax(y_hat_probs, axis=1)
                k_fold_df['Accuracy'].append(balanced_accuracy_score(test_y, y_hat))
                k_fold_df['F1 Score'].append(f1_score(test_y, y_hat, average='weighted'))
                if binary:
                    _val_roc_fpr, _val_roc_tpr, _ = roc_curve(y_score, y_hat_probs)
                    k_fold_df['AUC'].append(roc_auc_score(y_score, y_hat_probs, average='weighted'))
                    roc_curves.append((_val_roc_fpr, _val_roc_tpr))
                else:
                    _roc_auc = multiclass_roc(3, test_y, y_score, clf, experiment_name)
                    k_fold_df['AUC 0'].append(_roc_auc[0])
                    k_fold_df['AUC 1'].append(_roc_auc[1])
                    k_fold_df['AUC 2'].append(_roc_auc[2])
            pd.DataFrame(k_fold_df).to_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_training.csv'))
            if binary:
                for i in range(0, len(roc_curves)):
                    plt.plot(roc_curves[0], roc_curves[1])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.legend(loc=4)
                fig = plt.gcf()
                fig.set_size_inches(10, 7)
                plt.title(f"{str(clf)} Mean AUC: {np.mean(k_fold_df['AUC'])}")
                plt.savefig(os.path.join('results', str(experiment_name), f'{experiment_name}_{str(clf)[0:5]}_roc.png'))
                plt.clf()


def write_results(val_results, experiment_name, binary=True):
    models_list = [str(i) for i in val_results[2]]
    if binary:
        for i in range(0, len(val_results[3])):
            plt.plot(val_results[3][i][0], val_results[3][i][1],
                     label=f"{models_list[i]}: AUC={val_results[1][i][4]:.4f}")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        plt.title(f"Best Models ROC for {experiment_name}")
        plt.savefig(os.path.join('results', str(experiment_name), f'{experiment_name}_models_roc.png'))
        plt.clf()
        values = []
        for v in val_results[1]:
            values.append(v[0:6])
        val_df = pd.DataFrame(values,
                              columns=['Accuracy', 'F1 Score', 'Recall', 'Precision', 'ROC AUC', 'mAP'],
                              index=models_list)
        val_df.to_csv(os.path.join('results', str(experiment_name), f'{experiment_name}_best_results.csv'))
    else:
        values = []
        for v in val_results[1]:
            values.append(v[0:5] + list(v[6]) + list(v[7]) + list(v[8]) + list(v[9]))
        val_df = pd.DataFrame(values,
                              columns=['Accuracy', 'F1 Score', 'Recall', 'Precision', 'ROC AUC',
                                       'Class 0 F1 Score', 'Class 1 F1 Score', 'Class 2 F1 Score',
                                       'Class 0 Recall', 'Class 1 Recall', 'Class 2 Recall',
                                       'Class 0 Precision', 'Class 1 Precision', 'Class 2 Precision',
                                       'Class 0 ROC AUC', 'Class 1 ROC AUC', 'Class 2 ROC AUC'],
                              index=models_list)
        val_df.to_csv(os.path.join('results', str(experiment_name), f'{experiment_name}_best_results.csv'))


def get_metrics(datasets, labels, dev_data, dev_labels):
    """Runs a dataset against five machine learning algorithms and returns common metrics for performance
    Parameters
    :param datasets: list
        List of data to be used in training
    :param labels: list
        List of labels to be used in training
    :param dev_data: ndarray
        Array of data used to get accuracy of model
    :param dev_labels: ndarray
        Array of labels used to get accuracy of model
    :return: list
        List of metrics in order of execution
    """
    classifiers = [LinearDiscriminantAnalysis(),
                   KNeighborsClassifier(9),
                   DecisionTreeClassifier(min_samples_split=20),
                   RandomForestClassifier(min_samples_split=20, n_estimators=100),
                   AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(min_samples_split=20))]
    model_metrics = []
    for clf in classifiers:
        for x, y in zip(datasets, labels):
            print(clf, len(x), len(y))
            clf.fit(x, y)
            y_pred = clf.predict(dev_data)
            accuracy = metrics.accuracy_score(dev_labels, y_pred)
            confusion = metrics.confusion_matrix(dev_labels, y_pred)
            precision, recall, f1_score_weighted, _ = metrics.precision_recall_fscore_support(dev_labels, y_pred,
                                                                                              average='weighted')
            model_metrics.append([accuracy, precision, recall, f1_score_weighted, confusion])
    return model_metrics


# Generate datasets
window_size = 5
clasir_noacc_multi, clasir_noacc_binary = clasir.windowed_feature_extraction(window_size, exclude_acc=True,
                                                                             dataset_name='clasir_no_acc')
wesad_noacc_multi, wesad_noacc_binary = wesad.windowed_feature_extraction(window_size, exclude_acc=True,
                                                                          dataset_name='wesad_no_acc')

wesad_multi, wesad_binary = wesad.windowed_feature_extraction(window_size)
clasir_multi, clasir_binary = clasir.windowed_feature_extraction(window_size)

case_multi, case_binary = case.windowed_feature_extraction(window_size)

case_clasir_binary = [clasir_noacc_binary[0] + case_binary[0],
                      clasir_noacc_binary[1] + case_binary[1]]
case_clasir_multi = [clasir_noacc_multi[0] + case_multi[0],
                     clasir_noacc_multi[1] + case_multi[1]]
clasir_wesad_binary = [clasir_noacc_binary[0] + wesad_noacc_binary[0],
                       clasir_noacc_binary[1] + wesad_noacc_binary[1]]
clasir_wesad_multi = [clasir_noacc_multi[0] + wesad_noacc_multi[0],
                      clasir_noacc_multi[1] + wesad_noacc_multi[1]]
case_wesad_binary = [wesad_noacc_binary[0] + case_multi[0],
                     wesad_noacc_binary[1] + case_multi[1]]
case_wesad_multi = [wesad_noacc_multi[0] + case_multi[0],
                    wesad_noacc_multi[1] + case_multi[1]]

full_dataset_binary = [
    clasir_noacc_binary[0] + case_binary[0] + wesad_noacc_binary[0],
    clasir_noacc_binary[1] + case_binary[1] + wesad_noacc_binary[1]
]

full_dataset_multi = [
    clasir_noacc_multi[0] + case_multi[0] + wesad_noacc_multi[0],
    clasir_noacc_multi[1] + case_multi[1] + wesad_noacc_multi[1]
]
# Perform classic benchmarking with SciKit Learn built in models on each dataset
# wesad_binary_alone = train_fresh_models(wesad_binary, 'WESAD Binary Task')
# wesad_multi_alone = train_fresh_models(wesad_multi, 'WESAD Multiclass Task', binary=False)

clasir_binary_alone = train_fresh_models(clasir_wesad_multi, 'cLASIr Binary Task')
# clasir_multi_alone = train_fresh_models(clasir_multi, 'cLASIr Multiclass Task', binary=False)

# Perform classic benchmarking with SciKit Learn built in models on no accelerometer datasets
# wesad_noacc_binary_alone = train_fresh_models(wesad_noacc_binary, 'WESAD Binary Task, No Accelerometer')
# wesad_noacc_multi_alone = train_fresh_models(wesad_noacc_multi, 'WESAD Multiclass Task, No Accelerometer', binary=False)

# clasir_noacc_binary_alone = train_fresh_models(clasir_noacc_binary, 'cLASIr Binary Task, No Accelerometer')
# clasir_noacc_multi_alone = train_fresh_models(clasir_noacc_multi, 'cLASIr Multiclass Task, No Accelerometer',
#                                               binary=False)

# case_binary_alone = train_fresh_models(case_binary, 'CASE Binary Task')
# # case_multi_alone = train_fresh_models(case_multi, 'CASE Multiclass Task', binary=False)
#
# # Observe domain shift by using best model for each dataset on the other datasets
# evaluate_pretrained_models(wesad_multi_alone, clasir_multi, "WESAD on cLASIr, Multi", binary=False)
# evaluate_pretrained_models(wesad_binary_alone, clasir_binary, "WESAD on cLASIr, Binary")
#
# evaluate_pretrained_models(clasir_multi_alone, wesad_multi, "cLASIr on WESAD, Multi", binary=False)
# evaluate_pretrained_models(clasir_binary_alone, wesad_binary, "cLASIr on WESAD, Binary")
#
# evaluate_pretrained_models(wesad_noacc_binary_alone, case_clasir_binary, "WESAD on Others, Binary")
# evaluate_pretrained_models(wesad_noacc_multi_alone, case_clasir_multi, "WESAD on Others, Multi", binary=False)
#
# evaluate_pretrained_models(clasir_noacc_binary_alone, case_wesad_binary, "cLASIr on Others, Binary")
# evaluate_pretrained_models(clasir_noacc_multi_alone, case_wesad_multi, "cLASIr on Others, Multi", binary=False)
#
# evaluate_pretrained_models(case_binary_alone, clasir_wesad_binary, "CASE on Others, Binary")
# evaluate_pretrained_models(case_multi_alone, clasir_wesad_multi, "CASE on Others, Multi", binary=False)
#
# # Perform domain adaptation using dataset mixing, benchmark in the same way
# train_pretrained_models(wesad_noacc_binary_alone, case_clasir_binary, "WESAD Transfer, Binary")
# train_pretrained_models(wesad_noacc_multi_alone, case_clasir_multi, "WESAD Transfer, Multi", binary=False)
#
# train_pretrained_models(clasir_noacc_binary_alone, case_wesad_binary, "cLASIr Transfer, Binary")
# train_pretrained_models(clasir_noacc_multi_alone, case_wesad_multi, "cLASIr Transfer, Multi", binary=False)
#
# train_pretrained_models(case_binary_alone, clasir_wesad_binary, "CASE Transfer, Binary")
# train_pretrained_models(case_multi_alone, clasir_wesad_multi, "CASE Transfer, Multi", binary=False)

# Perform domain adaptation using a trainable transform layer, benchmark in the same way
# https://stackoverflow.com/a/47520976
