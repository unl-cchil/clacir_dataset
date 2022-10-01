import csv
import os

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_curve, roc_auc_score, average_precision_score, \
    accuracy_score
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import case_interfacing as case
import clasir_interfacing as clasir
import wesad_interfacing as wesad


def domain_transfer_train():
    pass


def train_pretrained_models(clf, datasets, experiment_name, threshold=0.7):
    x_train = datasets[0][0] + datasets[0][1]
    y_train = datasets[1][0] + datasets[1][1]
    x_test = datasets[0][2]
    y_test = datasets[1][2]
    score_metrics = (
        'accuracy', 'f1', 'f1_micro', 'f1_macro', 'roc_auc'
    )
    scores = cross_validate(clf, x_train, y_train,
                            cv=10, return_estimator=True, scoring=score_metrics)
    # TODO: Get scores on eval set using best model
    best_clf = scores[-1]['estimator'][scores[-1]['accuracy'].index(max(scores[-1]['accuracy']))]
    y_pred = cross_val_predict(best_clf, x_test, y_test, cv=10)
    y_pred_thresh = y_pred > threshold
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    _val_f1 = f1_score(y_test, y_pred_thresh, average='macro', zero_division=0)
    _val_recall = recall_score(y_test, y_pred_thresh, average='macro', zero_division=0)
    _val_precision = precision_score(y_test, y_pred_thresh, average='macro', zero_division=0)
    _val_roc_fpr, _val_roc_tpr, _val_roc_thresh = roc_curve(y_test, y_pred)
    _val_roc_auc = roc_auc_score(y_test, y_pred, average='weighted')
    _val_map = average_precision_score(y_test, y_pred)
    # Write metrics to file
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
    results_csv_file = os.path.join('results', str(experiment_name), 'results.csv')
    pred_csv_file = os.path.join('results', str(experiment_name), 'pred.csv')
    roc_csv_file = os.path.join('results', str(experiment_name), 'roc.csv')
    results_headers = ['Test Accuracy', 'F1 Score', 'Recall', 'Precision', 'ROC-AUC', 'mAP']
    with open(results_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results_headers)
        writer.writerow([accuracy, _val_f1, _val_recall, _val_precision, _val_roc_auc, _val_map])
    with open(pred_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(y_test)
        writer.writerow(y_pred)
    with open(roc_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(_val_roc_fpr)
        writer.writerow(_val_roc_tpr)
        writer.writerow(_val_roc_thresh)
    return scores, [accuracy, _val_f1, _val_recall, _val_precision, _val_roc_auc, _val_map]


def evaluate_pretrained_models(clf, datasets, experiment_name, threshold=0.7):
    x = datasets[0][0] + datasets[0][1]
    y_true = datasets[1][0] + datasets[1][1]
    y_pred = cross_val_predict(clf, x, y_true, cv=10)
    y_pred_thresh = y_pred > threshold
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    _val_f1 = f1_score(y_true, y_pred_thresh, average='macro', zero_division=0)
    _val_recall = recall_score(y_true, y_pred_thresh, average='macro', zero_division=0)
    _val_precision = precision_score(y_true, y_pred_thresh, average='macro', zero_division=0)
    _val_roc_fpr, _val_roc_tpr, _val_roc_thresh = roc_curve(y_true, y_pred)
    _val_roc_auc = roc_auc_score(y_true, y_pred, average='weighted')
    _val_map = average_precision_score(y_true, y_pred)
    # Write metrics to file
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
    results_csv_file = os.path.join('results', str(experiment_name), 'results.csv')
    pred_csv_file = os.path.join('results', str(experiment_name), 'pred.csv')
    roc_csv_file = os.path.join('results', str(experiment_name), 'roc.csv')
    results_headers = ['Test Accuracy', 'F1 Score', 'Recall', 'Precision', 'ROC-AUC', 'mAP']
    with open(results_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results_headers)
        writer.writerow([accuracy, _val_f1, _val_recall, _val_precision, _val_roc_auc, _val_map])
    with open(pred_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(y_true)
        writer.writerow(y_pred)
    with open(roc_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(_val_roc_fpr)
        writer.writerow(_val_roc_tpr)
        writer.writerow(_val_roc_thresh)
    return [accuracy, _val_f1, _val_recall, _val_precision, _val_roc_auc, _val_map]


def train_fresh_models(datasets, experiment_name, threshold=0.7):
    x_train = np.vstack((datasets[0][0], datasets[0][1]))
    y_train = np.vstack((datasets[1][0], datasets[1][1]))
    x_test = datasets[0][2]
    y_test = datasets[1][2]
    classifiers = [LinearDiscriminantAnalysis(),
                   KNeighborsClassifier(9),
                   DecisionTreeClassifier(min_samples_split=20),
                   RandomForestClassifier(min_samples_split=20, n_estimators=100),
                   AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(min_samples_split=20))]
    score_metrics = (
        'accuracy', 'f1', 'f1_micro', 'f1_macro', 'roc_auc'
    )
    scores = []
    best_model_scores = []
    best_clfs = []
    for clf in classifiers:
        scores.append(cross_validate(clf, x_train, y_train, cv=10, return_estimator=True, scoring=score_metrics))
        # Test best classifier from folds
        best_clf = scores[-1]['estimator'][np.argmax(scores[-1]['test_accuracy'])]
        best_clfs.append(best_clf)
        y_pred = cross_val_predict(best_clf, x_test, y_test, cv=10)
        y_pred_thresh = y_pred > threshold
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        _val_f1 = f1_score(y_test, y_pred_thresh, average='macro', zero_division=0)
        _val_recall = recall_score(y_test, y_pred_thresh, average='macro', zero_division=0)
        _val_precision = precision_score(y_test, y_pred_thresh, average='macro', zero_division=0)
        _val_roc_fpr, _val_roc_tpr, _val_roc_thresh = roc_curve(y_test, y_pred)
        _val_roc_auc = roc_auc_score(y_test, y_pred, average='weighted')
        _val_map = average_precision_score(y_test, y_pred)
        # Write metrics to file
        if not os.path.exists(os.path.join('results', str(experiment_name))):
            os.mkdir(os.path.join('results', str(experiment_name)))
        results_csv_file = os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_results.csv')
        pred_csv_file = os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_pred.csv')
        roc_csv_file = os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_roc.csv')
        results_headers = ['Test Accuracy', 'F1 Score', 'Recall', 'Precision', 'ROC-AUC', 'mAP']
        with open(results_csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(results_headers)
            writer.writerow([accuracy, _val_f1, _val_recall, _val_precision, _val_roc_auc, _val_map])
        with open(pred_csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(y_test)
            writer.writerow(y_pred)
        with open(roc_csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(_val_roc_fpr)
            writer.writerow(_val_roc_tpr)
            writer.writerow(_val_roc_thresh)
        best_model_scores.append([accuracy, _val_f1, _val_recall, _val_precision, _val_roc_auc, _val_map])
    return scores, best_model_scores, best_clfs


window_size = 5
clasir_noacc_multi, clasir_noacc_binary = clasir.windowed_feature_extraction(window_size, exclude_acc=True,
                                                                             dataset_name='clasir_no_acc')

wesad_noacc_multi, wesad_noacc_binary = wesad.windowed_feature_extraction(window_size, exclude_acc=True,
                                                                          dataset_name='wesad_no_acc')

wesad_multi, wesad_binary = wesad.windowed_feature_extraction(window_size)
clasir_multi, clasir_binary = clasir.windowed_feature_extraction(window_size)

case_multi, case_binary = case.windowed_feature_extraction(window_size)

# Perform classic benchmarking with SciKit Learn built in models on each dataset
clasir_binary_alone = train_fresh_models(clasir_binary, 'clasir_binary_alone')
clasir_multi_alone = train_fresh_models(clasir_multi, 'clasir_multi_alone')

wesad_binary_alone = train_fresh_models(wesad_binary, 'wesad_binary_alone')
wesad_multi_alone = train_fresh_models(wesad_multi, 'wesad_multi_alone')

clasir_noacc_binary_alone = train_fresh_models(clasir_noacc_binary, 'clasir_noacc_binary_alone')
clasir_noacc_multi_alone = train_fresh_models(clasir_noacc_multi, 'clasir_noacc_multi_alone')

wesad_noacc_binary_alone = train_fresh_models(wesad_noacc_binary, 'wesad_noacc_binary_alone')
wesad_noacc_multi_alone = train_fresh_models(wesad_noacc_multi, 'wesad_noacc_multi_alone')

case_binary_alone = train_fresh_models(case_binary, 'case_binary_alone')
case_multi_alone = train_fresh_models(case_multi, 'case_multi_alone')

# Observe domain shift by using best model for each dataset on the other datasets

# Perform domain adaptation using dataset mixing, benchmark in the same way

# Perform domain adaptation using a trainable transform layer, benchmark in the same way
