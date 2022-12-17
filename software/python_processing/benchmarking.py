"""
cognitive Load and Subsequent Intervention recognition (cLASIr) Dataset Benchmarking

    This script compares the baseline performance of cLASIr in comparison to the Wearable Stress and Affect Detection
    (WESAD) and Continuously Annotated Signals of Emotion (CASE) datasets using a suite of models including linear
    discriminators, neural networks, nearest neighbor heuristic, ensemble learning, and decision trees in both binary
    and multiclass tasks.

    These three datasets were then mixed together in various permutations to evaluate the affect on learning when the
    different subjects, methodologies, and sensors were used simultaneously.  Finally, all three datasets are combined.

    The reported measures are top-1 accuracy, F1 score, TPR @ 5% FPR, TPR @ 10% FPR, Equal Error Rate, Area Under the
    Receiver Operating Characteristic Curve, and Area Under the Precision Recall Curve.  These metrics are generated
    for each fold of a 10-fold, stratified, identity split, cross validation.  The metrics are reported as the averages
    with standard deviation in a final output Excel spreadsheet for each classifier, separated by binary versus
    multiclass.

    Additionally, raw classification tables and a confusion matrix is generated for the best of the trained models in
    the 10-fold split and permutation feature importance is performed for each fold and ten iterations are performed.

    It takes approximately 48 hours to run this script from start to finish.  Once a folder has been created for the
    classification task in the 'results' folder, the script assumes that task has been completed previously and will
    skip it.

Author: Walker Arce
Date:   7 Sep 2022
"""

import warnings

from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.neural_network import MLPClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, average_precision_score, \
    confusion_matrix, precision_recall_curve, auc, cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from eli5.sklearn import PermutationImportance

import case_interfacing as case
import clasir_interfacing as clasir
import wesad_interfacing as wesad

feature_names = ['hrv_mean_nni', 'hrv_median_nni', 'hrv_range_nni', 'hrv_sdsd', 'hrv_rmssd', 'hrv_nni_50',
                 'hrv_pnni_50', 'hrv_nni_20', 'hrv_pnni_20', 'hrv_cvsd', 'hrv_sdnn', 'hrv_cvnni', 'hrv_mean_hr',
                 'hrv_min_hr', 'hrv_max_hr', 'hrv_std_hr', 'hrv_SD1', 'hrv_SD2', 'hrv_SD2SD1', 'hrv_CSI', 'hrv_CVI',
                 'hrv_CSI_Modified', 'hrv_mean', 'hrv_std', 'hrv_min', 'hrv_max', 'hrv_ptp', 'hrv_sum', 'hrv_energy',
                 'hrv_skewness', 'hrv_kurtosis', 'hrv_peaks', 'hrv_rms', 'hrv_lineintegral', 'hrv_n_above_mean',
                 'hrv_n_below_mean', 'hrv_n_sign_changes', 'hrv_iqr', 'hrv_iqr_5_95', 'hrv_pct_5', 'hrv_pct_95',
                 'hrv_entropy', 'hrv_perm_entropy', 'hrv_svd_entropy', 'hrv_total_power', 'hrv_vlf', 'hrv_lf', 'hrv_hf',
                 'hrv_lf_hf_ratio', 'hrv_lfnu', 'hrv_hfnu', 'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_ptp',
                 'eda_sum', 'eda_energy', 'eda_skewness', 'eda_kurtosis', 'eda_peaks', 'eda_rms', 'eda_lineintegral',
                 'eda_n_above_mean', 'eda_n_below_mean', 'eda_n_sign_changes', 'eda_iqr', 'eda_iqr_5_95', 'eda_pct_5',
                 'eda_pct_95', 'eda_entropy', 'eda_perm_entropy', 'eda_svd_entropy', 'eda_mean', 'eda_std', 'eda_min',
                 'eda_max', 'eda_ptp', 'eda_sum', 'eda_energy', 'eda_skewness', 'eda_kurtosis', 'eda_peaks', 'eda_rms',
                 'eda_lineintegral', 'eda_n_above_mean', 'eda_n_below_mean', 'eda_n_sign_changes', 'eda_iqr',
                 'eda_iqr_5_95', 'eda_pct_5', 'eda_pct_95', 'eda_entropy', 'eda_perm_entropy', 'eda_svd_entropy',
                 'accx_mean', 'accx_std', 'accx_min', 'accx_max', 'accx_ptp', 'accx_sum', 'accx_energy',
                 'accx_skewness', 'accx_kurtosis', 'accx_peaks', 'accx_rms', 'accx_lineintegral', 'accx_n_above_mean',
                 'accx_n_below_mean', 'accx_n_sign_changes', 'accx_iqr', 'accx_iqr_5_95', 'accx_pct_5', 'accx_pct_95',
                 'accx_entropy', 'accx_perm_entropy', 'accx_svd_entropy', 'accy_mean', 'accy_std', 'accy_min',
                 'accy_max', 'accy_ptp', 'accy_sum', 'accy_energy', 'accy_skewness', 'accy_kurtosis', 'accy_peaks',
                 'accy_rms', 'accy_lineintegral', 'accy_n_above_mean', 'accy_n_below_mean', 'accy_n_sign_changes',
                 'accy_iqr', 'accy_iqr_5_95', 'accy_pct_5', 'accy_pct_95', 'accy_entropy', 'accy_perm_entropy',
                 'accy_svd_entropy', 'accz_mean', 'accz_std', 'accz_min', 'accz_max', 'accz_ptp', 'accz_sum',
                 'accz_energy', 'accz_skewness', 'accz_kurtosis', 'accz_peaks', 'accz_rms', 'accz_lineintegral',
                 'accz_n_above_mean', 'accz_n_below_mean', 'accz_n_sign_changes', 'accz_iqr', 'accz_iqr_5_95',
                 'accz_pct_5', 'accz_pct_95', 'accz_entropy', 'accz_perm_entropy', 'accz_svd_entropy', 'tmp_mean',
                 'tmp_std', 'tmp_min', 'tmp_max', 'tmp_ptp', 'tmp_sum', 'tmp_energy', 'tmp_skewness', 'tmp_kurtosis',
                 'tmp_peaks', 'tmp_rms', 'tmp_lineintegral', 'tmp_n_above_mean', 'tmp_n_below_mean',
                 'tmp_n_sign_changes', 'tmp_iqr', 'tmp_iqr_5_95', 'tmp_pct_5', 'tmp_pct_95', 'tmp_entropy',
                 'tmp_perm_entropy', 'tmp_svd_entropy'
                 ]

feature_names_no_acc = ['hrv_mean_nni', 'hrv_median_nni', 'hrv_range_nni', 'hrv_sdsd', 'hrv_rmssd', 'hrv_nni_50',
                        'hrv_pnni_50', 'hrv_nni_20', 'hrv_pnni_20', 'hrv_cvsd', 'hrv_sdnn', 'hrv_cvnni', 'hrv_mean_hr',
                        'hrv_min_hr', 'hrv_max_hr', 'hrv_std_hr', 'hrv_SD1', 'hrv_SD2', 'hrv_SD2SD1', 'hrv_CSI',
                        'hrv_CVI', 'hrv_CSI_Modified', 'hrv_mean', 'hrv_std', 'hrv_min', 'hrv_max', 'hrv_ptp',
                        'hrv_sum', 'hrv_energy', 'hrv_skewness', 'hrv_kurtosis', 'hrv_peaks', 'hrv_rms',
                        'hrv_lineintegral', 'hrv_n_above_mean', 'hrv_n_below_mean', 'hrv_n_sign_changes', 'hrv_iqr',
                        'hrv_iqr_5_95', 'hrv_pct_5', 'hrv_pct_95', 'hrv_entropy', 'hrv_perm_entropy', 'hrv_svd_entropy',
                        'hrv_total_power', 'hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio', 'hrv_lfnu', 'hrv_hfnu',
                        'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_ptp', 'eda_sum', 'eda_energy', 'eda_skewness',
                        'eda_kurtosis', 'eda_peaks', 'eda_rms', 'eda_lineintegral', 'eda_n_above_mean',
                        'eda_n_below_mean', 'eda_n_sign_changes', 'eda_iqr', 'eda_iqr_5_95', 'eda_pct_5', 'eda_pct_95',
                        'eda_entropy', 'eda_perm_entropy', 'eda_svd_entropy', 'eda_mean', 'eda_std', 'eda_min',
                        'eda_max', 'eda_ptp', 'eda_sum', 'eda_energy', 'eda_skewness', 'eda_kurtosis', 'eda_peaks',
                        'eda_rms', 'eda_lineintegral', 'eda_n_above_mean', 'eda_n_below_mean', 'eda_n_sign_changes',
                        'eda_iqr', 'eda_iqr_5_95', 'eda_pct_5', 'eda_pct_95', 'eda_entropy', 'eda_perm_entropy',
                        'eda_svd_entropy', 'tmp_mean', 'tmp_std', 'tmp_min', 'tmp_max', 'tmp_ptp', 'tmp_sum',
                        'tmp_energy', 'tmp_skewness', 'tmp_kurtosis', 'tmp_peaks', 'tmp_rms', 'tmp_lineintegral',
                        'tmp_n_above_mean', 'tmp_n_below_mean', 'tmp_n_sign_changes', 'tmp_iqr', 'tmp_iqr_5_95',
                        'tmp_pct_5', 'tmp_pct_95', 'tmp_entropy', 'tmp_perm_entropy', 'tmp_svd_entropy'
                        ]


def get_dataset_stats(dataset, name):
    dataset_df = pd.DataFrame()
    subjects = dataset[0]
    labels = dataset[1]
    for i in range(0, len(subjects)):
        classes = list(set(labels[i].ravel()))
        subject_data = {f"ID": f"Subject {i}"}
        for c in classes:
            subject_data.update({
                f"Class {c}": len(np.where(labels[i].ravel() == c)[0])
            })
        dataset_df = dataset_df.append(subject_data, ignore_index=True)
    dataset_df.set_index("ID", inplace=True)
    dataset_df.to_excel(os.path.join('dataset_stats', f"{name} Stats.xlsx"))


def get_feature_importance(model, x_test, y_test, fold, fi_df):
    perm = PermutationImportance(model, n_iter=10, cv='prefit').fit(x_test, y_test)
    for i in range(0, len(perm.results_)):
        fi_df[f"{str(model.named_steps['clf'])[0:5]} {fold} Iter {i}"] = perm.results_[i]


def get_classification_table(y_true, y_score):
    classes = list(set(y_true))
    class_table = [[0] * (len(classes) + 1) for _ in range(len(classes))]
    for y, y_s in zip(y_true, y_score):
        class_table[int(y)][0] += 1
        class_table[int(y)][y_s + 1] += 1
    rows = [f"Class {i}" for i in classes]
    cols = [" "] + rows
    return pd.DataFrame(class_table, index=rows, columns=cols)


def calculate_auprc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)


def calculate_tpr_10_per(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    tpr_at_1_fpr = tpr[np.abs(fpr - 0.1).argmin()]
    return tpr_at_1_fpr


def calculate_tpr_5_per(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    tpr_at_5_fpr = tpr[np.abs(fpr - 0.05).argmin()]
    return tpr_at_5_fpr


def calculate_eer(y_true, y_score):
    """
    Returns the equal error rate for a binary classifier output.
    https://github.com/scikit-learn/scikit-learn/issues/15247#issuecomment-542138349
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def stratifiedgroupkfold_test():
    clasir_multi, clasir_binary = clasir.windowed_feature_extraction(window_size)
    groups = []
    datasets = clasir_binary
    for i in range(0, len(datasets[0])):
        group = np.ndarray(len(datasets[0][i]))
        group[:] = i
        groups.extend(group)
    k_fold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=1)
    x = np.vstack(datasets[0])
    y = np.vstack(datasets[1]).ravel()
    for train_ix, test_ix in k_fold.split(x, y, groups=groups):
        train_sort = sorted(set(np.array(groups)[train_ix]))
        test_sort = sorted(set(np.array(groups)[test_ix]))
        print(f"Overlapping Identities: {np.any(train_sort == test_sort)}")
        print(f"Train Sorted: {np.all(sorted(train_ix) == train_ix)}")
        print(f"Training Identities: {len(train_sort)}")
        print(f"Test Sorted: {np.all(sorted(test_ix) == test_ix)}")
        print(f"Testing Identities: {len(test_sort)}")
        print(
            f"Train / Test Percentage: {(len(train_ix) / len(x)) * 100:.2f} % / {(len(test_ix) / len(x)) * 100:.2f} %")
        print()


def evaluate_pretrained_models(clfs, datasets, experiment_name, report_df, features, binary=True):
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
        x_test = np.vstack((datasets[0]))
        y_test = np.vstack((datasets[1])).ravel()
        feature_importance_df = dict()
        if binary:
            model_df = dict(
                (sub, []) for sub in
                ['F1 Score', 'Accuracy', 'AUC', 'AP', 'AUPRC', 'TPR@1%FPR', 'TPR@5%FPR', 'EER', 'Model'])
        else:
            model_df = dict(
                (sub, []) for sub in
                ['F1 Score', 'Accuracy', 'mAUC', 'mAP', 'mAUPRC', 'mTPR@1%FPR', 'mTPR@5%FPR', 'mEER', 'Model'])
        for clf in clfs:
            print(f"Testing {str(clf)} on {experiment_name}")
            y_hat_probs = clf.predict_proba(x_test)
            y_hat = np.argmax(y_hat_probs, axis=1)
            if features is not None:
                get_feature_importance(clf, x_test, y_test, 'eval', feature_importance_df)
            if binary:
                model_df['AP'].append(average_precision_score(y_test, y_hat_probs[:, 1], average='weighted'))
                model_df['AUC'].append(roc_auc_score(y_test, y_hat_probs[:, 1], average='weighted'))
                model_df['EER'].append(calculate_eer(y_test, y_hat_probs[:, 1]))
                model_df['TPR@1%FPR'].append(calculate_tpr_10_per(y_test, y_hat_probs[:, 1]))
                model_df['TPR@5%FPR'].append(calculate_tpr_5_per(y_test, y_hat_probs[:, 1]))
                model_df['AUPRC'].append(calculate_auprc(y_test, y_hat_probs[:, 1]))
            else:
                y_score = label_binarize(y_test, classes=[0, 1, 2])
                mean_average_precision = 0
                eer = 0
                tpr_1_per, tpr_5_per = 0, 0
                roc_auc = 0
                pr_auc = 0
                for i in range(3):
                    tpr_1_per += calculate_tpr_10_per(y_score[:, i], y_hat_probs[:, i])
                    tpr_5_per += calculate_tpr_5_per(y_score[:, i], y_hat_probs[:, i])
                    eer += calculate_eer(y_score[:, i], y_hat_probs[:, i])
                    mean_average_precision += average_precision_score(y_score[:, i], y_hat_probs[:, i],
                                                                      average='weighted')
                    roc_auc += roc_auc_score(y_score[:, i], y_hat_probs[:, i], average='weighted')
                    pr_auc += calculate_auprc(y_score[:, i], y_hat_probs[:, i])
                model_df['mAP'].append(float(mean_average_precision) / 3.0)
                model_df['mEER'].append(float(eer) / 3.0)
                model_df['mTPR@1%FPR'].append(float(tpr_1_per) / 3.0)
                model_df['mTPR@5%FPR'].append(float(tpr_5_per) / 3.0)
                model_df['mAUC'].append(float(roc_auc) / 3.0)
                model_df['mAUPRC'].append(float(pr_auc) / 3.0)
            model_df['Accuracy'].append(accuracy_score(y_test, y_hat))
            model_df['F1 Score'].append(f1_score(y_test, y_hat, average='weighted'))
            model_df['Model'].append(str(clf))
        pd.DataFrame(model_df).to_csv(
            os.path.join('results', str(experiment_name), f'{experiment_name}_evaluating.csv'))
        if features is not None:
            features_df = pd.DataFrame(feature_importance_df, index=features).transpose()
            features_df.to_csv(os.path.join('results', str(experiment_name), f'{experiment_name}_fi.csv'))
            features_df.describe().to_csv(
                os.path.join(os.path.join('results', str(experiment_name), f'{experiment_name}_fi_explain.csv')))
    clf_results = {f"Experiment": experiment_name}
    results_df = pd.read_csv(os.path.join('results', str(experiment_name), f'{experiment_name}_evaluating.csv'))
    features_df = pd.read_csv(os.path.join('results', str(experiment_name), f'{experiment_name}_fi.csv'))
    features_df.describe().to_csv(
        os.path.join(os.path.join('results', str(experiment_name), f'{experiment_name}_fi_explain.csv')))
    clf_names = ['LDA', 'KNC', 'DTC', 'RFC', 'MLP']
    for i, row in results_df.iterrows():
        clf_name = clf_names[i]
        if binary:
            clf_results.update({
                f"{clf_name} Accuracy": f"{np.mean(results_df['Accuracy']) * 100:.2f} \u00B1 {np.std(results_df['Accuracy']) * 100:.2f}",
                f"{clf_name} F1 Score": f"{np.mean(results_df['F1 Score']) * 100:.2f} \u00B1 {np.std(results_df['F1 Score']) * 100:.2f}",
                f"{clf_name} AUC": f"{np.mean(results_df['AUC']) * 100:.2f} \u00B1 {np.std(results_df['AUC']) * 100:.2f}",
                f"{clf_name} AP": f"{np.mean(results_df['AP']) * 100:.2f} \u00B1 {np.std(results_df['AP']) * 100:.2f}",
                f"{clf_name} AUPRC": f"{np.mean(results_df['AUPRC']) * 100:.2f} \u00B1 {np.std(results_df['AUPRC']) * 100:.2f}",
                f"{clf_name} EER": f"{np.mean(results_df['EER']) * 100:.2f} \u00B1 {np.std(results_df['EER']) * 100:.2f}",
                f"{clf_name} TPR@10%FPR": f"{np.mean(results_df['TPR@1%FPR']) * 100:.2f} \u00B1 {np.std(results_df['TPR@1%FPR']) * 100:.2f}",
                f"{clf_name} TPR@5%FPR": f"{np.mean(results_df['TPR@5%FPR']) * 100:.2f} \u00B1 {np.std(results_df['TPR@5%FPR']) * 100:.2f}",
            })
        else:
            clf_results.update({
                f"{clf_name} Accuracy": f"{np.mean(results_df['Accuracy']) * 100:.2f} \u00B1 {np.std(results_df['Accuracy']) * 100:.2f}",
                f"{clf_name} F1 Score": f"{np.mean(results_df['F1 Score']) * 100:.2f} \u00B1 {np.std(results_df['F1 Score']) * 100:.2f}",
                f"{clf_name} mAUC": f"{np.mean(results_df['mAUC']) * 100:.2f} \u00B1 {np.std(results_df['mAUC']) * 100:.2f}",
                f"{clf_name} mAP": f"{np.mean(results_df['mAP']) * 100:.2f} \u00B1 {np.std(results_df['mAP']) * 100:.2f}",
                f"{clf_name} mAUPRC": f"{np.mean(results_df['mAUPRC']) * 100:.2f} \u00B1 {np.std(results_df['mAUPRC']) * 100:.2f}",
                f"{clf_name} mEER": f"{np.mean(results_df['mEER']) * 100:.2f} \u00B1 {np.std(results_df['mEER']) * 100:.2f}",
                f"{clf_name} mTPR@10%FPR": f"{np.mean(results_df['mTPR@1%FPR']) * 100:.2f} \u00B1 {np.std(results_df['mTPR@1%FPR']) * 100:.2f}",
                f"{clf_name} mTPR@5%FPR": f"{np.mean(results_df['mTPR@5%FPR']) * 100:.2f} \u00B1 {np.std(results_df['mTPR@5%FPR']) * 100:.2f}",
            })
    return report_df.append(clf_results, ignore_index=True)


def train_fresh_models(datasets, experiment_name, report_df, features=None, binary=True):
    rng = np.random.RandomState(0)
    if features is None:
        classifiers = [LinearDiscriminantAnalysis(),
                       KNeighborsClassifier(9, n_jobs=-1),
                       DecisionTreeClassifier(min_samples_split=20, random_state=rng),
                       RandomForestClassifier(min_samples_split=20, n_estimators=100, random_state=rng, n_jobs=-1),
                       MLPClassifier(max_iter=1000)]
    else:
        classifiers = [LinearDiscriminantAnalysis(),
                       DecisionTreeClassifier(min_samples_split=20, random_state=rng),
                       RandomForestClassifier(min_samples_split=20, n_estimators=100, random_state=rng, n_jobs=-1),
                       MLPClassifier(max_iter=1000)]
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
        groups = []
        for i in range(0, len(datasets[0])):
            group = np.ndarray(len(datasets[0][i]))
            group[:] = i
            groups.extend(group)
        x = np.vstack(datasets[0])
        y = np.vstack(datasets[1]).ravel()

        best_models = []
        for clf in classifiers:
            fold = 1
            feature_importance_df = dict()
            k_fold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=rng)
            if binary:
                k_fold_df = dict((sub, []) for sub in
                                 ['F1 Score', 'Accuracy', 'AUC', 'AP', 'AUPRC', 'Kappa', 'TPR@1%FPR', 'TPR@5%FPR',
                                  'EER',
                                  'Fitted Models', 'Test IX'])
            else:
                k_fold_df = dict(
                    (sub, []) for sub in
                    ['F1 Score', 'Accuracy', 'mAUC', 'mAP', 'mAUPRC', 'Kappa', 'mTPR@1%FPR', 'mTPR@5%FPR', 'mEER',
                     'Fitted Models', 'Test IX'])
            print(f"Testing {str(clf)} on {experiment_name}")
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('clf', clf)])
            for train_ix, test_ix in k_fold.split(x, y, groups=groups):
                print(f"\tRunning fold {fold} of 10")
                k_fold_df['Test IX'].append(test_ix)
                train_x, x_test = x[train_ix], x[test_ix]
                train_y, y_test = y[train_ix], y[test_ix]
                model = pipeline.fit(train_x, train_y)
                y_hat_probs = model.predict_proba(x_test)
                y_hat = np.argmax(y_hat_probs, axis=1)

                if binary:
                    k_fold_df['AP'].append(average_precision_score(y_test, y_hat_probs[:, 1], average='weighted'))
                    k_fold_df['AUC'].append(roc_auc_score(y_test, y_hat_probs[:, 1], average='weighted'))
                    k_fold_df['EER'].append(calculate_eer(y_test, y_hat_probs[:, 1]))
                    k_fold_df['TPR@1%FPR'].append(calculate_tpr_10_per(y_test, y_hat_probs[:, 1]))
                    k_fold_df['TPR@5%FPR'].append(calculate_tpr_5_per(y_test, y_hat_probs[:, 1]))
                    k_fold_df['AUPRC'].append(calculate_auprc(y_test, y_hat_probs[:, 1]))
                else:
                    y_score = label_binarize(y_test, classes=[0, 1, 2])
                    mean_average_precision = 0
                    eer = 0
                    tpr_1_per, tpr_5_per = 0, 0
                    roc_auc = 0
                    pr_auc = 0
                    for i in range(3):
                        tpr_1_per += calculate_tpr_10_per(y_score[:, i], y_hat_probs[:, i])
                        tpr_5_per += calculate_tpr_5_per(y_score[:, i], y_hat_probs[:, i])
                        eer += calculate_eer(y_score[:, i], y_hat_probs[:, i])
                        mean_average_precision += average_precision_score(y_score[:, i], y_hat_probs[:, i],
                                                                          average='weighted')
                        roc_auc += roc_auc_score(y_score[:, i], y_hat_probs[:, i], average='weighted')
                        pr_auc += calculate_auprc(y_score[:, i], y_hat_probs[:, i])
                    k_fold_df['mAP'].append(float(mean_average_precision) / 3.0)
                    k_fold_df['mEER'].append(float(eer) / 3.0)
                    k_fold_df['mTPR@1%FPR'].append(float(tpr_1_per) / 3.0)
                    k_fold_df['mTPR@5%FPR'].append(float(tpr_5_per) / 3.0)
                    k_fold_df['mAUC'].append(float(roc_auc) / 3.0)
                    k_fold_df['mAUPRC'].append(float(pr_auc) / 3.0)
                k_fold_df['Accuracy'].append(accuracy_score(y_test, y_hat))
                k_fold_df['Kappa'].append(cohen_kappa_score(y_test, y_hat))
                k_fold_df['F1 Score'].append(f1_score(y_test, y_hat, average='weighted'))
                k_fold_df['Fitted Models'].append(model)
                fold += 1
            if binary:
                best_models.append(k_fold_df['Fitted Models'][np.argmax(k_fold_df['AP'])])
                fi_ix = k_fold_df['Test IX'][np.argmax(k_fold_df['AP'])]
            else:
                best_models.append(k_fold_df['Fitted Models'][np.argmax(k_fold_df['mAP'])])
                fi_ix = k_fold_df['Test IX'][np.argmax(k_fold_df['mAP'])]
            get_classification_table(y, np.argmax(best_models[-1].predict_proba(x), axis=1)).to_csv(
                os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_best_class_table.csv'))
            pd.DataFrame(confusion_matrix(y, np.argmax(best_models[-1].predict_proba(x), axis=1))).to_csv(
                os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_best_conf_mat.csv'))
            pd.DataFrame(k_fold_df).to_csv(
                os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_training.csv'))
            if features is not None:
                get_feature_importance(best_models[-1], x[fi_ix], y[fi_ix], 'train', feature_importance_df)
                features_df = pd.DataFrame(feature_importance_df, index=features).transpose()
                features_df.to_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_fi.csv'))
                features_df.describe().to_csv(
                    os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_fi_explain.csv'))
        with open(os.path.join('results', str(experiment_name), f'{experiment_name}_models.pkl'), 'wb') as f:
            pickle.dump(best_models, f)
    with open(os.path.join('results', str(experiment_name), f'{experiment_name}_models.pkl'), 'rb') as f:
        best_models = pickle.load(f)
    clf_results = {f"Experiment": experiment_name}
    for clf in classifiers:
        features_df = pd.read_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_fi.csv'))
        features_df.describe().to_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_fi_explain.csv'))
        clf_name = ''.join(filter(str.isupper, str(clf).split('(')[0]))
        results_df = pd.read_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_training.csv'))
        if binary:
            clf_results.update({
                f"{clf_name} Accuracy": f"{np.mean(results_df['Accuracy']) * 100:.2f} \u00B1 {np.std(results_df['Accuracy']) * 100:.2f}",
                f"{clf_name} F1 Score": f"{np.mean(results_df['F1 Score']) * 100:.2f} \u00B1 {np.std(results_df['F1 Score']) * 100:.2f}",
                f"{clf_name} AUC": f"{np.mean(results_df['AUC']) * 100:.2f} \u00B1 {np.std(results_df['AUC']) * 100:.2f}",
                f"{clf_name} AP": f"{np.mean(results_df['AP']) * 100:.2f} \u00B1 {np.std(results_df['AP']) * 100:.2f}",
                f"{clf_name} Kappa": f"{np.mean(results_df['Kappa']):.2f} \u00B1 {np.std(results_df['Kappa']):.2f}",
                f"{clf_name} AUPRC": f"{np.mean(results_df['AUPRC']) * 100:.2f} \u00B1 {np.std(results_df['AUPRC']) * 100:.2f}",
                f"{clf_name} EER": f"{np.mean(results_df['EER']) * 100:.2f} \u00B1 {np.std(results_df['EER']) * 100:.2f}",
                f"{clf_name} TPR@10%FPR": f"{np.mean(results_df['TPR@1%FPR']) * 100:.2f} \u00B1 {np.std(results_df['TPR@1%FPR']) * 100:.2f}",
                f"{clf_name} TPR@5%FPR": f"{np.mean(results_df['TPR@5%FPR']) * 100:.2f} \u00B1 {np.std(results_df['TPR@5%FPR']) * 100:.2f}",
            })
        else:
            clf_results.update({
                f"{clf_name} Accuracy": f"{np.mean(results_df['Accuracy']) * 100:.2f} \u00B1 {np.std(results_df['Accuracy']) * 100:.2f}",
                f"{clf_name} F1 Score": f"{np.mean(results_df['F1 Score']) * 100:.2f} \u00B1 {np.std(results_df['F1 Score']) * 100:.2f}",
                f"{clf_name} mAUC": f"{np.mean(results_df['mAUC']) * 100:.2f} \u00B1 {np.std(results_df['mAUC']) * 100:.2f}",
                f"{clf_name} mAP": f"{np.mean(results_df['mAP']) * 100:.2f} \u00B1 {np.std(results_df['mAP']) * 100:.2f}",
                f"{clf_name} Kappa": f"{np.mean(results_df['Kappa']):.2f} \u00B1 {np.std(results_df['Kappa']):.2f}",
                f"{clf_name} mAUPRC": f"{np.mean(results_df['mAUPRC']) * 100:.2f} \u00B1 {np.std(results_df['mAUPRC']) * 100:.2f}",
                f"{clf_name} mEER": f"{np.mean(results_df['mEER']) * 100:.2f} \u00B1 {np.std(results_df['mEER']) * 100:.2f}",
                f"{clf_name} mTPR@10%FPR": f"{np.mean(results_df['mTPR@1%FPR']) * 100:.2f} \u00B1 {np.std(results_df['mTPR@1%FPR']) * 100:.2f}",
                f"{clf_name} mTPR@5%FPR": f"{np.mean(results_df['mTPR@5%FPR']) * 100:.2f} \u00B1 {np.std(results_df['mTPR@5%FPR']) * 100:.2f}",
            })
    report_df = report_df.append(clf_results, ignore_index=True)
    return report_df, best_models


no_temp_ds = [True, True, True, False]
no_temp_acc_ds = [True, True, False, False]
if __name__ == '__main__':
    # Generate datasets
    window_size = 5
    wesad_r_multi, wesad_r_binary = wesad.respiban_windowed_feature_extraction(window_size, datastreams=no_temp_ds)
    get_dataset_stats(wesad_r_multi, "WESAD Respiban Dataset Multi")
    get_dataset_stats(wesad_r_binary, "WESAD Respiban Dataset Binary")
    wesad_r_noacc_multi, wesad_r_noacc_binary = wesad.respiban_windowed_feature_extraction(window_size, datastreams=no_temp_acc_ds)
    get_dataset_stats(wesad_r_noacc_multi, "WESAD Respiban No Acc Multi")
    get_dataset_stats(wesad_r_noacc_binary, "WESAD Respiban No Acc Binary")

    wesad_multi, wesad_binary = wesad.e4_windowed_feature_extraction(window_size, datastreams=no_temp_ds)
    get_dataset_stats(wesad_multi, "WESAD Dataset Multi")
    get_dataset_stats(wesad_binary, "WESAD Dataset Binary")
    wesad_noacc_multi, wesad_noacc_binary = wesad.e4_windowed_feature_extraction(window_size, datastreams=no_temp_acc_ds)
    get_dataset_stats(wesad_noacc_multi, "WESAD No Acc Multi")
    get_dataset_stats(wesad_noacc_binary, "WESAD No Acc Binary")

    clasir_multi, clasir_binary, clasir_ap = clasir.windowed_feature_extraction(window_size, datastreams=no_temp_ds)
    get_dataset_stats(clasir_multi, "cLASIr Dataset Multi")
    get_dataset_stats(clasir_binary, "cLASIr Dataset Binary")
    get_dataset_stats(clasir_ap, "cLASIr Dataset AP")
    clasir_noacc_multi, clasir_noacc_binary, clasir_noacc_ap = clasir.windowed_feature_extraction(window_size, datastreams=no_temp_acc_ds)
    get_dataset_stats(clasir_noacc_binary, "cLASIr No Acc Binary")
    get_dataset_stats(clasir_noacc_multi, "cLASIr No Acc Multi")
    get_dataset_stats(clasir_noacc_ap, "cLASIr No Acc AP")
    
    case_multi, case_binary = case.windowed_feature_extraction(window_size, datastreams=no_temp_ds)
    get_dataset_stats(case_multi, "CASE Dataset Multi")
    get_dataset_stats(case_binary, "CASE Dataset Binary")

    case_clasir_binary = [clasir_noacc_binary[0] + case_binary[0],
                          clasir_noacc_binary[1] + case_binary[1]]
    get_dataset_stats(case_clasir_binary, "CASE cLASIr Dataset Binary")
    case_clasir_multi = [clasir_noacc_multi[0] + case_multi[0],
                         clasir_noacc_multi[1] + case_multi[1]]
    get_dataset_stats(case_clasir_multi, "CASE cLASIr Dataset Multi")

    clasir_wesad_binary = [clasir_noacc_binary[0] + wesad_noacc_binary[0],
                           clasir_noacc_binary[1] + wesad_noacc_binary[1]]
    get_dataset_stats(clasir_wesad_binary, "cLASIr WESAD Dataset Binary")
    clasir_wesad_multi = [clasir_noacc_multi[0] + wesad_noacc_multi[0],
                          clasir_noacc_multi[1] + wesad_noacc_multi[1]]
    get_dataset_stats(clasir_wesad_multi, "cLASIr WESAD Dataset Multi")

    case_wesad_binary = [wesad_noacc_binary[0] + case_binary[0],
                         wesad_noacc_binary[1] + case_binary[1]]
    get_dataset_stats(case_wesad_binary, "CASE WESAD Dataset Binary")
    case_wesad_multi = [wesad_noacc_multi[0] + case_multi[0],
                        wesad_noacc_multi[1] + case_multi[1]]
    get_dataset_stats(case_wesad_multi, "CASE WESAD Dataset Multi")

    clasir_respiban_binary = [clasir_noacc_binary[0] + wesad_r_noacc_binary[0],
                              clasir_noacc_binary[1] + wesad_r_noacc_binary[1]]
    get_dataset_stats(clasir_respiban_binary, "cLASIr Respiban Dataset Binary")
    clasir_respiban_multi = [clasir_noacc_multi[0] + wesad_r_noacc_multi[0],
                             clasir_noacc_multi[1] + wesad_r_noacc_multi[1]]
    get_dataset_stats(clasir_respiban_multi, "cLASIr Respiban Dataset Multi")

    case_respiban_binary = [wesad_r_noacc_binary[0] + case_binary[0],
                            wesad_r_noacc_binary[1] + case_binary[1]]
    get_dataset_stats(case_respiban_binary, "CASE Respiban Dataset Binary")
    case_respiban_multi = [wesad_r_noacc_multi[0] + case_multi[0],
                           wesad_r_noacc_multi[1] + case_multi[1]]
    get_dataset_stats(case_respiban_multi, "CASE Respiban Dataset Multi")

    case_clasir_wesad_binary = [
        clasir_noacc_binary[0] + case_binary[0] + wesad_noacc_binary[0],
        clasir_noacc_binary[1] + case_binary[1] + wesad_noacc_binary[1]
    ]
    get_dataset_stats(case_clasir_wesad_binary, "CASE cLASIr WESAD Dataset Binary")
    case_clasir_wesad_multi = [
        clasir_noacc_multi[0] + case_multi[0] + wesad_noacc_multi[0],
        clasir_noacc_multi[1] + case_multi[1] + wesad_noacc_multi[1]
    ]
    get_dataset_stats(case_clasir_wesad_multi, "CASE cLASIr WESAD Dataset Multi")

    full_dataset_binary = [
        clasir_noacc_binary[0] + case_binary[0] + wesad_noacc_binary[0] + wesad_r_noacc_binary[0],
        clasir_noacc_binary[1] + case_binary[1] + wesad_noacc_binary[1] + wesad_r_noacc_binary[1]
    ]
    get_dataset_stats(full_dataset_binary, "Full Dataset Binary")
    full_dataset_multi = [
        clasir_noacc_multi[0] + case_multi[0] + wesad_noacc_multi[0] + wesad_r_noacc_multi[0],
        clasir_noacc_multi[1] + case_multi[1] + wesad_noacc_multi[1] + wesad_r_noacc_multi[1]
    ]
    get_dataset_stats(full_dataset_multi, "Full Dataset Multi")

    # Collect data into DataFrame
    multi_results_df = pd.DataFrame()
    binary_results_df = pd.DataFrame()

    multi_eval_df = pd.DataFrame()
    binary_eval_df = pd.DataFrame()

    # Perform classic benchmarking with SciKit Learn built in models on each dataset
    binary_results_df, wesad_r_binary_alone = train_fresh_models(wesad_r_binary, 'WESAD Respiban Binary Task',
                                                                 binary_results_df,
                                                                 feature_names)
    multi_results_df, wesad_r_multi_alone = train_fresh_models(wesad_r_multi, 'WESAD Respiban Multiclass Task',
                                                               multi_results_df,
                                                               feature_names, binary=False)

    binary_results_df, wesad_binary_alone = train_fresh_models(wesad_binary, 'WESAD Binary Task', binary_results_df,
                                                               feature_names)
    multi_results_df, wesad_multi_alone = train_fresh_models(wesad_multi, 'WESAD Multiclass Task', multi_results_df,
                                                             feature_names, binary=False)

    binary_results_df, clasir_binary_alone = train_fresh_models(clasir_binary, 'cLASIr Binary Task', binary_results_df,
                                                                feature_names)
    multi_results_df, clasir_multi_alone = train_fresh_models(clasir_multi, 'cLASIr Multiclass Task', multi_results_df,
                                                              feature_names, binary=False)

    binary_results_df, _ = train_fresh_models(clasir_ap, 'cLASIr AvP Task', binary_results_df, feature_names)

    # Perform classic benchmarking with SciKit Learn built in models on no accelerometer datasets
    binary_results_df, wesad_r_noacc_binary_alone = train_fresh_models(wesad_r_noacc_binary,
                                                                       'WESAD Respiban Binary Task, No Accelerometer',
                                                                       binary_results_df, feature_names_no_acc)
    multi_results_df, wesad_r_noacc_multi_alone = train_fresh_models(wesad_r_noacc_multi,
                                                                     'WESAD Respiban Multiclass Task, No Accelerometer',
                                                                     multi_results_df, feature_names_no_acc,
                                                                     binary=False)

    binary_results_df, wesad_noacc_binary_alone = train_fresh_models(wesad_noacc_binary,
                                                                     'WESAD Binary Task, No Accelerometer',
                                                                     binary_results_df, feature_names_no_acc)
    multi_results_df, wesad_noacc_multi_alone = train_fresh_models(wesad_noacc_multi,
                                                                   'WESAD Multiclass Task, No Accelerometer',
                                                                   multi_results_df, feature_names_no_acc, binary=False)

    binary_results_df, clasir_noacc_binary_alone = train_fresh_models(clasir_noacc_binary,
                                                                      'cLASIr Binary Task, No Accelerometer',
                                                                      binary_results_df, feature_names_no_acc)
    multi_results_df, clasir_noacc_multi_alone = train_fresh_models(clasir_noacc_multi,
                                                                    'cLASIr Multiclass Task, No Accelerometer',
                                                                    multi_results_df, feature_names_no_acc,
                                                                    binary=False)

    binary_results_df, _ = train_fresh_models(clasir_noacc_ap, 'cLASIr AvP Task, No Accelerometer', binary_results_df,
                                              feature_names_no_acc)

    binary_results_df, case_binary_alone = train_fresh_models(case_binary, 'CASE Binary Task', binary_results_df,
                                                              feature_names_no_acc)
    multi_results_df, case_multi_alone = train_fresh_models(case_multi, 'CASE Multiclass Task', multi_results_df,
                                                            feature_names_no_acc, binary=False)

    # Observe domain shift by using best model for each dataset on the other datasets
    multi_eval_df = evaluate_pretrained_models(wesad_r_multi_alone, clasir_multi, "Respiban on cLASIr, Multi",
                                               multi_eval_df,
                                               feature_names, binary=False)
    binary_eval_df = evaluate_pretrained_models(wesad_r_binary_alone, clasir_binary, "Respiban on cLASIr, Binary",
                                                binary_eval_df, feature_names)

    multi_eval_df = evaluate_pretrained_models(wesad_multi_alone, clasir_multi, "WESAD on cLASIr, Multi", multi_eval_df,
                                               feature_names, binary=False)
    binary_eval_df = evaluate_pretrained_models(wesad_binary_alone, clasir_binary, "WESAD on cLASIr, Binary",
                                                binary_eval_df, feature_names)

    multi_eval_df = evaluate_pretrained_models(clasir_multi_alone, wesad_multi, "cLASIr on WESAD, Multi", multi_eval_df,
                                               feature_names, binary=False)
    binary_eval_df = evaluate_pretrained_models(clasir_binary_alone, wesad_binary, "cLASIr on WESAD, Binary",
                                                binary_eval_df, feature_names)

    binary_eval_df = evaluate_pretrained_models(wesad_noacc_binary_alone, case_clasir_binary, "WESAD on Others, Binary",
                                                binary_eval_df, feature_names_no_acc)
    multi_eval_df = evaluate_pretrained_models(wesad_noacc_multi_alone, case_clasir_multi, "WESAD on Others, Multi",
                                               multi_eval_df, feature_names_no_acc, binary=False)

    binary_eval_df = evaluate_pretrained_models(clasir_noacc_binary_alone, case_wesad_binary,
                                                "cLASIr on Others, Binary", binary_eval_df, feature_names_no_acc)
    multi_eval_df = evaluate_pretrained_models(clasir_noacc_multi_alone, case_wesad_multi, "cLASIr on Others, Multi",
                                               multi_eval_df, feature_names_no_acc, binary=False)

    binary_eval_df = evaluate_pretrained_models(case_binary_alone, clasir_wesad_binary, "CASE on Others, Binary",
                                                binary_eval_df, feature_names_no_acc)
    multi_eval_df = evaluate_pretrained_models(case_multi_alone, clasir_wesad_multi, "CASE on Others, Multi",
                                               multi_eval_df, feature_names_no_acc, binary=False)

    # Perform domain adaptation using dataset mixing, benchmark in the same way
    binary_results_df, case_clasir_mix_binary = train_fresh_models(case_clasir_binary, "CASE cLASIr Transfer, Binary",
                                                                   binary_results_df, feature_names_no_acc)
    multi_results_df, case_clasir_mix_multi = train_fresh_models(case_clasir_multi, "CASE cLASIr Transfer, Multi",
                                                                 multi_results_df, feature_names_no_acc, binary=False)

    binary_results_df, case_wesad_mix_binary = train_fresh_models(case_wesad_binary, "CASE WESAD Transfer, Binary",
                                                                  binary_results_df, feature_names_no_acc)
    multi_results_df, case_wesad_mix_multi = train_fresh_models(case_wesad_multi, "CASE WESAD Transfer, Multi",
                                                                multi_results_df, feature_names_no_acc, binary=False)

    binary_results_df, case_respiban_mix_binary = train_fresh_models(case_respiban_binary,
                                                                     "CASE Respiban Transfer, Binary",
                                                                     binary_results_df, feature_names_no_acc)
    multi_results_df, case_respiban_mix_multi = train_fresh_models(case_respiban_multi, "CASE Respiban Transfer, Multi",
                                                                   multi_results_df, feature_names_no_acc, binary=False)

    binary_results_df, clasir_wesad_mix_binary = train_fresh_models(clasir_wesad_binary,
                                                                    "cLASIr WESAD Transfer, Binary", binary_results_df,
                                                                    feature_names_no_acc)
    multi_results_df, clasir_wesad_mix_multi = train_fresh_models(clasir_wesad_multi, "cLASIr WESAD Transfer, Multi",
                                                                  multi_results_df, feature_names_no_acc, binary=False)

    binary_results_df, clasir_respiban_mix_binary = train_fresh_models(clasir_respiban_binary,
                                                                       "cLASIr Respiban Transfer, Binary",
                                                                       binary_results_df,
                                                                       feature_names_no_acc)
    multi_results_df, clasir_respiban_mix_multi = train_fresh_models(clasir_respiban_multi,
                                                                     "cLASIr Respiban Transfer, Multi",
                                                                     multi_results_df, feature_names_no_acc,
                                                                     binary=False)

    binary_results_df, case_clasir_wesad_mix_binary = train_fresh_models(case_clasir_wesad_binary,
                                                                         "CASE cLASIr WESAD Transfer, Binary",
                                                                         binary_results_df,
                                                                         feature_names_no_acc)
    multi_results_df, case_clasir_wesad_mix_multi = train_fresh_models(case_clasir_wesad_multi,
                                                                       "CASE cLASIr WESAD Transfer, Multi",
                                                                       multi_results_df, feature_names_no_acc,
                                                                       binary=False)

    # Evaluate improved domain shift
    binary_eval_df = evaluate_pretrained_models(case_clasir_mix_binary, wesad_noacc_binary,
                                                "CASE cLASIr on Others, Binary", binary_eval_df, feature_names_no_acc)
    multi_eval_df = evaluate_pretrained_models(case_clasir_mix_multi, wesad_noacc_multi, "CASE cLASIr on Others, Multi",
                                               multi_eval_df, feature_names_no_acc, binary=False)

    binary_eval_df = evaluate_pretrained_models(case_wesad_mix_binary, clasir_respiban_binary,
                                                "CASE WESAD on Others, Binary", binary_eval_df, feature_names_no_acc)
    multi_eval_df = evaluate_pretrained_models(case_wesad_mix_multi, clasir_respiban_multi, "CASE WESAD on Others, Multi",
                                               multi_eval_df, feature_names_no_acc, binary=False)

    binary_eval_df = evaluate_pretrained_models(clasir_wesad_mix_binary, case_respiban_binary, "cLASIr WESAD on Others, Binary",
                                                binary_eval_df, feature_names_no_acc)
    multi_eval_df = evaluate_pretrained_models(clasir_wesad_mix_multi, case_respiban_multi, "cLASIr WESAD on Others, Multi",
                                               multi_eval_df, feature_names_no_acc, binary=False)

    binary_eval_df = evaluate_pretrained_models(clasir_respiban_mix_binary, case_wesad_binary,
                                                "cLASIr Respiban on Others, Binary",
                                                binary_eval_df, feature_names_no_acc)
    multi_eval_df = evaluate_pretrained_models(clasir_respiban_mix_multi, case_wesad_multi,
                                               "cLASIr Respiban on Others, Multi",
                                               multi_eval_df, feature_names_no_acc, binary=False)

    binary_eval_df = evaluate_pretrained_models(case_respiban_mix_binary, clasir_wesad_binary,
                                                "CASE Respiban on Others, Binary",
                                                binary_eval_df, feature_names_no_acc)
    multi_eval_df = evaluate_pretrained_models(case_respiban_mix_multi, clasir_wesad_multi,
                                               "CASE Respiban on Others, Multi",
                                               multi_eval_df, feature_names_no_acc, binary=False)

    binary_eval_df = evaluate_pretrained_models(case_clasir_wesad_mix_binary, wesad_r_noacc_binary,
                                                "CASE cLASIr WESAD on Others, Binary",
                                                binary_eval_df, feature_names_no_acc)
    multi_eval_df = evaluate_pretrained_models(case_clasir_wesad_mix_multi, wesad_r_noacc_multi,
                                               "CASE cLASIr WESAD on Others, Multi",
                                               multi_eval_df, feature_names_no_acc, binary=False)

    # Evaluate using all data, 34 classification tasks total
    binary_results_df, full_mix_binary = train_fresh_models(full_dataset_binary, "Full Dataset, Binary",
                                                            binary_results_df, feature_names_no_acc)
    multi_results_df, full_mix_multi = train_fresh_models(full_dataset_multi, "Full Dataset, Multi", multi_results_df,
                                                          feature_names_no_acc, binary=False)

    # Prepare and save results
    multi_results_df.set_index('Experiment', inplace=True)
    multi_results_df.to_excel(os.path.join('results', 'Multiclass Training Results.xlsx'))

    binary_results_df.set_index('Experiment', inplace=True)
    binary_results_df.to_excel(os.path.join('results', 'Binary Training Results.xlsx'))

    multi_eval_df.set_index('Experiment', inplace=True)
    multi_eval_df.to_excel(os.path.join('results', 'Multiclass Evaluation Results.xlsx'))

    binary_eval_df.set_index('Experiment', inplace=True)
    binary_eval_df.to_excel(os.path.join('results', 'Binary Evaluation Results.xlsx'))
