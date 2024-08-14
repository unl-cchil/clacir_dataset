"""
cognitive Load and Canine Intervention recognition (cLACIr) Dataset Benchmarking

    This script compares the baseline performance of cLACIr in comparison to the Wearable Stress and Affect Detection
    (WESAD) and Continuously Annotated Signals of Emotion (CASE) datasets using a suite of models including linear
    discriminators, neural networks, nearest neighbor heuristic, ensemble learning, and decision trees in both binary
    and multiclass tasks.

    These three datasets were then mixed together in various permutations to evaluate the affect on learning when the
    different subjects, methodologies, and sensors were used simultaneously.  Finally, all three datasets are combined.

    The reported measures are top-1 accuracy, F1 score, TPR @ 5% FPR, TPR @ 10% FPR, Equal Error Rate, Area Under the
    Receiver Operating Characteristic Curve, Cohen's Kappa, and Area Under the Precision Recall Curve.  These metrics
    are generated for each fold of a 10-fold, stratified, identity split, cross validation.  The metrics are reported
    as the averages with standard deviation in a final output Excel spreadsheet for each classifier, separated by
    binary versus multiclass.

    Additionally, raw classification tables and a confusion matrix is generated for the best of the trained models in
    the 10-fold split and permutation feature importance is performed for each fold and ten iterations are performed.

    It takes approximately 48 hours to run this script from start to finish.  Once a folder has been created for the
    classification task in the 'results' folder, the script assumes that task has been completed previously and will
    skip it.

Author:     Walker Arce
Version:    3
Date:       17 Mar 2023
"""
import random
import time
import warnings

from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, average_precision_score, \
    confusion_matrix, precision_recall_curve, auc, cohen_kappa_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from eli5.sklearn import PermutationImportance
from sklearn.utils import shuffle
import case_interfacing as case
import clasir_interfacing as clasir
import wesad_interfacing as wesad

hrv_feature_names = [
    'hrv_mean', 'hrv_std', 'hrv_min', 'hrv_max', 'hrv_ptp', 'hrv_sum', 'hrv_energy',
    'hrv_skewness', 'hrv_kurtosis', 'hrv_peaks', 'hrv_rms', 'hrv_lineintegral', 'hrv_n_above_mean',
    'hrv_n_below_mean', 'hrv_n_sign_changes', 'hrv_iqr', 'hrv_iqr_5_95', 'hrv_pct_5', 'hrv_pct_95',
    'hrv_entropy', 'hrv_perm_entropy', 'hrv_svd_entropy',
]

eda_feature_names = [
    'scl_mean', 'scl_std', 'scl_min', 'scl_max', 'scl_ptp',
    'scl_sum', 'scl_energy', 'scl_skewness', 'scl_kurtosis', 'scl_peaks', 'scl_rms', 'scl_lineintegral',
    'scl_n_above_mean', 'scl_n_below_mean', 'scl_n_sign_changes', 'scl_iqr', 'scl_iqr_5_95', 'scl_pct_5',
    'scl_pct_95', 'scl_entropy', 'scl_perm_entropy', 'scl_svd_entropy',

    'scr_mean', 'scr_std', 'scr_min',
    'scr_max', 'scr_ptp', 'scr_sum', 'scr_energy', 'scr_skewness', 'scr_kurtosis', 'scr_peaks', 'scr_rms',
    'scr_lineintegral', 'scr_n_above_mean', 'scr_n_below_mean', 'scr_n_sign_changes', 'scr_iqr',
    'scr_iqr_5_95', 'scr_pct_5', 'scr_pct_95', 'scr_entropy', 'scr_perm_entropy', 'scr_svd_entropy',
]

acc_feature_names = [
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
    'accz_pct_5', 'accz_pct_95', 'accz_entropy', 'accz_perm_entropy', 'accz_svd_entropy'
]

feature_names = [
    hrv_feature_names + eda_feature_names + acc_feature_names
]

feature_names_no_acc = [
    hrv_feature_names + eda_feature_names
]

feature_names_no_acc_eda = [
    eda_feature_names
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


def evaluate_pretrained_models(clfs, datasets, experiment_name, mda_features, report_df, features, binary=True):
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
        x_test = np.vstack((datasets[0]))
        y_test = np.vstack((datasets[1])).ravel()

        if binary:
            model_df = dict(
                (sub, []) for sub in
                ['F1 Score', 'Accuracy', 'AUC', 'AP', 'AUPRC', 'Kappa', 'TPR@1%FPR', 'TPR@5%FPR', 'EER', 'Model'])
        else:
            model_df = dict(
                (sub, []) for sub in
                ['F1 Score', 'Accuracy', 'mAUC', 'mAP', 'mAUPRC', 'Kappa', 'mTPR@1%FPR', 'mTPR@5%FPR', 'mEER', 'Model'])
        for clf in clfs:
            print(f"Testing {str(clf)} on {experiment_name}")
            y_hat_probs = clf.predict_proba(x_test)
            y_hat = np.argmax(y_hat_probs, axis=1)
            if features is not None:
                feature_importance_df = dict()
                get_feature_importance(clf, x_test, y_test, 'eval', feature_importance_df)
                features_df = pd.DataFrame(feature_importance_df, index=features).transpose()
                features_df.to_csv(
                    os.path.join('results', str(experiment_name),
                                 f'{str(clf.named_steps["clf"])[0:5]}_{experiment_name}_fi.csv'))
                features_df.describe().to_csv(
                    os.path.join(os.path.join('results', str(experiment_name),
                                              f'{str(clf.named_steps["clf"])[0:5]}_{experiment_name}_fi_explain.csv')))
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
            model_df['Accuracy'].append(balanced_accuracy_score(y_test, y_hat))
            model_df['Kappa'].append(cohen_kappa_score(y_test, y_hat))
            model_df['F1 Score'].append(f1_score(y_test, y_hat, average='weighted'))
            model_df['Model'].append(str(clf))
        pd.DataFrame(model_df).to_csv(
            os.path.join('results', str(experiment_name), f'{experiment_name}_evaluating.csv'))
    clf_results = {f"Experiment": experiment_name}
    results_df = pd.read_csv(os.path.join('results', str(experiment_name), f'{experiment_name}_evaluating.csv'))
    clf_names = ['LDA', 'DTC', 'RFC', 'MLP']
    for i, row in results_df.iterrows():
        clf_name = clf_names[i]
        if binary:
            clf_results.update({
                f"{clf_name} Accuracy": f"{row['Accuracy'] * 100:.2f}",
                f"{clf_name} F1 Score": f"{row['F1 Score'] * 100:.2f}",
                f"{clf_name} AUC": f"{row['AUC'] * 100:.2f}",
                f"{clf_name} AP": f"{row['AP'] * 100:.2f}",
                f"{clf_name} AUPRC": f"{row['AUPRC'] * 100:.2f}",
                f"{clf_name} EER": f"{row['EER'] * 100:.2f}",
                f"{clf_name} Kappa": f"{row['Kappa']:.4f}",
                f"{clf_name} TPR@10%FPR": f"{row['TPR@1%FPR'] * 100:.2f}",
                f"{clf_name} TPR@5%FPR": f"{row['TPR@5%FPR'] * 100:.2f}",
            })
        else:
            clf_results.update({
                f"{clf_name} Accuracy": f"{row['Accuracy'] * 100:.2f}",
                f"{clf_name} F1 Score": f"{row['F1 Score'] * 100:.2f}",
                f"{clf_name} mAUC": f"{row['mAUC'] * 100:.2f}",
                f"{clf_name} mAP": f"{row['mAP'] * 100:.2f}",
                f"{clf_name} mAUPRC": f"{row['mAUPRC'] * 100:.2f}",
                f"{clf_name} mEER": f"{row['mEER'] * 100:.2f}",
                f"{clf_name} Kappa": f"{row['Kappa']:.4f}",
                f"{clf_name} mTPR@10%FPR": f"{row['mTPR@1%FPR'] * 100:.2f}",
                f"{clf_name} mTPR@5%FPR": f"{row['mTPR@5%FPR'] * 100:.2f}",
            })
    for clf in clfs:
        features_df = pd.read_csv(os.path.join('results', str(experiment_name),
                                               f'{str(clf.named_steps["clf"])[0:5]}_{experiment_name}_fi.csv'))
        features_df.describe().to_csv(os.path.join('results', str(experiment_name),
                                                   f'{str(clf.named_steps["clf"])[0:5]}_{experiment_name}_fi_explain.csv'))
        features_describe = pd.read_csv(
            os.path.join('results', str(experiment_name),
                         f'{str(clf.named_steps["clf"])[0:5]}_{experiment_name}_fi_explain.csv'))
        top_10 = features_describe.set_index('Unnamed: 0').transpose().nlargest(10, 'mean')
        top_10_names = list(top_10.index)
        top_10_values = [f"{mean:.4f} \u00B1 {std:.4f}" for mean, std in zip(top_10['mean'], top_10['std'])]
        bot_10 = features_describe.set_index('Unnamed: 0').transpose().nsmallest(10, 'mean')
        bot_10_names = list(bot_10.index)
        bot_10_values = [f"{mean:.4f} \u00B1 {std:.4f}" for mean, std in zip(bot_10['mean'], bot_10['std'])]
        features_describe = []
        features_describe.extend(top_10_names)
        features_describe.extend(bot_10_names)
        features_describe.append(f"{experiment_name} {str(clf.named_steps['clf'])[0:5]}")
        features_values = []
        features_values.extend(top_10_values)
        features_values.extend(bot_10_values)
        mda_features.append(features_describe)
        mda_features.append(features_values)
    return report_df.append(clf_results, ignore_index=True)


def train_fresh_models(datasets, experiment_name, report_df, mda_features, features=None, binary=True, folds=10):
    if features is None:
        classifiers = [
            LinearDiscriminantAnalysis(),
            # KNeighborsClassifier(9, n_jobs=-1),
            # DecisionTreeClassifier(min_samples_split=20, random_state=rng),
            # RandomForestClassifier(min_samples_split=20, n_estimators=100, random_state=rng, n_jobs=-1),
            # MLPClassifier(max_iter=1000)
        ]
    else:
        classifiers = [
            LinearDiscriminantAnalysis(),
            # DecisionTreeClassifier(min_samples_split=20, random_state=rng),
            # RandomForestClassifier(min_samples_split=20, n_estimators=100, random_state=rng, n_jobs=-1),
            # MLPClassifier(max_iter=1000)
        ]
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
        groups = []
        for i in range(0, len(datasets[0])):
            group = np.ndarray(len(datasets[0][i]))
            group[:] = i
            groups.extend(group)
        x = np.vstack(datasets[0])
        y = np.vstack(datasets[1]).ravel()
        groups = np.array(groups)

        best_models = []
        for clf in classifiers:
            fold = 1
            feature_importance_df = dict()
            k_fold = GroupKFold(n_splits=folds)
            # k_fold = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=1)
            # k_fold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rng)
            if binary:
                k_fold_df = dict((sub, []) for sub in
                                 ['F1 Score', 'BAccuracy', 'Accuracy', 'Class', 'Baseline',
                                  # 'AUC',
                                  # 'AP', 'AUPRC', 'Kappa', 'TPR@1%FPR',
                                  # 'TPR@5%FPR',
                                  # 'EER',
                                  'Fitted Models', 'Test IX'])
            else:
                k_fold_df = dict(
                    (sub, []) for sub in
                    ['F1 Score',
                     'BAccuracy',
                     'Accuracy',
                     '0 AUC',
                     '0 AP',
                     '0 EER',
                     '0 TPR@1%FPR',
                     '0 TPR@5%FPR',
                     '0 AUPRC',

                     '1 AUC',
                     '1 AP',
                     '1 EER',
                     '1 TPR@1%FPR',
                     '1 TPR@5%FPR',
                     '1 AUPRC',

                     '2 AUC',
                     '2 AP',
                     '2 EER',
                     '2 TPR@1%FPR',
                     '2 TPR@5%FPR',
                     '2 AUPRC',

                     'Kappa',
                     'Fitted Models', 'Test IX'])
            best_models_roc = []

            print(f"Testing {str(clf)} on {experiment_name}")
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('clf', clf)])
            # pipeline = clf
            for train_ix, test_ix in k_fold.split(x, y, groups=groups):
                # for train_ix, test_ix in k_fold.split(x, y):
                print(f"\tRunning fold {fold} of {folds}")

                train_ix = shuffle(train_ix)

                train_x, x_test = x[train_ix], x[test_ix]
                train_y, y_test = y[train_ix], y[test_ix]
                model = pipeline.fit(train_x, train_y)
                y_hat_probs = model.predict_proba(x_test)
                y_hat = np.argmax(y_hat_probs, axis=1)
                best_models_roc.append((y_hat_probs, y_test))
                y_val = y_test
                print("Prediction results")
                print(accuracy_score(y_val, y_hat))
                print(balanced_accuracy_score(y_val, y_hat))
                print(np.unique(groups[test_ix]))
                k_fold_df['BAccuracy'].append(balanced_accuracy_score(y_val, y_hat))
                k_fold_df['Accuracy'].append(accuracy_score(y_val, y_hat))
                # k_fold_df['Kappa'].append(cohen_kappa_score(y_val, y_hat))
                k_fold_df['F1 Score'].append(f1_score(y_val, y_hat, average='weighted'))
                k_fold_df['Test IX'].append(np.unique(groups[test_ix]))
                k_fold_df['Class'].append(np.unique(y_val))
                k_fold_df['Baseline'].append(len(np.where(x_test == 1)[0]) / len(x_test))

                # if binary:
                #     k_fold_df['AP'].append(average_precision_score(y_val, y_hat_probs[:, 1], average='weighted'))
                #     # k_fold_df['AUC'].append(roc_auc_score(y_val, y_hat_probs[:, 1], average='weighted'))
                #     k_fold_df['EER'].append(calculate_eer(y_val, y_hat_probs[:, 1]))
                #     k_fold_df['TPR@1%FPR'].append(calculate_tpr_10_per(y_val, y_hat_probs[:, 1]))
                #     k_fold_df['TPR@5%FPR'].append(calculate_tpr_5_per(y_val, y_hat_probs[:, 1]))
                #     k_fold_df['AUPRC'].append(calculate_auprc(y_val, y_hat_probs[:, 1]))
                # else:
                #     y_score = label_binarize(y_val, classes=[0, 1, 2])
                #     for i in range(3):
                #         tpr_1_per = calculate_tpr_10_per(y_score[:, i], y_hat_probs[:, i])
                #         tpr_5_per = calculate_tpr_5_per(y_score[:, i], y_hat_probs[:, i])
                #         eer = calculate_eer(y_score[:, i], y_hat_probs[:, i])
                #         mean_average_precision = average_precision_score(y_score[:, i], y_hat_probs[:, i],
                #                                                          average='weighted')
                #         roc_auc = roc_auc_score(y_score[:, i], y_hat_probs[:, i], average='weighted')
                #         pr_auc = calculate_auprc(y_score[:, i], y_hat_probs[:, i])
                #         k_fold_df[str(i) + ' AP'].append(float(mean_average_precision))
                #         k_fold_df[str(i) + ' EER'].append(float(eer))
                #         k_fold_df[str(i) + ' TPR@1%FPR'].append(float(tpr_1_per))
                #         k_fold_df[str(i) + ' TPR@5%FPR'].append(float(tpr_5_per))
                #         k_fold_df[str(i) + ' AUC'].append(float(roc_auc))
                #         k_fold_df[str(i) + ' AUPRC'].append(float(pr_auc))

                k_fold_df['Fitted Models'].append(model)
                fold += 1
            # if binary:
            #     best_model = np.argmax(k_fold_df['Kappa'])
            #     best_models.append(k_fold_df['Fitted Models'][best_model])
            #     fi_ix = k_fold_df['Test IX'][best_model]
            #     best_model_roc = best_models_roc[best_model]
            #     with open(os.path.join('results', str(experiment_name), f"{str(clf)[0:5]}_best_predictions.pkl"),
            #               'wb') as f:
            #         pickle.dump(best_model_roc, f)
            # else:
            #     best_model = np.argmax(k_fold_df['Kappa'])
            #     best_models.append(k_fold_df['Fitted Models'][best_model])
            #     fi_ix = k_fold_df['Test IX'][best_model]
            #     best_model_roc = best_models_roc[best_model]
            #     with open(os.path.join('results', str(experiment_name), f"{str(clf)[0:5]}_best_predictions.pkl"),
            #               'wb') as f:
            #         pickle.dump(best_model_roc, f)
            # get_classification_table(y, np.argmax(best_models[-1].predict_proba(x), axis=1)).to_csv(
            #     os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_best_class_table.csv'))
            # pd.DataFrame(confusion_matrix(y, np.argmax(best_models[-1].predict_proba(x), axis=1))).to_csv(
            #     os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_best_conf_mat.csv'))
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
        if features:
            features_df = pd.read_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_fi.csv'))
            features_df.describe().to_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_fi_explain.csv'))
            features_describe = pd.read_csv(
                os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_fi_explain.csv'))
            top_10 = features_describe.set_index('Unnamed: 0').transpose().nlargest(10, 'mean')
            top_10_names = list(top_10.index)
            top_10_values = [f"{mean:.4f} \u00B1 {std:.4f}" for mean, std in zip(top_10['mean'], top_10['std'])]
            bot_10 = features_describe.set_index('Unnamed: 0').transpose().nsmallest(10, 'mean')
            bot_10_names = list(bot_10.index)
            bot_10_values = [f"{mean:.4f} \u00B1 {std:.4f}" for mean, std in zip(bot_10['mean'], bot_10['std'])]
            features_describe = []
            features_describe.extend(top_10_names)
            features_describe.extend(bot_10_names)
            features_describe.append(f"{experiment_name} {str(clf)[0:5]}")
            features_values = []
            features_values.extend(top_10_values)
            features_values.extend(bot_10_values)
            mda_features.append(features_describe)
            mda_features.append(features_values)
        clf_name = ''.join(filter(str.isupper, str(clf).split('(')[0]))
        results_df = pd.read_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_training.csv'))
        # if binary:
        #     clf_results.update({
        #         f"{clf_name} BAccuracy": f"{np.mean(results_df['BAccuracy']) * 100:.2f} \u00B1 {np.std(results_df['BAccuracy']) * 100:.2f}",
        #         f"{clf_name} Accuracy": f"{np.mean(results_df['Accuracy']) * 100:.2f} \u00B1 {np.std(results_df['Accuracy']) * 100:.2f}",
        #         f"{clf_name} F1 Score": f"{np.mean(results_df['F1 Score']) * 100:.2f} \u00B1 {np.std(results_df['F1 Score']) * 100:.2f}",
        #         f"{clf_name} AUC": f"{np.mean(results_df['AUC']) * 100:.2f} \u00B1 {np.std(results_df['AUC']) * 100:.2f}",
        #         f"{clf_name} AP": f"{np.mean(results_df['AP']) * 100:.2f} \u00B1 {np.std(results_df['AP']) * 100:.2f}",
        #         f"{clf_name} Kappa": f"{np.mean(results_df['Kappa']):.2f} \u00B1 {np.std(results_df['Kappa']):.2f}",
        #         f"{clf_name} AUPRC": f"{np.mean(results_df['AUPRC']) * 100:.2f} \u00B1 {np.std(results_df['AUPRC']) * 100:.2f}",
        #         f"{clf_name} EER": f"{np.mean(results_df['EER']) * 100:.2f} \u00B1 {np.std(results_df['EER']) * 100:.2f}",
        #         f"{clf_name} TPR@10%FPR": f"{np.mean(results_df['TPR@1%FPR']) * 100:.2f} \u00B1 {np.std(results_df['TPR@1%FPR']) * 100:.2f}",
        #         f"{clf_name} TPR@5%FPR": f"{np.mean(results_df['TPR@5%FPR']) * 100:.2f} \u00B1 {np.std(results_df['TPR@5%FPR']) * 100:.2f}",
        #     })
        # else:
        #     clf_results.update({
        #         f"{clf_name} BAccuracy": f"{np.mean(results_df['BAccuracy']) * 100:.2f} \u00B1 {np.std(results_df['BAccuracy']) * 100:.2f}",
        #         f"{clf_name} Accuracy": f"{np.mean(results_df['Accuracy']) * 100:.2f} \u00B1 {np.std(results_df['Accuracy']) * 100:.2f}",
        #         f"{clf_name} F1 Score": f"{np.mean(results_df['F1 Score']) * 100:.2f} \u00B1 {np.std(results_df['F1 Score']) * 100:.2f}",
        #         f"{clf_name} Kappa": f"{np.mean(results_df['Kappa']):.2f} \u00B1 {np.std(results_df['Kappa']):.2f}",
        #
        #         f"{clf_name} 0 AUC": f"         {np.mean(results_df['0 AUC']) * 100:.2f} \u00B1         {np.std(results_df['0 AUC']) * 100:.2f}",
        #         f"{clf_name} 0 AP": f"          {np.mean(results_df['0 AP']) * 100:.2f} \u00B1          {np.std(results_df['0 AP']) * 100:.2f}",
        #         f"{clf_name} 0 AUPRC": f"       {np.mean(results_df['0 AUPRC']) * 100:.2f} \u00B1       {np.std(results_df['0 AUPRC']) * 100:.2f}",
        #         f"{clf_name} 0 EER": f"         {np.mean(results_df['0 EER']) * 100:.2f} \u00B1         {np.std(results_df['0 EER']) * 100:.2f}",
        #         f"{clf_name} 0 TPR@10%FPR": f"  {np.mean(results_df['0 TPR@1%FPR']) * 100:.2f} \u00B1   {np.std(results_df['0 TPR@1%FPR']) * 100:.2f}",
        #         f"{clf_name} 0 TPR@5%FPR": f"   {np.mean(results_df['0 TPR@5%FPR']) * 100:.2f} \u00B1   {np.std(results_df['0 TPR@5%FPR']) * 100:.2f}",
        #
        #         f"{clf_name} 1 AUC": f"         {np.mean(results_df['1 AUC']) * 100:.2f} \u00B1         {np.std(results_df['1 AUC']) * 100:.2f}",
        #         f"{clf_name} 1 AP": f"          {np.mean(results_df['1 AP']) * 100:.2f} \u00B1          {np.std(results_df['1 AP']) * 100:.2f}",
        #         f"{clf_name} 1 AUPRC": f"       {np.mean(results_df['1 AUPRC']) * 100:.2f} \u00B1       {np.std(results_df['1 AUPRC']) * 100:.2f}",
        #         f"{clf_name} 1 EER": f"         {np.mean(results_df['1 EER']) * 100:.2f} \u00B1         {np.std(results_df['1 EER']) * 100:.2f}",
        #         f"{clf_name} 1 TPR@10%FPR": f"  {np.mean(results_df['1 TPR@1%FPR']) * 100:.2f} \u00B1   {np.std(results_df['1 TPR@1%FPR']) * 100:.2f}",
        #         f"{clf_name} 1 TPR@5%FPR": f"   {np.mean(results_df['1 TPR@5%FPR']) * 100:.2f} \u00B1   {np.std(results_df['1 TPR@5%FPR']) * 100:.2f}",
        #
        #         f"{clf_name} 2 AUC": f"         {np.mean(results_df['2 AUC']) * 100:.2f} \u00B1         {np.std(results_df['2 AUC']) * 100:.2f}",
        #         f"{clf_name} 2 AP": f"          {np.mean(results_df['2 AP']) * 100:.2f} \u00B1          {np.std(results_df['2 AP']) * 100:.2f}",
        #         f"{clf_name} 2 AUPRC": f"       {np.mean(results_df['2 AUPRC']) * 100:.2f} \u00B1       {np.std(results_df['2 AUPRC']) * 100:.2f}",
        #         f"{clf_name} 2 EER": f"         {np.mean(results_df['2 EER']) * 100:.2f} \u00B1         {np.std(results_df['2 EER']) * 100:.2f}",
        #         f"{clf_name} 2 TPR@10%FPR": f"  {np.mean(results_df['2 TPR@1%FPR']) * 100:.2f} \u00B1   {np.std(results_df['2 TPR@1%FPR']) * 100:.2f}",
        #         f"{clf_name} 2 TPR@5%FPR": f"   {np.mean(results_df['2 TPR@5%FPR']) * 100:.2f} \u00B1   {np.std(results_df['2 TPR@5%FPR']) * 100:.2f}",
        #     })
    report_df = report_df.append(clf_results, ignore_index=True)
    return report_df, best_models


no_temp_ds = [False, True, True, False]
no_temp_acc_ds = [False, True, False, False]
if __name__ == '__main__':
    # Record start time
    start = time.time()
    # Fix the random seed
    random.seed(42)
    # Generate datasets
    window_size = 5
    folds = 70

    clasir_multi, clasir_binary, clasir_ap, clasir_int = clasir.windowed_feature_extraction(window_size,
                                                                                            datastreams=no_temp_ds)
    get_dataset_stats(clasir_multi, "cLASIr Dataset Multi")
    get_dataset_stats(clasir_binary, "cLASIr Dataset Binary")
    get_dataset_stats(clasir_ap, "cLASIr Dataset AP")
    get_dataset_stats(clasir_int, "cLASIr Dataset Int")
    clasir_noacc_multi, clasir_noacc_binary, clasir_noacc_ap, clasir_noacc_int = clasir.windowed_feature_extraction(
        window_size,
        datastreams=no_temp_acc_ds)
    get_dataset_stats(clasir_noacc_binary, "cLASIr No Acc Binary")
    get_dataset_stats(clasir_noacc_multi, "cLASIr No Acc Multi")
    get_dataset_stats(clasir_noacc_ap, "cLASIr No Acc AP")
    get_dataset_stats(clasir_noacc_int, "cLASIr No Acc Int")

    # Collect data into DataFrame
    multi_results_df = pd.DataFrame()
    binary_results_df = pd.DataFrame()

    multi_eval_df = pd.DataFrame()
    binary_eval_df = pd.DataFrame()

    top_features = []
    top_training_features = []

    # Perform classic benchmarking with SciKit Learn built in models on each dataset
    # binary_results_df, _ = train_fresh_models(clasir_ap, 'cLASIr AvP Task', binary_results_df, top_training_features,
    #                                           feature_names)
    # binary_results_df, _ = train_fresh_models(clasir_int, 'cLASIr Int Task', binary_results_df, top_training_features,
    #                                           eda_feature_names + acc_feature_names)

    # Perform classic benchmarking with SciKit Learn built in models on no accelerometer datasets
    binary_results_df, _ = train_fresh_models(clasir_noacc_ap, 'cLASIr AvP Task, No Accelerometer', binary_results_df,
                                              top_training_features,
                                              None, folds=folds)
    binary_results_df, _ = train_fresh_models(clasir_noacc_int, 'cLASIr Int Task, No Accelerometer', binary_results_df,
                                              top_training_features,
                                              None, folds=folds)
    # binary_results_df, _ = train_fresh_models(clasir_noacc_binary, 'cLASIr Binary Task, No Accelerometer', binary_results_df,
    #                                           top_training_features,
    #                                           None, folds=folds)
    # Prepare and save results
    binary_results_df.set_index('Experiment', inplace=True)
    binary_results_df.to_excel(os.path.join('results', 'Binary Training Results.xlsx'))

    pd.DataFrame(top_features).to_excel(os.path.join('results', 'Top Evaluation Features.xlsx'))
    pd.DataFrame(top_training_features).to_excel(os.path.join('results', 'Top Training Features.xlsx'))

    # Record script time
    seconds = time.time() - start
    minutes = seconds / 60.0
    hours = minutes / 60.0
    days = hours / 24.0
    print(f"\n\nThis process completed in {int(days)}:{int(hours) % 24}:{int(minutes) % 60}:{int(seconds) % 60}")
