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
    accuracy_score, mean_absolute_error, balanced_accuracy_score, auc, classification_report, make_scorer
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold, cross_val_score, \
    StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import seaborn as sns

import case_interfacing as case
import clasir_interfacing as clasir
import wesad_interfacing as wesad


def domain_transfer_train():
    pass


def train_pretrained_models(clf, datasets, experiment_name, binary=True):
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        # os.mkdir(os.path.join('results', str(experiment_name)))
        pass


def evaluate_pretrained_models(clfs, datasets, experiment_name, binary=True):
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
        x_test = np.vstack((datasets[0]))
        y_test = np.vstack((datasets[1])).ravel()
        if binary:
            model_df = dict((sub, []) for sub in ['F1 Score', 'Accuracy', 'AUC', 'Model'])
        else:
            model_df = dict(
                (sub, []) for sub in ['F1 Score', 'Accuracy', 'AUC 0', 'AUC 1', 'AUC 2', 'Model'])
        for clf in clfs:
            print(f"Testing {str(clf)} on {experiment_name}")
            y_hat_probs = clf.predict_proba(x_test)
            y_hat = np.argmax(y_hat_probs, axis=1)
            if binary:
                model_df['AUC'].append(roc_auc_score(y_test, y_hat_probs[:, 1], average='weighted'))
            else:
                y_score = label_binarize(y_test, classes=[0, 1, 2])
                for i in range(3):
                    roc_auc = roc_auc_score(y_score[:, i], y_hat_probs[:, i], average='weighted')
                    model_df[f'AUC {i}'].append(roc_auc)
            model_df['Accuracy'].append(accuracy_score(y_test, y_hat))
            model_df['F1 Score'].append(f1_score(y_test, y_hat, average='weighted'))
            model_df['Models'].append(str(clf))
        pd.DataFrame(model_df).to_csv(os.path.join('results', str(experiment_name), f'{experiment_name}_evaluating.csv'))


def train_fresh_models(datasets, experiment_name, binary=True):
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
        groups = []
        for i in range(0, len(datasets[0])):
            group = np.ndarray(len(datasets[0][i]))
            group[:] = i
            groups.extend(group)
        x = np.vstack(datasets[0])
        y = np.vstack(datasets[1]).ravel()

        y_0, y_1 = len(y[y == 0]), len(y[y == 1])
        print(f"{experiment_name} Dataset | Class 0 Exemplars: {y_0} | Class 1 Exemplars: {y_1}")

        rng = np.random.RandomState(0)
        classifiers = [LinearDiscriminantAnalysis(),
                       KNeighborsClassifier(9),
                       DecisionTreeClassifier(min_samples_split=60, random_state=rng),
                       RandomForestClassifier(min_samples_split=60, n_estimators=100, random_state=rng),
                       AdaBoostClassifier(n_estimators=50, base_estimator=DecisionTreeClassifier(min_samples_split=60), random_state=rng),
                       # SVC(kernel="linear", C=0.025, probability=True),
                       # SVC(gamma=2, C=1, probability=True),
                       # GaussianProcessClassifier(1.0 * RBF(1.0)),
                       # MLPClassifier(alpha=1, max_iter=1000),
                       # GaussianNB(),
                       # QuadraticDiscriminantAnalysis()
                       ]

        best_models = []
        for clf in classifiers:
            fold = 0
            k_fold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=rng)
            if binary:
                k_fold_df = dict((sub, []) for sub in ['F1 Score', 'Accuracy', 'AUC', 'Fitted Models'])
            else:
                k_fold_df = dict((sub, []) for sub in ['F1 Score', 'Accuracy', 'AUC 0', 'AUC 1', 'AUC 2', 'Fitted Models'])
            print(f"Testing {str(clf)} on {experiment_name}")
            pipeline = make_pipeline(StandardScaler(), clf)
            for train_ix, test_ix in k_fold.split(x, y, groups=groups):
                print(f"\tRunning fold {fold} of 10")
                train_x, x_test = x[train_ix], x[test_ix]
                train_y, y_test = y[train_ix], y[test_ix]
                model = pipeline.fit(train_x, train_y)
                y_hat_probs = model.predict_proba(x_test)
                y_hat = np.argmax(y_hat_probs, axis=1)
                if binary:
                    k_fold_df['AUC'].append(roc_auc_score(y_test, y_hat_probs[:, 1], average='weighted'))
                else:
                    y_score = label_binarize(y_test, classes=[0, 1, 2])
                    for i in range(3):
                        roc_auc = roc_auc_score(y_score[:, i], y_hat_probs[:, i], average='weighted')
                        k_fold_df[f'AUC {i}'].append(roc_auc)
                k_fold_df['Accuracy'].append(accuracy_score(y_test, y_hat))
                k_fold_df['F1 Score'].append(f1_score(y_test, y_hat, average='weighted'))
                k_fold_df['Fitted Models'].append(model)
                fold += 1
            if binary:
                best_models.append(k_fold_df['Fitted Models'][np.argmax(k_fold_df['AUC'])])
            else:
                best_models.append(k_fold_df['Fitted Models'][np.argmax(np.mean([i, j, k]) for i, j, k in zip(k_fold_df['AUC 0'], k_fold_df['AUC 1'], k_fold_df['AUC 2']))])
            pd.DataFrame(k_fold_df).to_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_training.csv'))
        with open(os.path.join('results', str(experiment_name), f'{experiment_name}_models.pkl'), 'wb') as f:
            pickle.dump(best_models, f)
        return best_models
    with open(os.path.join('results', str(experiment_name), f'{experiment_name}_models.pkl'), 'rb') as f:
        best_models = pickle.load(f)
    return best_models


# Generate datasets
window_size = 5
wesad_multi, wesad_binary = wesad.windowed_feature_extraction(window_size)

clasir_multi, clasir_binary = clasir.windowed_feature_extraction(window_size)

clasir_noacc_multi, clasir_noacc_binary = clasir.windowed_feature_extraction(window_size, exclude_acc=True, dataset_name='clasir_no_acc')

wesad_noacc_multi, wesad_noacc_binary = wesad.windowed_feature_extraction(window_size, exclude_acc=True, dataset_name='wesad_no_acc')

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
wesad_binary_alone = train_fresh_models(wesad_binary, 'WESAD Binary Task')
wesad_multi_alone = train_fresh_models(wesad_multi, 'WESAD Multiclass Task', binary=False)

clasir_binary_alone = train_fresh_models(clasir_binary, 'cLASIr Binary Task')
clasir_multi_alone = train_fresh_models(clasir_multi, 'cLASIr Multiclass Task', binary=False)

# Perform classic benchmarking with SciKit Learn built in models on no accelerometer datasets
wesad_noacc_binary_alone = train_fresh_models(wesad_noacc_binary, 'WESAD Binary Task, No Accelerometer')
wesad_noacc_multi_alone = train_fresh_models(wesad_noacc_multi, 'WESAD Multiclass Task, No Accelerometer', binary=False)

clasir_noacc_binary_alone = train_fresh_models(clasir_noacc_binary, 'cLASIr Binary Task, No Accelerometer')
clasir_noacc_multi_alone = train_fresh_models(clasir_noacc_multi, 'cLASIr Multiclass Task, No Accelerometer', binary=False)

case_binary_alone = train_fresh_models(case_binary, 'CASE Binary Task')
case_multi_alone = train_fresh_models(case_multi, 'CASE Multiclass Task', binary=False)

# Observe domain shift by using best model for each dataset on the other datasets
evaluate_pretrained_models(wesad_multi_alone, clasir_multi, "WESAD on cLASIr, Multi", binary=False)
evaluate_pretrained_models(wesad_binary_alone, clasir_binary, "WESAD on cLASIr, Binary")

evaluate_pretrained_models(clasir_multi_alone, wesad_multi, "cLASIr on WESAD, Multi", binary=False)
evaluate_pretrained_models(clasir_binary_alone, wesad_binary, "cLASIr on WESAD, Binary")

evaluate_pretrained_models(wesad_noacc_binary_alone, case_clasir_binary, "WESAD on Others, Binary")
evaluate_pretrained_models(wesad_noacc_multi_alone, case_clasir_multi, "WESAD on Others, Multi", binary=False)

evaluate_pretrained_models(clasir_noacc_binary_alone, case_wesad_binary, "cLASIr on Others, Binary")
evaluate_pretrained_models(clasir_noacc_multi_alone, case_wesad_multi, "cLASIr on Others, Multi", binary=False)

evaluate_pretrained_models(case_binary_alone, clasir_wesad_binary, "CASE on Others, Binary")
evaluate_pretrained_models(case_multi_alone, clasir_wesad_multi, "CASE on Others, Multi", binary=False)

# Perform domain adaptation using dataset mixing, benchmark in the same way
train_fresh_models(case_clasir_binary, "WESAD Transfer, Binary")
train_fresh_models(case_clasir_multi, "WESAD Transfer, Multi", binary=False)

train_fresh_models(case_wesad_binary, "cLASIr Transfer, Binary")
train_fresh_models(case_wesad_multi, "cLASIr Transfer, Multi", binary=False)

train_fresh_models(clasir_wesad_binary, "CASE Transfer, Binary")
train_fresh_models(clasir_wesad_multi, "CASE Transfer, Multi", binary=False)

# Perform domain adaptation using a trainable transform layer, benchmark in the same way
# https://stackoverflow.com/a/47520976
