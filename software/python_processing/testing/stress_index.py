import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, make_scorer, \
    accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import case_interfacing as case
import clacir_interfacing as clasir
import wesad_interfacing as wesad


def calculate_auprc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)


def regression_tests(datasets, experiment_name):
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        os.mkdir(os.path.join('results', str(experiment_name)))
    groups = []
    for i in range(0, len(datasets[0])):
        group = np.ndarray(len(datasets[0][i]))
        group[:] = i
        groups.extend(group)
    x = np.vstack(datasets[0])
    y = np.vstack(datasets[1]).ravel()

    scoring = {
        "Accuracy": make_scorer(accuracy_score, needs_proba=False),
        "F1 Score": make_scorer(f1_score, needs_proba=False),
        "ROC AUC": make_scorer(roc_auc_score, needs_proba=True),
        "AUPRC": make_scorer(calculate_auprc, needs_proba=True)
    }
    for solver, penalty in zip(['newton-cg', 'lbfgs', 'sag', 'saga'],
                               [['l2', 'none'], ['l2', 'none'], ['l2', 'none'],
                                ['elasticnet', 'l1', 'l2', 'none']]):
        if not os.path.exists(os.path.join('results', str(experiment_name),
                                           f"{str(experiment_name)}_{solver}.xlsx")):
            print(f"Running {experiment_name} with {solver} and {penalty}...")
            k_fold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=1).split(x, y, groups)
            grid_values = {'logisticregression__penalty': penalty,
                           'logisticregression__C': [0.01, 1, 10, 100],
                           'logisticregression__max_iter': [250]}
            pipe = make_pipeline(StandardScaler(), LogisticRegression(solver=solver, warm_start=True))
            grid_search = GridSearchCV(pipe, param_grid=grid_values, cv=k_fold, n_jobs=-1,
                                       error_score=np.nan, scoring=scoring, refit='AUPRC')
            clf = grid_search.fit(x, y)
            pd.DataFrame(grid_search.cv_results_).to_excel(
                os.path.join('results', str(experiment_name),
                             f"{str(experiment_name)}_{solver}.xlsx"))


no_temp_ds = [True, True, True, False]
no_temp_acc_ds = [True, True, False, False]
avp = True
if __name__ == '__main__':
    window_size = 5
    if not avp:
        _, clasir_noacc_binary, _ = clasir.windowed_feature_extraction(window_size,
                                                                       datastreams=no_temp_acc_ds)
        _, wesad_noacc_binary = wesad.e4_windowed_feature_extraction(window_size,
                                                                     datastreams=no_temp_acc_ds)
        _, case_binary = case.windowed_feature_extraction(window_size, datastreams=no_temp_ds)

        full_dataset = [
            clasir_noacc_binary[0] + wesad_noacc_binary[0] + case_binary[0],
            clasir_noacc_binary[1] + wesad_noacc_binary[1] + case_binary[1],
        ]

        regression_tests(full_dataset, "Full Dataset Regression Binary, No Accelerometer")
    else:
        _, clasir_binary, clasir_avp = clasir.windowed_feature_extraction(window_size, datastreams=no_temp_ds)
        _, clasir_noacc_binary, clasir_noacc_avp = clasir.windowed_feature_extraction(window_size,
                                                                                      datastreams=no_temp_acc_ds)
        regression_tests(clasir_noacc_binary, "cLASIr Regression Binary, No Accelerometer")
        regression_tests(clasir_noacc_avp, "cLASIr Regression AvP, No Accelerometer")

        regression_tests(clasir_binary, "cLASIr Regression Binary")
        regression_tests(clasir_avp, "cLASIr Regression AvP")
