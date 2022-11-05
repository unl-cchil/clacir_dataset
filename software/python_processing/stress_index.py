# Possibly use Shapley values for model explainability
# https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137
# https://datascience.stackexchange.com/a/107710
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, r2_score, roc_auc_score, auc, precision_recall_curve, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import clasir_interfacing as clasir


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
    k_fold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=1).split(x, y, groups)
    scoring = {
        "Cohen's Kappa": make_scorer(cohen_kappa_score, needs_proba=True),
        "R2": make_scorer(r2_score, needs_proba=True),
        "ROC AUC": make_scorer(roc_auc_score, needs_proba=True),
        "AUPRC": make_scorer(calculate_auprc, needs_proba=True)
    }
    for solver, penalty in zip(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                               [['l2', 'none'], ['l2', 'none'], ['l1', 'l2'], ['l2', 'none'],
                                ['elasticnet', 'l1', 'l2', 'none']]):
        grid_values = {'logisticregression__penalty': penalty, 'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        pipe = make_pipeline(StandardScaler(), LogisticRegression(solver=solver))
        grid_search = GridSearchCV(pipe, param_grid=grid_values, cv=k_fold, n_jobs=-1,
                                   error_score=np.nan, scoring=scoring, refit='AUPRC')
        clf = grid_search.fit(x, y)
        pd.DataFrame(grid_search.cv_results_).to_excel(
            os.path.join('results', str(experiment_name),
                         f"{str(experiment_name)}_{solver}.xlsx"))
        with open(os.path.join('results', str(experiment_name), f"{str(experiment_name)}_best.pkl"), 'wb') as f:
            pickle.dump(clf, f)


if __name__ == '__main__':
    window_size = 5
    clasir_noacc_multi, clasir_noacc_binary, clasir_noacc_avp = clasir.windowed_feature_extraction(window_size, exclude_acc=True,
                                                                                 dataset_name='clasir_no_acc')
    regression_tests(clasir_noacc_binary, "cLASIr Regression Binary")

    clasir_noacc_multi = (clasir_noacc_multi[0], clasir_noacc_multi[1] * 0.5)
    regression_tests(clasir_noacc_multi, "cLASIr Regression Multi")
