import os

import numpy as np
import pandas as pd
import shap
import sklearn
from matplotlib.pyplot import gcf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import clasir_interfacing as clasir

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


def pose_active_passive(dataset, labels):
    # 0 = Control | 1 = Precondition | 2 = HAI
    trimmed_dataset, trimmed_labels = [], []
    for x, y in zip(dataset, labels):
        del_indxs = np.where(y == 1)[0]
        trimmed_labels.append(np.delete(y, del_indxs, 0))
        trimmed_dataset.append(np.delete(x, del_indxs, 0))
        trimmed_labels[-1][trimmed_labels[-1] == 2] = 1
    return trimmed_dataset, trimmed_labels


def run_shapley_tests(dataset, features, experiment_name):
    if not os.path.exists(os.path.join('results', 'FI')):
        os.mkdir(os.path.join('results', 'FI'))
    rng = np.random.RandomState(0)
    groups = []
    for i in range(0, len(clasir_multi[0])):
        group = np.ndarray(len(dataset[0][i]))
        group[:] = i
        groups.extend(group)
    x = np.vstack(dataset[0])
    y = np.vstack(dataset[1]).ravel()

    model_coefs = []
    model_fi = None
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('clf', LinearDiscriminantAnalysis())])
    k_fold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=rng)
    for train_ix, test_ix in k_fold.split(x, y, groups):
        train_x, x_test = x[train_ix], x[test_ix]
        train_y, y_test = y[train_ix], y[test_ix]

        base_model = pipeline.fit(train_x, train_y)
        model_coefs.append(base_model.named_steps['clf'].coef_[0])
        explainer = shap.Explainer(base_model.predict, x_test, feature_names=features)
        shap_values = explainer(x_test)
        shap.summary_plot(shap_values, plot_type='violin', show=False, feature_names=features)
        fig = gcf()
        fig.savefig(os.path.join('results', 'FI', f'{experiment_name}_{len(model_coefs)}.png'))
        fig.clear()
    pd.DataFrame(model_coefs, columns=features).to_excel(os.path.join('results', 'FI', f'{experiment_name}.xlsx'))


if __name__ == '__main__':
    clasir_multi, _ = clasir.windowed_feature_extraction(5)
    clasir_int = pose_active_passive(clasir_multi[0], clasir_multi[1])

    clasir_noacc_multi, _ = clasir.windowed_feature_extraction(5, exclude_acc=True,
                                                               dataset_name='clasir_no_acc')
    clasir_noacc_int = pose_active_passive(clasir_noacc_multi[0], clasir_noacc_multi[1])

    run_shapley_tests(clasir_int, feature_names, 'acc_fi')
    run_shapley_tests(clasir_noacc_int, feature_names_no_acc, 'no_acc_fi')
