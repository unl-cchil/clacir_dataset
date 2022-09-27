from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import case_interfacing as case
import clasir_interfacing as clasir
import wesad_interfacing as wesad


def domain_transfer_train():
    pass


def train_pretrained_models(clf, datasets):
    score_metrics = (
        'f1', 'f1_micro', 'f1_macro', 'roc_auc'
    )
    scores = (cross_validate(clf, datasets[0][0] + datasets[0][1], datasets[1][0] + datasets[1][1],
                             cv=10, return_estimator=True, scoring=score_metrics))
    # TODO: Get scores on eval set using best model
    return scores


def evaluate_pretrained_models(clf, datasets):
    y_pred = (cross_val_predict(clf, datasets[0][0] + datasets[0][1], datasets[1][0] + datasets[1][1],
                                cv=10))
    return scores


def train_fresh_models(datasets):
    classifiers = [LinearDiscriminantAnalysis(),
                   KNeighborsClassifier(9),
                   DecisionTreeClassifier(min_samples_split=20),
                   RandomForestClassifier(min_samples_split=20, n_estimators=100),
                   AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(min_samples_split=20))]
    score_metrics = (
        'f1', 'f1_micro', 'f1_macro', 'roc_auc'
    )
    scores = []
    for clf in classifiers:
        # clf.fit(datasets[0], labels[0])
        # score.append(clf.score(datasets[2], labels[2]))
        scores.append(cross_validate(clf, datasets[0][0] + datasets[0][1], datasets[1][0] + datasets[1][1],
                                     cv=10, return_estimator=True, scoring=score_metrics))
        # TODO: Get scores on eval set using best model
    return scores


window_size = 5
wesad_multi, wesad_binary = wesad.windowed_feature_extraction(window_size, exclude_acc=True,
                                                              dataset_name='wesad_no_acc')
clasir_multi, clasir_binary = clasir.windowed_feature_extraction(window_size, exclude_acc=True,
                                                                 dataset_name='clasir_no_acc')
case_multi, case_binary = case.windowed_feature_extraction(window_size)

# Perform classic benchmarking with SciKit Learn built in models on each dataset
clasir_binary_alone = train_fresh_models(clasir_binary)
clasir_multi_alone = train_fresh_models(clasir_multi)

wesad_binary_alone = train_fresh_models(wesad_binary)
wesad_multi_alone = train_fresh_models(wesad_multi)

case_binary_alone = train_fresh_models(case_binary)
case_multi_alone = train_fresh_models(case_multi)

# Observe domain shift by using best model for each dataset on the other datasets

# Perform domain adaptation using dataset mixing, benchmark in the same way

# Perform domain adaptation using a trainable transform layer, benchmark in the same way
