"""
Don't forget that line 1135 of venv\Lib\site-packages\sklearn\model_selection\_validation.py was changed from n_classes
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import case_interfacing as case
import clasir_interfacing as clasir
import wesad_interfacing as wesad


def domain_transfer_train():
    pass


def train_pretrained_models(clf, datasets, experiment_name, binary=True):
    if not os.path.exists(os.path.join('results', str(experiment_name))):
        # os.mkdir(os.path.join('results', str(experiment_name)))
        pass


def evaluate_pretrained_models(clfs, datasets, experiment_name, report_df, binary=True):
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
            model_df['Model'].append(str(clf))
        pd.DataFrame(model_df).to_csv(os.path.join('results', str(experiment_name), f'{experiment_name}_evaluating.csv'))
    clf_results = {f"Experiment": experiment_name}
    results_df = pd.read_csv(os.path.join('results', str(experiment_name), f'{experiment_name}_evaluating.csv'))
    for _, row in results_df.iterrows():
        clf_name = ''.join(filter(str.isupper, row['Model'].split(' ')[-1].split('(')[0]))
        if binary:
            clf_results.update({
                f"{clf_name} Accuracy": f"{row['Accuracy'] * 100:.2f}",
                f"{clf_name} F1 Score": f"{row['F1 Score'] * 100:.2f}",
                f"{clf_name} AUC": f"{row['AUC'] * 100:.2f}"
            })
        else:
            clf_results.update({
                f"{clf_name} Accuracy": f"{row['Accuracy'] * 100:.2f}",
                f"{clf_name} F1 Score": f"{row['F1 Score'] * 100:.2f}",
                f"{clf_name} AUC 0": f"{row['AUC 0'] * 100:.2f}",
                f"{clf_name} AUC 1": f"{row['AUC 1'] * 100:.2f}",
                f"{clf_name} AUC 2": f"{row['AUC 2'] * 100:.2f}"
            })
    return report_df.append(clf_results, ignore_index=True)


def train_fresh_models(datasets, experiment_name, report_df, binary=True):
    rng = np.random.RandomState(0)
    classifiers = [LinearDiscriminantAnalysis(),
                   KNeighborsClassifier(9),
                   DecisionTreeClassifier(min_samples_split=60, random_state=rng),
                   RandomForestClassifier(min_samples_split=60, n_estimators=100, random_state=rng),
                   # AdaBoostClassifier(n_estimators=50, base_estimator=DecisionTreeClassifier(min_samples_split=60), random_state=rng),
                   # SVC(kernel="linear", C=0.025, probability=True),
                   # SVC(gamma=2, C=1, probability=True),
                   # GaussianProcessClassifier(1.0 * RBF(1.0)),
                   # MLPClassifier(alpha=1, max_iter=1000),
                   # GaussianNB(),
                   # QuadraticDiscriminantAnalysis()
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

        y_0, y_1 = len(y[y == 0]), len(y[y == 1])
        print(f"{experiment_name} Dataset | Class 0 Exemplars: {y_0} | Class 1 Exemplars: {y_1}")

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
        # return best_models
    with open(os.path.join('results', str(experiment_name), f'{experiment_name}_models.pkl'), 'rb') as f:
        best_models = pickle.load(f)
    clf_results = {f"Experiment": experiment_name}
    for clf in classifiers:
        clf_name = ''.join(filter(str.isupper, str(clf).split('(')[0]))
        results_df = pd.read_csv(os.path.join('results', str(experiment_name), f'{str(clf)[0:5]}_training.csv'))
        if binary:
            clf_results.update({
                f"{clf_name} Accuracy": f"{np.mean(results_df['Accuracy'])*100:.2f} \u00B1 {np.std(results_df['Accuracy'])*100:.2f}",
                f"{clf_name} F1 Score": f"{np.mean(results_df['F1 Score'])*100:.2f} \u00B1 {np.std(results_df['F1 Score'])*100:.2f}",
                f"{clf_name} AUC": f"{np.mean(results_df['AUC'])*100:.2f} \u00B1 {np.std(results_df['AUC'])*100:.2f}"
            })
        else:
            clf_results.update({
                f"{clf_name} Accuracy": f"{np.mean(results_df['Accuracy']) * 100:.2f} \u00B1 {np.std(results_df['Accuracy']) * 100:.2f}",
                f"{clf_name} F1 Score": f"{np.mean(results_df['F1 Score']) * 100:.2f} \u00B1 {np.std(results_df['F1 Score']) * 100:.2f}",
                f"{clf_name} AUC 0": f"{np.mean(results_df['AUC 0']) * 100:.2f} \u00B1 {np.std(results_df['AUC 0']) * 100:.2f}",
                f"{clf_name} AUC 1": f"{np.mean(results_df['AUC 1']) * 100:.2f} \u00B1 {np.std(results_df['AUC 1']) * 100:.2f}",
                f"{clf_name} AUC 2": f"{np.mean(results_df['AUC 2']) * 100:.2f} \u00B1 {np.std(results_df['AUC 2']) * 100:.2f}"
            })
    report_df = report_df.append(clf_results, ignore_index=True)
    return report_df, best_models


# Generate datasets
window_size = 5
wesad_multi, wesad_binary = wesad.windowed_feature_extraction(window_size)
wesad_noacc_multi, wesad_noacc_binary = wesad.windowed_feature_extraction(window_size, exclude_acc=True, dataset_name='wesad_no_acc')

clasir_multi, clasir_binary = clasir.windowed_feature_extraction(window_size)
clasir_noacc_multi, clasir_noacc_binary = clasir.windowed_feature_extraction(window_size, exclude_acc=True, dataset_name='clasir_no_acc')

case_multi, case_binary = case.windowed_feature_extraction(window_size)

case_clasir_binary = [clasir_noacc_binary[0] + case_binary[0],
                      clasir_noacc_binary[1] + case_binary[1]]
case_clasir_multi = [clasir_noacc_multi[0] + case_multi[0],
                     clasir_noacc_multi[1] + case_multi[1]]
clasir_wesad_binary = [clasir_noacc_binary[0] + wesad_noacc_binary[0],
                       clasir_noacc_binary[1] + wesad_noacc_binary[1]]
clasir_wesad_multi = [clasir_noacc_multi[0] + wesad_noacc_multi[0],
                      clasir_noacc_multi[1] + wesad_noacc_multi[1]]
case_wesad_binary = [wesad_noacc_binary[0] + case_binary[0],
                     wesad_noacc_binary[1] + case_binary[1]]
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

# Collect data into DataFrame
multi_results_df = pd.DataFrame()
binary_results_df = pd.DataFrame()

multi_eval_df = pd.DataFrame()
binary_eval_df = pd.DataFrame()

# Perform classic benchmarking with SciKit Learn built in models on each dataset
binary_results_df, wesad_binary_alone = train_fresh_models(wesad_binary, 'WESAD Binary Task', binary_results_df)
multi_results_df, wesad_multi_alone = train_fresh_models(wesad_multi, 'WESAD Multiclass Task', multi_results_df, binary=False)

binary_results_df, clasir_binary_alone = train_fresh_models(clasir_binary, 'cLASIr Binary Task', binary_results_df)
multi_results_df, clasir_multi_alone = train_fresh_models(clasir_multi, 'cLASIr Multiclass Task', multi_results_df, binary=False)

# Perform classic benchmarking with SciKit Learn built in models on no accelerometer datasets
binary_results_df, wesad_noacc_binary_alone = train_fresh_models(wesad_noacc_binary, 'WESAD Binary Task, No Accelerometer', binary_results_df)
multi_results_df, wesad_noacc_multi_alone = train_fresh_models(wesad_noacc_multi, 'WESAD Multiclass Task, No Accelerometer', multi_results_df, binary=False)

binary_results_df, clasir_noacc_binary_alone = train_fresh_models(clasir_noacc_binary, 'cLASIr Binary Task, No Accelerometer', binary_results_df)
multi_results_df, clasir_noacc_multi_alone = train_fresh_models(clasir_noacc_multi, 'cLASIr Multiclass Task, No Accelerometer', multi_results_df, binary=False)

binary_results_df, case_binary_alone = train_fresh_models(case_binary, 'CASE Binary Task', binary_results_df)
multi_results_df, case_multi_alone = train_fresh_models(case_multi, 'CASE Multiclass Task', multi_results_df, binary=False)

# Observe domain shift by using best model for each dataset on the other datasets
multi_eval_df = evaluate_pretrained_models(wesad_multi_alone, clasir_multi, "WESAD on cLASIr, Multi", multi_eval_df, binary=False)
binary_eval_df = evaluate_pretrained_models(wesad_binary_alone, clasir_binary, "WESAD on cLASIr, Binary", binary_eval_df)

multi_eval_df = evaluate_pretrained_models(clasir_multi_alone, wesad_multi, "cLASIr on WESAD, Multi", multi_eval_df, binary=False)
binary_eval_df = evaluate_pretrained_models(clasir_binary_alone, wesad_binary, "cLASIr on WESAD, Binary", binary_eval_df)

binary_eval_df = evaluate_pretrained_models(wesad_noacc_binary_alone, case_clasir_binary, "WESAD on Others, Binary", binary_eval_df)
multi_eval_df = evaluate_pretrained_models(wesad_noacc_multi_alone, case_clasir_multi, "WESAD on Others, Multi", multi_eval_df, binary=False)

binary_eval_df = evaluate_pretrained_models(clasir_noacc_binary_alone, case_wesad_binary, "cLASIr on Others, Binary", binary_eval_df)
multi_eval_df = evaluate_pretrained_models(clasir_noacc_multi_alone, case_wesad_multi, "cLASIr on Others, Multi", multi_eval_df, binary=False)

binary_eval_df = evaluate_pretrained_models(case_binary_alone, clasir_wesad_binary, "CASE on Others, Binary", binary_eval_df)
multi_eval_df = evaluate_pretrained_models(case_multi_alone, clasir_wesad_multi, "CASE on Others, Multi", multi_eval_df, binary=False)

# Perform domain adaptation using dataset mixing, benchmark in the same way
binary_results_df, case_clasir_mix_binary = train_fresh_models(case_clasir_binary, "CASE cLASIr Transfer, Binary", binary_results_df)
multi_results_df, case_clasir_mix_multi = train_fresh_models(case_clasir_multi, "CASE cLASIr Transfer, Multi", multi_results_df, binary=False)

binary_results_df, case_wesad_mix_binary = train_fresh_models(case_wesad_binary, "CASE WESAD Transfer, Binary", binary_results_df)
multi_results_df, case_wesad_mix_multi = train_fresh_models(case_wesad_multi, "CASE WESAD Transfer, Multi", multi_results_df, binary=False)

binary_results_df, clasir_wesad_mix_binary = train_fresh_models(clasir_wesad_binary, "cLASIr WESAD Transfer, Binary", binary_results_df)
multi_results_df, clasir_wesad_mix_multi = train_fresh_models(clasir_wesad_multi, "cLASIr WESAD Transfer, Multi", multi_results_df, binary=False)

# Evaluate improved domain shift
binary_eval_df = evaluate_pretrained_models(case_clasir_mix_binary, wesad_noacc_binary, "CASE cLASIr on Others, Binary", binary_eval_df)
multi_eval_df = evaluate_pretrained_models(case_clasir_mix_multi, wesad_noacc_multi, "CASE cLASIr on Others, Multi", multi_eval_df, binary=False)

binary_eval_df = evaluate_pretrained_models(case_wesad_mix_binary, clasir_noacc_binary, "CASE WESAD on Others, Binary", binary_eval_df)
multi_eval_df = evaluate_pretrained_models(case_wesad_mix_multi, clasir_noacc_multi, "CASE WESAD on Others, Multi", multi_eval_df, binary=False)

binary_eval_df = evaluate_pretrained_models(clasir_wesad_mix_binary, case_binary, "cLASIr WESAD on Others, Binary", binary_eval_df)
multi_eval_df = evaluate_pretrained_models(clasir_wesad_mix_multi, case_multi, "cLASIr WESAD on Others, Multi", multi_eval_df, binary=False)

# Evaluate using all data, 34 classification tasks total
binary_results_df, full_mix_binary = train_fresh_models(full_dataset_binary, "Full Dataset, Binary", binary_results_df)
multi_results_df, full_mix_multi = train_fresh_models(full_dataset_multi, "Full Dataset, Multi", multi_results_df, binary=False)

# Prepare and save results
multi_results_df.set_index('Experiment', inplace=True)
multi_results_df.to_excel(os.path.join('results', 'Multiclass Training Results.xlsx'))

binary_results_df.set_index('Experiment', inplace=True)
binary_results_df.to_excel(os.path.join('results', 'Binary Training Results.xlsx'))

multi_eval_df.set_index('Experiment', inplace=True)
multi_eval_df.to_excel(os.path.join('results', 'Multiclass Evaluation Results.xlsx'))

binary_eval_df.set_index('Experiment', inplace=True)
binary_eval_df.to_excel(os.path.join('results', 'Binary Evaluation Results.xlsx'))

# Perform domain adaptation using a trainable transform layer, benchmark in the same way
# https://stackoverflow.com/a/47520976
