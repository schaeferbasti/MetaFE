import os
import tempfile

from autogluon.tabular import TabularPredictor
from utils.tabrepo_2024_custom import zeroshot2024


def run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=False):
    label = "target"
    train_data = X_train
    train_data[label] = y_train
    test_data = X_test
    test_data[label] = y_test

    allowed_models = ["GBM"]  # , "RF", "KNN", "XT", "CAT", "XGB", "LR", "FASTAI", "AG_AUTOMM", "NN_TORCH"]

    zeroshot2024 = get_zeroshot_models(allowed_models, zeroshot)
    # -- Run AutoGluon
    predictor = init_and_fit_predictor(label, train_data, zeroshot2024)
    lb = predictor.leaderboard(test_data)
    return lb

def run_autogluon_lgbm_classification(X_train, y_train, X_test, y_test, zeroshot=False):
    label = "target"
    train_data = X_train
    train_data[label] = y_train
    test_data = X_test
    test_data[label] = y_test

    allowed_models = ["GBM"]  # , "RF", "KNN", "XT", "CAT", "XGB", "LR", "FASTAI", "AG_AUTOMM", "NN_TORCH"]

    zeroshot2024 = get_zeroshot_models(allowed_models, zeroshot)
    # -- Run AutoGluon
    predictor = init_and_fit_improvement_predictor_classification(label, train_data, zeroshot2024)
    lb = predictor.leaderboard(test_data)
    return lb


def run_autogluon_lgbm_regression(X_train, y_train, X_test, y_test, zeroshot=False):
    label = "target"
    train_data = X_train
    train_data[label] = y_train
    test_data = X_test
    test_data[label] = y_test

    allowed_models = ["GBM"]  # , "RF", "KNN", "XT", "CAT", "XGB", "LR", "FASTAI", "AG_AUTOMM", "NN_TORCH"]

    zeroshot2024 = get_zeroshot_models(allowed_models, zeroshot)
    # -- Run AutoGluon
    predictor = init_and_fit_improvement_predictor_regression(label, train_data, zeroshot2024)
    lb = predictor.leaderboard(test_data)
    return lb


def init_and_fit_predictor(label, train_data, zeroshot2024):
    predictor = TabularPredictor(
        label=label,
        eval_metric="log_loss",  # roc_auc (binary), log_loss (multiclass) root_mean_squared_error (regression)
        problem_type="multiclass",  # binary, multiclass, regression
        verbosity=0,
    )
    predictor.fit(
        time_limit=int(60 * 60* 10),
        memory_limit=48,
        num_cpus=8,
        num_gpus=0,
        train_data=train_data,
        presets="high_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
        fit_weighted_ensemble=False
    )
    return predictor


def init_and_fit_improvement_predictor_classification(label, train_data, zeroshot2024):
    predictor = TabularPredictor(
        label=label,
        eval_metric="log_loss",  # roc_auc (binary), log_loss (multiclass)
        problem_type="multiclass",  # binary, multiclass
        verbosity=0,
        path=tempfile.mkdtemp() + os.sep,
    )
    predictor.fit(
        time_limit=int(60 * 60 * 4),
        memory_limit=48,
        num_cpus=8,
        num_gpus=0,
        train_data=train_data,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
        fit_weighted_ensemble=False
    )
    return predictor


def init_and_fit_improvement_predictor_regression(label, train_data, zeroshot2024):
    predictor = TabularPredictor(
        label=label,
        eval_metric="root_mean_squared_error",  # roc_auc (binary), log_loss (multiclass)
        problem_type="regression",  # binary, multiclass
        verbosity=0,
        path=tempfile.mkdtemp() + os.sep,
    )
    predictor.fit(
        time_limit=int(60 * 60 * 4),
        memory_limit=48,
        num_cpus=8,
        num_gpus=0,
        train_data=train_data,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
        fit_weighted_ensemble=False
    )
    return predictor


def get_zeroshot_models(allowed_models, zeroshot):
    for k in list(zeroshot2024.keys()):
        if k not in allowed_models:
            del zeroshot2024[k]
        else:
            if not zeroshot:
                zeroshot2024[k] = zeroshot2024[k][:1]
            else:
                zeroshot2024[k] = zeroshot2024[k][1:]
    return zeroshot2024


