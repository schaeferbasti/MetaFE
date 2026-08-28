import os

import numpy as np
import pandas as pd
import openml

def get_openml_dataset_split_and_metadata(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    dataset_metadata = {"task_id": task.task_id, "task_type": task.task_type, "number_of_classes": 'N/A'}
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    return X_train, y_train, X_test, y_test, dataset_metadata


def concat_data(X_train, y_train, X_test, y_test, target_label):
    y_train = y_train.to_frame(target_label)
    train_data = pd.concat([X_train, y_train], axis=1)
    y_test = y_test.to_frame(target_label)
    test_data = pd.concat([X_test, y_test], axis=1)
    data = pd.concat([train_data, test_data], axis=0)
    return data


def get_name_and_split_and_save_dataset(openml_task_id):
    name = str(openml_task_id)
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")
    root_dir = "MetaFE/Metadata/d2v/dataset2vec/datasets/" + name + "/"
    try:
        os.makedirs(root_dir)
    except FileExistsError:
        pass
    len = X.shape[0]
    folds, validation_folds = get_folds_and_validation_folds(len)
    X.to_csv(root_dir + name + '_py.dat', header=False, index=False)
    folds.to_csv(root_dir + "/" + 'folds_py.dat',  header=False, index=False)
    y.to_csv(root_dir + "/" + 'labels_py.dat', header=False, index=False)
    validation_folds.to_csv(root_dir + "/" + 'validation_folds_py.dat',  header=False, index=False)
    return name, 0


def get_folds_and_validation_folds(len):
    folds = pd.DataFrame()
    for i in range(len):
        row = pd.DataFrame([np.random.choice([1, 0], size=4)])
        folds = pd.concat([folds, row], ignore_index=True)
    validation_folds = pd.DataFrame()
    for i in range(len):
        row = pd.DataFrame([np.random.choice([1, 0], size=4)])
        validation_folds = pd.concat([validation_folds, row], ignore_index=True)
    return folds, validation_folds
