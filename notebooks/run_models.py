import os
import traceback
from joblib import dump
import pandas as pd
import numpy as np


# test/train split and hyperparameters optimisation
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupShuffleSplit

# ML

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from imblearn.over_sampling import SMOTE


dict_normal_names = {
    7: "Pine",
    5: "Aspen",
    1: "Birch",
    6: "Silver fir",
    15: "Burnt forest",
    13: "Deforestation",
    14: "Grass",
    12: "Soil",
    16: "Swamp",
    11: "Water body",
    17: "Settlements",
}

colors = [
    "#117733",
    "#50CE57",
    "#23A28F",
    "#5BD0AE",
    "#88CCEE",
    "#92462D",
    "#DE7486",
    "#DDCC77",
    "#AA4499",
    "#0f62fe",
    "#be95ff",
]


def get_predictions(
    data,
    model,
    param_grid,
    target_column: str = "class",
    stratify_column: str = "key",
    to_remove_columns: list = ["key"],
    test_size: float = 0.3,
    smote_balance: bool = True,
    cv: int = 5,
    n_iter_search: int = 15,
    label_encoder: bool = False,
    verbose: int = 0,
):
    # test/train spliting considering key overlap problems and missed classes
    while True:
        train_inds, test_inds = next(
            GroupShuffleSplit(
                test_size=test_size, n_splits=2  # ,random_state = 40
            ).split(data, groups=data[stratify_column])
        )
        # because we need pixels from same plots to be separated in train and test

        train = data.iloc[train_inds]
        test = data.iloc[test_inds]
        train_classes = train[target_column].nunique()
        test_classes = test[target_column].nunique()
        all_classes = data[target_column].nunique()
        # because we need classes to be represented in train and test
        if train_classes == test_classes == all_classes:
            break
    train = train.drop(columns=to_remove_columns)
    test = test.drop(columns=to_remove_columns)
    # class balansing with smote
    if smote_balance is True:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(
            train.loc[:, train.columns != target_column], train[target_column]
        )  # drops 3 columns: key, class, and forest
        df_smote = pd.DataFrame(
            X, columns=train.loc[:, train.columns != target_column].columns.tolist()
        )  # drops 3 columns: key, class, and forest

        # we set train/test from SMOTE results
        X_train = df_smote
        y_train = y
        X_test = test.loc[:, train.columns != target_column]
        y_test = test[target_column]
        # we set train/test as it is
    else:
        X_train = train.loc[:, train.columns != target_column]
        y_train = train[target_column]
        X_test = test.loc[:, train.columns != target_column]
        y_test = test[target_column]

    gs = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        cv=cv,
        scoring="f1_weighted",
        verbose=verbose,
        n_jobs=1,
    )

    if label_encoder is True:
        print("label_encoder == True")
        le = LabelEncoder()
        gs.fit(X_train, le.fit_transform(y_train))
        y_pred = gs.best_estimator_.predict(X_test.values)
        model_fit = gs.best_estimator_

        results = {
            "model": model_fit,
            "X_train data": X_train,
            "y train data": y_train,
            "X test data": X_test,
            "y test data": y_test,
            "y predicted": le.inverse_transform(y_pred),
        }

    else:
        gs.fit(X_train, y_train)
        y_pred = gs.best_estimator_.predict(X_test.values)
        model_fit = gs.best_estimator_

        results = {
            "model": model_fit,
            "X_train data": X_train,
            "y train data": y_train,
            "X test data": X_test,
            "y test data": y_test,
            "y predicted": y_pred,
        }

    return results


def get_classes_metrics(models_vector):
    # vector with model variations, y predicted and y true from the dataset
    class_metrics_dataframe = pd.DataFrame()
    count = 0  # counter of iteration

    for i in models_vector:

        count += 1  # counting
        pred = i["y predicted"]  # predicted values
        true = i["y test data"]  # corresponding labels from random test set
        names_list = list(np.unique(true))

        temp = pd.DataFrame(
            {
                "iteration": [count] * len(names_list),
                "names": list(map(dict_normal_names.get, names_list)),
                "f1_scores": f1_score(true, pred, average=None).round(2).tolist(),
                "precision_list": precision_score(true, pred, average=None)
                .round(2)
                .tolist(),
                "recall": recall_score(true, pred, average=None).round(2).tolist(),
            }
        )  # dataset for each model

        class_metrics_dataframe = pd.concat(
            [class_metrics_dataframe, temp], ignore_index=True
        )
    return class_metrics_dataframe


# getting dataset with average metrics for each random prediction
def get_metrics_average(
    models_vector,
):  # vector with model variations, y predicted and y true from the dataset
    average_metrics_dataframe = pd.DataFrame()
    count = 0  # counter of iteration

    for i in models_vector:

        count += 1  # counting
        pred = i["y predicted"]  # predicted values
        true = i["y test data"]  # corresponding labels from random test set

        temp = pd.DataFrame(
            {
                "iteration": [count],  # *len(names_list),
                #'names': list(map(dict_normal_names.get, names_list)),
                "f1_scores": f1_score(true, pred, average="macro").round(2).tolist(),
                "precision": precision_score(true, pred, average="weighted")
                .round(2)
                .tolist(),
                "recall": recall_score(true, pred, average="weighted")
                .round(2)
                .tolist(),
            }
        )  # dataset for each model

        average_metrics_dataframe = pd.concat(
            [average_metrics_dataframe, temp], ignore_index=True
        )
    return average_metrics_dataframe


def get_best_model(datavector_models):
    number = (
        get_metrics_average(datavector_models)
        .sort_values(by="f1_scores", ascending=False)
        .head(1)
        .reset_index()["index"]
        .values[0]
    )
    best_model = datavector_models[number]["model"]
    return best_model


def get_scaled_data(path: str, cols_remove=None):
    """Get scaled data

    Returns:
        _type_: _description_
    """
    if cols_remove is None:
        cols_remove = ["key", "class"]
    df = pd.read_csv(path, index_col=0)
    x = df.drop(columns=cols_remove).values
    # minmax scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    target_cols = [col for col in list(df.columns) if col not in cols_remove]
    df.loc[:, target_cols] = x_scaled
    return df, min_max_scaler


# Number of trees in random forest
def get_random_forest():
    n_estimators = np.arange(100, 350, 10)
    max_depth = np.arange(10, 110, 11)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    random_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }
    return {
        "model": RandomForestClassifier(bootstrap=True, n_jobs=1),
        "grid": random_grid,
    }


def get_svm():
    svc_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["rbf", "linear", "poly"],
    }
    return {"model": SVC(probability=True), "grid": svc_grid}


def get_KNN():
    metric = ["euclidean", "manhattan", "chebyshev", "minkowski"]
    n_neighbors = np.arange(4, 15, 2)
    weights = ["uniform", "distance"]
    random_grid_knn = {"n_neighbors": n_neighbors, "weights": weights, "metric": metric}
    return {"model": KNeighborsClassifier(), "grid": random_grid_knn}


def get_XGB():
    params = {
        "max_depth": [3, 6, 10],
        "min_child_weight": [0.5, 1, 2],
        "n_estimators": np.arange(10, 100, 20),
        "colsample_bytree": [0.3, 0.7, 1],
    }
    return {"model": xgb.XGBClassifier(n_jobs=1), "grid": params}


def model_loop(df_forest, model, settings, smote_balance, problems, verbose: int = 0):
    """

    Args:
        df_forest (_type_): _description_
        settings (_type_): _description_
        smote_balance (_type_): _description_
        problems (_type_): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    local_datavector = []
    for i in range(15):
        print(f"---- {model} ---- {i}")
        try:
            trained_model = get_predictions(
                data=df_forest,
                model=settings["model"],
                param_grid=settings["grid"],
                target_column="class",
                to_remove_columns=["key"],
                smote_balance=smote_balance,
                cv=5,
                n_iter_search=30,
                label_encoder=True if model == "XGB" else False,
                verbose=verbose,
            )
            local_datavector.append(trained_model)
            status = "Done"
        except Exception as e:
            print(e)
            traceback.print_exc()
            problem = {}
            # problem["fname"] = dataset
            problem["model"] = model
            # problem["scale"] = scale
            status = "Error"
            problems.append(problem)
            print(status)
            continue
    return local_datavector, problems


def main(scale: int):
    """main"""
    folder = "../shape_data/filtered_datasets_2024/"

    metric_container = pd.DataFrame()
    problems: list = []
    dataset = os.path.join(folder, f"df{scale}_filtered_modified.csv")
    df_scaled, min_max_scaler = get_scaled_data(os.path.join(folder, dataset))
    scaler_path = os.path.join(f"../models/best_models/scaler_dataset_{scale}.joblib")
    dump(min_max_scaler, scaler_path)
    mask_forest = df_scaled["class"] < 10
    df_forest = df_scaled.loc[mask_forest]
    models = {
        "RandomForest": get_random_forest(),
        "SVC": get_svm(),
        "kNN": get_KNN(),
        "XGB": get_XGB(),
    }
    for model, settings in models.items():
        for smote_balance in [True, False]:
            print(model, scale, smote_balance)
            datavector, problems = model_loop(
                df_forest, model, settings, smote_balance, problems, verbose=0
            )
            model_metrics = get_classes_metrics(datavector)
            model_metrics["model"] = model
            model_metrics["smote_balance"] = smote_balance
            model_metrics["scale"] = scale
            model_metrics["fname"] = dataset
            model_metrics["experiment_status"] = dataset
            metric_container = pd.concat([metric_container, model_metrics], axis=0)
            best_model = get_best_model(datavector)
            core = dataset.split(".")[0]
            model_path = os.path.join(f"../models/best_models/{model}_{core}.joblib")
            dump(best_model, model_path)
        metric_container.to_csv(f"../shape_data/metric_results_scale_{scale}.csv")


if __name__ == "__main__":
    import logging
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int)
    args = parser.parse_args()

    logging.info("Start")
    main(scale=args.scale)
    logging.info("Done ✅")
