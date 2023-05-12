import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as crop_mask
import os, wget
from collections import Counter
import numpy as np

# import verde as vd
from shapely.geometry import box
from imblearn.under_sampling import RandomUnderSampler
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt

from shapely import affinity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform

# from xgboost import XGBClassifier
import importlib
from utils import Dataset
from utils import get_cluster_pixels, get_selection

from utils import scale_normalize, get_models, get_metric
from utils import decodeClasses, decodeClassesLevel1

importlib.reload(utils)


def get_dataset(
    gdf: gpd.GeoDataFrame, non_forest: gpd.GeoDataFrame, threshold: float = 80
):
    rename = {
        "SOS_PRC": "С",
        "OS_PRC": "ОС",
        "BER_PRC": "Б",
        "PICH_PRC": "П",
        "EL_PRC": "Е",
        "KEDR_PRC": "К",
        "LSTV_PRC": "Л",
    }

    code_class = {"С": 7, "ОС": 5, "Б": 1, "П": 6, "Е": 2, "К": 3, "Л": 4}

    target_cols = [
        "EL_PRC",
        "KEDR_PRC",
        "LSTV_PRC",
        "PICH_PRC",
        "SOS_PRC",
        "BER_PRC",
        "OS_PRC",
    ]

    mask = gdf[target_cols] > threshold
    select = gdf.loc[mask.any(axis=1)].copy()
    t = select.loc[:, target_cols].idxmax(axis=1)
    select.loc[:, "t"] = select.loc[:, target_cols].idxmax(axis=1)
    select.loc[:, "t_Клас"] = select["t"].apply(lambda x: rename[x])
    select.loc[:, "t_Class"] = select["t_Клас"].apply(lambda x: code_class[x])
    select.pop("t")
    non_forest[select.columns[:-3]] = 1
    select = pd.concat([select, non_forest[select.columns]])
    return select


def get_models(class_weights: dict) -> list:
    n_jobs = 8
    return [
        KNeighborsClassifier(
            n_jobs=n_jobs,
            algorithm="ball_tree",
            leaf_size=100,
            n_neighbors=10,
            weights="uniform",
        ),
        DecisionTreeClassifier(
            random_state=42,
            criterion="entropy",
            max_depth=9,
            max_features=None,
            min_samples_leaf=3,
            min_samples_split=2,
            splitter="best",
            class_weight=class_weights,
        ),
        RandomForestClassifier(
            n_jobs=n_jobs,
            random_state=42,
            criterion="gini",
            max_features="auto",
            class_weight=class_weights,
            max_depth=50,
            n_estimators=500,
            min_samples_leaf=2,
            min_samples_split=6,
        ),
        ExtraTreesClassifier(
            n_jobs=n_jobs,
            random_state=42,
            class_weight=class_weights,
            criterion="entropy",
            max_depth=9,
            max_features="log2",
            min_samples_leaf=5,
            min_samples_split=2,
            n_estimators=150,
        ),
        #         RidgeClassifier(
        #             random_state=42,
        #             solver="sag",
        #             class_weight=class_weights,
        #             fit_intercept=True,
        #             alpha=1.1,
        #             tol=1e-5,
        #         ),
        LogisticRegression(
            n_jobs=n_jobs,
            random_state=42,
            class_weight=class_weights,
            dual=False,
            fit_intercept=False,
            C=1.2,
            max_iter=100,
            tol=1e-04,
            penalty="l1",
            solver="saga",
        ),
        SVC(
            random_state=42,
            gamma="scale",
            class_weight=class_weights,
            kernel="poly",
            C=1,
            degree=1,
            tol=1e-5,
            probability=True,
        ),
        GradientBoostingClassifier(
            **{
                "n_estimators": 75,
                "min_samples_split": 47,
                "max_leaf_nodes": 52,
                "learning_rate": 0.1202,
            }
        )
        #                    XGBClassifier(n_jobs=-1, tree_method='gpu_hist', predictor='gpu_predictor', booster='gblinear', eta=0.3, gamma='auto', max_depth=20)
    ]


def get_metric(base_classfiers: list, y_test: np.ndarray, X_test: np.ndarray):
    score_classfiers_accuracy_score = []
    score_classfiers_roc_auc_score = []
    score_classfiers_f1_score = []
    df_score_class_dict = {}
    df_score_class_list = []

    name_classifiers = [
        "KNeighborsClassifier",
        "DecisionTreeClassifier",
        "RandomForest",
        "ExtraTreesClassifier",
        #         "RidgeClassifier",
        "LogisticRegression",
        "SVC",
        "GradientBoostingClassifier",
    ]
    for i in range(len(base_classfiers)):
        y_predict = base_classfiers[i].predict(X_test)
        score_classfiers_accuracy_score.append(accuracy_score(y_test, y_predict))
        if name_classifiers[i] != "RidgeClassifier":
            score_classfiers_roc_auc_score.append(
                roc_auc_score(
                    y_test, base_classfiers[i].predict_proba(X_test), multi_class="ovr"
                )
            )
        else:
            ridge_predict = []
            for k in range(len(X_test)):
                d = base_classfiers[i].decision_function(X_test)[k]
                probs = np.exp(d) / np.sum(np.exp(d))
                ridge_predict.append(probs)
            ridge_predict = np.array(ridge_predict)
            score_classfiers_roc_auc_score.append(
                roc_auc_score(y_test, ridge_predict, multi_class="ovr")
            )

        score_classfiers_f1_score.append(
            f1_score(y_test, y_predict, average="weighted")
        )
        df_i = pd.DataFrame(
            metrics.classification_report(y_test, y_predict, digits=2, output_dict=True)
        ).transpose()
        arrays_col = [
            [
                name_classifiers[i],
                name_classifiers[i],
                name_classifiers[i],
                name_classifiers[i],
            ],
            list(df_i.columns),
        ]
        df_i.columns = pd.MultiIndex.from_tuples(list(zip(*arrays_col)))
        df_score_class_list.append(df_i)
        df_score_class_dict[name_classifiers[i]] = df_i

    df_score_class = df_score_class_list[0]
    for i in range(1, len(df_score_class_list)):
        df_score_class = df_score_class.join(df_score_class_list[i])
    df_score_class_index = list(df_score_class.index)

    df_score_group = pd.DataFrame(
        columns=["Model", "Accuracy score", "ROC AUC score", "f1 score"]
    )
    df_score_group["Model"] = name_classifiers
    df_score_group["Accuracy score"] = score_classfiers_accuracy_score
    df_score_group["ROC AUC score"] = score_classfiers_roc_auc_score
    df_score_group["f1 score"] = score_classfiers_f1_score

    return df_score_group, df_score_class_dict


def main():
    # dataset with ~280 points with threshold > 80 %
    # small_dataset = "../shape_data/dataset_plots.geojson"
    # full dataset with ~800
    full_dataset = "../shape_data/forest.geojson"
    gdf = gpd.read_file(full_dataset)
    non_forest = gpd.read_file("../shape_data/non_forest.geojson")

    # threshold=80 -> Percent of main forest type for inventory plot
    gdf = get_dataset(gdf=gdf, non_forest=non_forest, threshold=50)
    gdf = gdf.reset_index(drop=True)
    gdf.loc[:, "key"] = gdf.index

    # mask = gdf['t_Class'] != 6
    # gdf = gdf.loc[mask]

    mask = gdf["t_Class"] != 2
    gdf = gdf.loc[mask]

    mask = gdf["t_Class"] != 4
    gdf = gdf.loc[mask]

    mask = gdf["t_Class"] != 3
    gdf = gdf.loc[mask]

    dataset = Dataset()
    dataset.download_dataset()

    gdf = dataset.procces_gdf(gdf)
    bands, terrain = dataset.get_dataset(gdf=gdf, scale=3.0)

    fci = pd.DataFrame(np.sqrt(bands["B04"] * bands["B08"]), columns=["FCI"])

    bands = pd.concat([bands.iloc[:, :-2], fci, bands.iloc[:, -2:]], axis=1)

    correlation_threshold = 0.7
    df = pd.concat([bands.iloc[:, :-2], terrain], axis=1)

    clustered_df = pd.DataFrame()

    for item in df.key.unique():
        attmpt = get_cluster_pixels(
            df, key=item, correlation_threshold=correlation_threshold
        )
        attmpt = get_selection(attmpt)
        clustered_df = pd.concat([clustered_df, attmpt])
        print(".", end="")

    print("All bands: ", len(clustered_df.columns))
    corr_matrix = bands.iloc[:, :-2].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.97)]
    # Drop features
    input_df = clustered_df.drop(to_drop, axis=1)
    print("Non correlated bands: ", len(input_df.columns))
    df = input_df.copy()
    df[dataset.terrain_cols] = df[dataset.terrain_cols].astype(float)
    df.iloc[:, :-3] = StandardScaler().fit_transform(df.iloc[:, :-3])

    # Resampling - to balance classes
    # df = resample_forest(df)
    only_forest = True
    if only_forest:
        forest_gdf = gdf.loc[gdf["t_Class"] < 8]
    else:
        forest_gdf = gdf.loc[gdf["t_Class"] < 30]

    X_train, X_test, y_train, y_test = train_test_split(
        forest_gdf, forest_gdf["class_name"], test_size=0.3
    )

    train = df.loc[df["key"].isin(X_train["key"])]
    test = df.loc[df["key"].isin(X_test["key"])]
    X_train = train.drop(columns=["key", "class"]).astype("float")
    X_test = test.drop(columns=["key", "class"]).astype("float")
    y_train = train["class"].astype(int)
    y_test = test["class"].astype(int)

    # class weights
    class_weights_vals = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = {x: y for x, y in zip(np.unique(y_train), class_weights_vals)}

    # models
    n_jobs = 8
    name_classfiers = [
        "KNeighborsClassifier",
        "DecisionTreeClassifier",
        "RandomForest",
        "ExtraTreesClassifier",
        #                    'RidgeClassifier',
        "LogisticRegression",
        "SVC",
        "GradientBoostingClassifier",
    ]
    base_classfiers = get_models(class_weights=class_weights)
    for i in range(len(base_classfiers)):
        base_classfiers[i].fit(X_train, y_train)
        print("Done: " + name_classfiers[i])

    df_score_group, df_score_class_dict = get_metric(
        base_classfiers=base_classfiers, y_test=y_test, X_test=X_test
    )
