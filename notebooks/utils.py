import datetime as dt
import os
from collections import Counter
from typing import Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import seaborn as sns
import wget
from rasterio.mask import mask as crop_mask
from scipy.stats import loguniform
from shapely import affinity
from shapely.geometry import box
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from tqdm.notebook import tqdm
from yellowbrick.cluster import KElbowVisualizer


class Dataset:
    def __init__(self, path_to_tiff_file: str):
        self.dates_images = []
        self.scale = 1.0
        self.terrain_cols = ["aspect", "slope", "wetnessindex", "sink"]
        self.cols = self.get_all_cols()
        self.path_to_tiff_file = path_to_tiff_file
        self.texture_columns = [
            "ASM1",
            "ASM2",
            "contrast1",
            "contrast2",
            "correlation1",
            "correlation2",
            "dissimilarity1",
            "dissimilarity2",
            "energy1",
            "energy2",
            "homogeneity1",
            "homogeneity2",
        ]

    def download_dataset(self):
        filename = "../rasters/bands_and_terrain.tiff"
        if not os.path.exists("../rasters/"):
            os.makedirs(name="../rasters/")
        if os.path.exists(filename):
            print("file downloaded")
            self.path_to_tiff_file = filename
            return filename
        url = "https://storage.yandexcloud.net/skoltech/forestmapping/bands_and_terrain_texture.tiff"
        filename = wget.download(url, out="../rasters/")
        self.path_to_tiff_file = filename
        return filename

    def normalize(self, band: pd.Series) -> pd.Series:
        band_min, band_max = (band.min(), band.max())
        return (band - band_min) / ((band_max - band_min))

    def scale_geom(self, shape, scale: float):
        return affinity.scale(shape, xfact=scale, yfact=scale, origin="center")

    def get_radius(self, x) -> float:
        return np.sqrt(x.area / np.pi)

    def normalize_pixel(self, X: np.ndarray) -> np.ndarray:
        X = X / 10000
        X = np.clip(X, 0, 0.3)
        return X

    def procces_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        non_forest = {
            "вода": 11,
            "болото": 16,
            "вырубка": 13,
            "гарь": 15,
            "земля": 12,
            "трава": 14,
        }

        dictRename = {
            "С": "Pine",
            "Е": "Fir",
            "ОС": "Aspen",
            "К": "Cedar",
            "Л": "Larch",
            "Б": "Birch",
            "П": "Silver fir",
            "гарь": "Burnt forest",
            "вырубка": "Deforestation",
            "трава": "Grass",
            "земля": "Soil",
            "болото": "Swamp",
            "вода": "Water body",
            "поселения": "Settlements",
        }
        mask = gdf["t_Class"] > 7
        for key, value in non_forest.items():
            mask = gdf["t_Class"] == value
            gdf.loc[mask, "t_Клас"] = key

        gdf.loc[:, "class_name"] = gdf["t_Клас"].apply(lambda x: dictRename[x])

        gdf = gdf.loc[gdf["class_name"] != "Cedar"]
        scale_resample = {
            11: 0.25,
            12: 0.65,
            13: 0.65,
            14: 0.7,
            15: 0.9,
        }
        for key, value in scale_resample.items():
            mask = gdf["t_Class"] == key
            gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].apply(
                lambda x: self.scale_geom(x, scale=value)
            )
        gdf = gdf.drop(columns="t_Клас")
        return gdf

    def get_all_cols(self):
        col_names = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ]
        all_cols = []
        if len(self.dates_images) == 0:
            return col_names
        for date in sorted(self.dates_images):
            for col in col_names:
                all_cols.append(date + "_" + col)
        return all_cols

    def get_svi_cols(self):
        col_names = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
            "NDVI",
            "EVI",
            "MSAVI",
            "NDRE",
            "FCI",
        ]  # CLM
        if len(self.dates_images) == 0:
            return col_names
        all_cols = []
        for date in sorted(self.dates_images):
            for col in col_names:
                all_cols.append(date + "_" + col)
        return all_cols

    def prepare_entire_region(self, gdf: gpd.GeoDataFrame):
        shape = box(*gdf.total_bounds)
        src = rio.open(self.path_to_tiff_file)
        out_image, _ = crop_mask(src, [shape], all_touched=True, crop=True)
        x = out_image[:-4, ...].reshape(
            len(self.cols), out_image.shape[1] * out_image.shape[2]
        )
        df_indices_field = pd.DataFrame(x.T, columns=self.cols)
        df_indices_field = self.get_SVI(df_indices_field)

        src = rio.open(self.path_to_tiff_file)
        out_image, _ = crop_mask(src, [shape], all_touched=True, crop=True)
        s2 = out_image[2, ...]
        sub_m = np.where(s2 > 0, out_image[-4:, ...], -1)
        x = sub_m[-4:, ...].reshape(
            len(self.terrain_cols), out_image.shape[1] * out_image.shape[2]
        )
        df_terrain = pd.DataFrame(x.T, columns=self.terrain_cols)
        mask = df_terrain.max(axis=1) == -1
        df_terrain = df_terrain.loc[~mask]
        return df_indices_field, df_terrain

    def get_dataset(
        self, gdf: gpd.GeoDataFrame, scale: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get dataset for classification

        Args:
            gdf (gpd.GeoDataFrame): Forest Inventory Data.
            scale (float, optional): Value to scale forest plots (initial raidus * scale). Defaults to 1.0.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: dataframes with bands+svi and terrain features
        """
        self.scale = scale

        if not self.path_to_tiff_file:
            print("No initial tiff data - Start loading")
            self.download_dataset()
            print("Done ✅")
        df_indices_field = pd.DataFrame(columns=[*self.cols, "key", "class"])
        shape_ = gdf["geometry"].values[0]
        radius = np.round(
            self.get_radius(self.scale_geom(shape=shape_, scale=self.scale)), 2
        )
        print("Radius of forest plot:", radius)
        print("Start preparing bands", end=" ")
        for invent_plot_data in gdf.iterrows():
            inv_dict = invent_plot_data[1].to_dict()
            if inv_dict["t_Class"] < 8:
                df = self.get_bands_by_shape(inv_dict, self.scale)
            else:
                df = self.get_bands_by_shape(inv_dict, 1.0)
            df_indices_field = pd.concat([df_indices_field, df])

        df_indices_field = self.get_SVI(df_indices_field)
        print(" -- Done ✅")
        print("Start preparing terrain")
        df_terrain = pd.DataFrame(columns=[*self.terrain_cols, "key", "class"])
        for invent_plot_data in gdf.iterrows():
            inv_dict = invent_plot_data[1].to_dict()
            if inv_dict["t_Class"] < 8:
                df = self.get_terrain_by_shape(inv_dict, self.scale)
            else:
                df = self.get_terrain_by_shape(inv_dict, 1.0)
            self._terrain_point = df
            df_terrain = pd.concat([df_terrain, df])

        print(" -- Done ✅")
        mask = df_terrain["wetnessindex"] < 0
        df_terrain.loc[mask, "wetnessindex"] = 0
        # for col in self.terrain_cols:
        #     df_terrain.loc[:, col] = self.normalize(df_terrain[col])

        print("Start preparing texture")
        df_texture = pd.DataFrame(columns=[*self.texture_columns, "key", "class"])
        for invent_plot_data in gdf.iterrows():
            inv_dict = invent_plot_data[1].to_dict()
            if inv_dict["t_Class"] < 8:
                df = self.get_texture_by_shape(inv_dict, self.scale)
            else:
                df = self.get_texture_by_shape(inv_dict, 1.0)
            self._texture_point = df
            df_texture = pd.concat([df_texture, df])

        print(" -- Done ✅")

        return df_indices_field, df_terrain, df_texture

    def get_SVI(self, df: pd.DataFrame) -> pd.DataFrame:
        nir = df.loc[:, "B08"]
        red = df.loc[:, "B04"]
        red_far = df.loc[:, "B05"]
        df.loc[:, "NDVI"] = self.NDVI(red=red, nir=nir)
        df.loc[:, "NDRE"] = self.NDRE(red_far=red_far, nir=nir)
        df.loc[:, "MSAVI"] = self.MSAVI(red=red, nir=nir)
        df.loc[:, "EVI"] = self.EVI(red=red, nir=nir)
        df.loc[:, "FCI"] = self.FCI(red=red, nir=nir)
        svi_cols = self.get_svi_cols()
        svi_cols.extend(["key", "class"])
        return df[svi_cols]

    def get_bands_by_shape(self, invent_plot_data: dict, scale: float) -> pd.DataFrame:
        """
        Get bands from geotiff by polygon mask

        Input: (Polygon) - shape

        Output: (pd.DataFrame) - df with bands, date, class

        """

        shape = invent_plot_data["geometry"]
        shape = self.scale_geom(shape=shape, scale=scale)
        src = rio.open(self.path_to_tiff_file)
        out_image, _ = crop_mask(src, [shape], all_touched=True, crop=True)
        x = out_image[:13, ...].reshape(
            len(self.cols), out_image.shape[1] * out_image.shape[2]
        )
        df = pd.DataFrame(x.T, columns=self.cols)
        mask = df.sum(axis=1) > 0
        df = df.loc[mask]
        df.loc[:, "key"] = invent_plot_data["key"]
        df.loc[:, "class"] = invent_plot_data["t_Class"]
        df[self.cols] = self.normalize_pixel(df[self.cols].values)
        return df

    def get_terrain_by_shape(
        self, invent_plot_data: dict, scale: float
    ) -> pd.DataFrame:
        """
        Get terrain features from geotiff by polygon mask

        Input: (Polygon) - shape

        Output: (pd.DataFrame) - df with terrain, key, class

        """

        shape = invent_plot_data["geometry"]
        shape = self.scale_geom(shape=shape, scale=scale)
        src = rio.open(self.path_to_tiff_file)

        out_image, _ = crop_mask(src, [shape], all_touched=True, crop=True)
        self._out_image = out_image
        s2 = out_image[2, ...]
        sub_m = np.where(s2 > 0, out_image[16:20, ...], -1)
        self._sub_m = sub_m
        x = sub_m.reshape(
            len(self.terrain_cols), out_image.shape[1] * out_image.shape[2]
        )

        df = pd.DataFrame(x.T, columns=self.terrain_cols)
        mask = df.max(axis=1) == -1
        df = df.loc[~mask]
        df.loc[:, "key"] = invent_plot_data["key"]
        df.loc[:, "class"] = invent_plot_data["t_Class"]

        return df

    def get_texture_by_shape(
        self, invent_plot_data: dict, scale: float
    ) -> pd.DataFrame:
        """
        Get texture features from geotiff by polygon mask

        Input: (Polygon) - shape

        Output: (pd.DataFrame) - df with texture, key, class

        """

        shape = invent_plot_data["geometry"]
        shape = self.scale_geom(shape=shape, scale=scale)
        src = rio.open(self.path_to_tiff_file)

        out_image, _ = crop_mask(src, [shape], all_touched=True, crop=True)
        s2 = out_image[2, ...]
        sub_m = np.where(s2 > 0, out_image[20:32, ...], -1)
        x = sub_m.reshape(
            len(self.texture_columns), out_image.shape[1] * out_image.shape[2]
        )
        df = pd.DataFrame(x.T, columns=self.texture_columns)
        mask = df.max(axis=1) == -1
        df = df.loc[~mask]
        df.loc[:, "key"] = invent_plot_data["key"]
        df.loc[:, "class"] = invent_plot_data["t_Class"]

        return df

    def NDVI(self, red: pd.Series, nir: pd.Series):
        ndvi = (nir - red) / ((nir + red).apply(lambda x: 0.000001 if x == 0 else x))
        return ndvi

    def EVI(self, red: pd.Series, nir: pd.Series):
        evi2 = (
            2.5
            * (nir - red)
            / ((nir + 2.4 * red + 1).apply(lambda x: 0.000001 if x == 0 else x))
        )
        return evi2

    def NDRE(self, red_far: pd.Series, nir: pd.Series):
        ndre = (nir - red_far) / (
            (nir + red_far).apply(lambda x: 0.000001 if x == 0 else x)
        )
        return ndre

    def FCI(self, red: pd.Series, nir: pd.Series):
        fci = np.sqrt(red * nir)
        return fci

    def MSAVI(self, red: pd.Series, nir: pd.Series):
        msavi = (2 * nir + 1 - ((2 * nir + 1) ** 2 - 8 * (nir - red)) ** (1 / 2)) / 2
        return msavi


def get_train_test(
    df: pd.DataFrame, gdf: gpd.GeoDataFrame, test_size: float, spacing: float = 0.015
) -> tuple:
    gdf_84 = gdf.to_crs(epsg=4326)
    gdf_84.loc[:, "latitude"] = gdf_84["geometry"].centroid.y
    gdf_84.loc[:, "longitude"] = gdf_84["geometry"].centroid.x
    data = gdf_84.copy()
    coordinates = (data.longitude, data.latitude)
    values = np.array(data.index)

    train_block, test_block = vd.train_test_split(
        coordinates, values, test_size=test_size, random_state=123
    )
    print(
        "Train and test size for random splits:",
        train_block[0][0].size,
        test_block[0][0].size,
    )

    train_block, test_block = vd.train_test_split(
        coordinates,
        values,
        spacing=spacing,
        test_size=test_size,
        random_state=213,
    )
    print(
        "Train and test size for block splits: ",
        train_block[0][0].size,
        test_block[0][0].size,
    )

    test_keys = gdf.loc[test_block[-2], "key"].values
    train_keys = gdf.loc[train_block[-2], "key"].values
    mask_test = df.loc[:, "key"].isin(test_keys)
    mask_train = df.loc[:, "key"].isin(train_keys)

    train = df.loc[mask_train]
    test = df.loc[mask_test]
    return train, test


def decodeClasses(x):
    dict_names = {
        7: 0,
        1: 1,
        6: 2,
        2: 3,
        5: 4,
        4: 5,
        11: 6,
        12: 7,
        13: 8,
        14: 9,
        15: 10,
        16: 11,
    }
    dict_normal_names = {
        7: "Pine",
        2: "Fir",
        5: "Aspen",
        3: "Cedar",
        4: "Larch",
        1: "Birch",
        6: "Silver fir",
        15: "Burnt forest",
        13: "Deforestation",
        14: "Grass",
        12: "Soil",
        16: "Swamp",
        11: "Water body",
    }
    if x.isdigit():
        reverse = {}
        for key, value in dict_names.items():
            reverse[value] = key
        return dict_normal_names[reverse[int(x)]]
    else:
        return x


def codeClasses(y: pd.Series) -> pd.Series:
    dict_names = {}
    for key, value in zip(y.unique(), range(y.nunique())):
        dict_names[key] = value
    return y.apply(lambda x: dict_names[x])


def decodeClassesLevel1(x):
    dict_names = {
        7: 0,
        1: 1,
        6: 2,
        2: 3,
        5: 4,
        4: 5,
        11: 6,
        12: 7,
        13: 8,
        14: 9,
        15: 10,
        16: 11,
    }
    dict_normal_names = {
        7: "Pine",
        2: "Fir",
        5: "Aspen",
        3: "Cedar",
        4: "Larch",
        1: "Birch",
        6: "Silver fir",
        15: "Burnt forest",
        13: "Deforestation",
        14: "Grass",
        12: "Soil",
        16: "Swamp",
        11: "Water body",
    }
    if x.isdigit():
        reverse = {}
        for key, value in dict_names.items():
            reverse[value] = key
        return dict_normal_names[int(x)]
    else:
        return x


def brighten(band: np.ndarray) -> np.ndarray:
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    outliers_fraction = 0.1
    iso_forest = IsolationForest(contamination=outliers_fraction, random_state=42)
    X = df.iloc[:, :-2]
    outs = iso_forest.fit(X).predict(X)
    mask = outs != -1
    df = df.loc[mask]
    return df


def resample(df: pd.DataFrame):
    rus = SMOTE()
    X_res, y_res = rus.fit_resample(df.iloc[:, :-1], df.iloc[:, -1].astype(int))
    sub_df_resampled = pd.DataFrame(X_res, columns=df.columns[:-1])
    sub_df_resampled.loc[:, "class"] = y_res


def resample_forest(df: pd.DataFrame):
    rus = SMOTE()
    forest = df.loc[df["class"] < 8]
    X_res, y_res = rus.fit_resample(forest.iloc[:, :-1], forest.iloc[:, -1].astype(int))
    sub_df_resampled = pd.DataFrame(X_res, columns=df.columns[:-1])
    sub_df_resampled.loc[:, "class"] = y_res
    df = pd.concat([df.loc[df["class"] > 8], sub_df_resampled])
    return df


# def scale_normalize(df: pd.DataFrame) -> pd.DataFrame:
#     _X = df.iloc[:, :-2]
#     scaled = StandardScaler().fit_transform(_X)
#     df.iloc[:, :-2] = scaled
#     return df


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
        "RidgeClassifier",
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
        RidgeClassifier(
            random_state=42,
            solver="sag",
            class_weight=class_weights,
            fit_intercept=True,
            alpha=1.1,
            tol=1e-5,
        ),
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
        ),
        #                    XGBClassifier(n_jobs=-1, tree_method='gpu_hist', predictor='gpu_predictor', booster='gblinear', eta=0.3, gamma='auto', max_depth=20)
    ]


# attaching clusters to each row according to the number of plot
# attaching clusters to each row according to the number of plot
def get_cluster_pixels(
    data: pd.DataFrame, key: int = 1, correlation_threshold: float = 0.7
) -> pd.DataFrame:
    attmpt = data[data.key == key]
    attmpt_c = attmpt.drop(columns=["key", "class"]).corr().abs()  #'index',
    # attmpt.corr().style.background_gradient(cmap="Blues")

    # Select upper triangle of correlation matrix
    upper = attmpt_c.where(np.triu(np.ones(attmpt_c.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.95
    to_drop = [
        column for column in upper.columns if any(upper[column] > correlation_threshold)
    ]

    # Drop features
    attmpt_ = attmpt.drop(to_drop, axis=1)  # , inplace=True)

    # preprocessing of the data
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(attmpt_.drop(columns=["key", "class"]))  #'index',
    scaled_data = attmpt_.drop(columns=["key", "class"])  #'index',
    # from yellowbrick.cluster import KElbowVisualizer
    model = KMeans(n_init=10)
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(1, 8), timings=True)
    visualizer.fit(scaled_data)  # Fit data to visualizer
    plt.close()
    elbow_value = visualizer.elbow_value_
    if elbow_value == None:
        elbow_value = 2
    kmeans_model = KMeans(
        n_clusters=elbow_value, random_state=100
    )  # elbow_value_ == number of clusters
    kmeans_model.fit(scaled_data)

    attmpt["clusters"] = kmeans_model.labels_
    attmpt.clusters.value_counts().reset_index()  # .duplicated(subset=['clusters'])#.iloc[0,0]

    return attmpt


##selection of rows related to most abundant clusters


def get_selection(attmpt: pd.DataFrame) -> pd.DataFrame:

    cluster_stat = attmpt.clusters.value_counts().to_dict()
    cluster_count = list(cluster_stat.values())
    cluster_non_equal = cluster_count[0] > cluster_count[1]
    if cluster_non_equal:
        target_cluster = list(cluster_stat.keys())[0]
        mask = attmpt.clusters == target_cluster
        data_grol = attmpt.loc[mask]
    else:
        print("equal cluster", end="")
        data_grol = pd.DataFrame()
    return data_grol


def get_gdf_dataset(
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
