from typing import Tuple, List, Optional, Union

import ast
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.transform import rowcol
from openmapflow.bands import BANDS, S2_BANDS, REMOVED_BANDS
from sklearn.metrics import (
    classification_report, auc, roc_curve,
)

START = 1 # January - 0
END = 13 

MONTHS = [
    "Jan",
    "Feb", 
    "Mar", 
    "Apr", 
    "May", 
    "Jun", 
    "Jul", 
    "Aug", 
    "Sep", 
    "Oct", 
    "Nov", 
    "Dec"
]

BANDS_DICT = {
    band : index for index, band in enumerate(BANDS) 
    if band in (S2_BANDS + ["NDVI"]) and band not in REMOVED_BANDS
}

def find_nearest(array : np.ndarray, value : float) -> int:
    idx = (np.abs(array - value)).argmin()
    return idx

def load_gdf(fn : str) -> gpd.GeoDataFrame:
    df = pd.read_csv(fn)
    df = df[~df.eo_data.isna()] 
    gdf = gpd.GeoDataFrame(data = df, geometry = gpd.points_from_xy(df.eo_lon, df.eo_lat))
    return gdf

def load_map(fn : str) -> np.ndarray:
    src = rio.open(fn)
    map_array, map_meta = src.read(), src.meta
    return map_array, map_meta

def plot_map(map_array : np.ndarray) -> None:
    plt.imshow(map_array.squeeze())
    plt.axis("off")
    plt.tight_layout()

def find_threshold(df : pd.DataFrame, fpr_threshold : float) -> float:
    fpr, tpr, thresholds = roc_curve(df.actual, df.probabilities)
    idx = find_nearest(fpr, fpr_threshold)
    threshold = thresholds[idx]
    print("FPR of at most {:.2f} - Threshold {:.4f} w/ TPR {:.4f}"
          .format(fpr_threshold, threshold, tpr[idx])
    )

# Adapted from 'binarize' in area_utils.py
def binarize_map(array : np.ndarray, meta : dict, threshold : Optional[float] = 0.50) -> np.ndarray: 
    binary = np.copy(array)
    binary[binary < threshold] = 0
    binary[((binary >= threshold) & (binary != meta["nodata"]))] = 1
    return binary.astype(np.uint8)

# Adapted from 'reference_sample_agree' in area_utils.py
def extract_from_map(
        gdf : gpd.GeoDataFrame, 
        binary_map : np.ndarray, 
        map_array : np.ndarray, 
        map_meta : dict
    ) -> Tuple[pd.Series]:
    """ Extracts true labels, probabilities, and predictions from corresponding map and geodataframe locations. """

    actuals = gdf["class_probability"].astype(np.uint8)
    probabilities = pd.Series(dtype = np.float32)
    predictions = pd.Series(dtype = np.uint8)
    for r, row in gdf.iterrows():
        geometry = row["geometry"]
        x, y = geometry.x, geometry.y
        px, py = rowcol(map_meta["transform"], x, y)
        probabilities.loc[r] = map_array[px, py]
        predictions.loc[r] = binary_map[px, py]
    return actuals, probabilities, predictions

def compare_gdf_and_map(
        gdf : gpd.GeoDataFrame, 
        map_array : np.ndarray, 
        map_meta : dict, 
        threshold : Optional[float] = 0.50
    ) -> pd.DataFrame:
    """ Creates dataframe w/ comparison of corresponding map and geodataframe locations. """

    binary_map = binarize_map(map_array, map_meta, threshold)
    actuals, probabilities, predictions = extract_from_map(gdf, binary_map, map_array, map_meta)
    comparison = pd.DataFrame({
        "actual" : actuals,
        "prediction" : predictions,
        "probabilities" : probabilities,
        "eo_data" : gdf["eo_data"],
        "crop" : gdf["Crop (Crop type)"],
        "geometry" : gdf.geometry
    })
    return comparison

def postprocess_predictions_dynamic(df : pd.DataFrame, c : Optional[float] = 1.0) -> pd.DataFrame:
    """ Filters positive predictions by values less than mean and one standard deviation. """
    
    postprocessed = df.copy()
    # Extract data
    data = np.array(list(df["eo_data"].apply(ast.literal_eval)))[:,START:END,:]
    indices = calculate_indices(data)
    t, ch = np.roll(MONTHS,-START).tolist().index("Sep"), 0 # argmax 
    # Filter
    pospreds = df.prediction == 1
    mean = np.mean(indices[pospreds,t,ch], axis = 0)
    std = np.std(indices[pospreds,t,ch], axis = 0)
    threshold = mean - (c * std)
    # Reclassification
    postprocessed.loc[(indices[:,t,ch] < threshold), "prediction"] = 0
    return postprocessed

def postprocess_predictions_constant(df: pd.DataFrame) -> pd.DataFrame:
    postprocessed = df.copy()
    # Extract data
    data = np.array(list(df["eo_data"].apply(ast.literal_eval)))[:,START:END,:]
    indices = calculate_indices(data)
    t, ch = np.roll(MONTHS, -START).tolist().index("Sep"), 0
    # Filter
    poslabels = df.actual == 1
    mean = np.mean(indices[poslabels,t,ch], axis = 0)
    std = np.std(indices[poslabels,t,ch], axis = 0)
    threshold = mean - std
    # Reclassification
    postprocessed.loc[(indices[:,t,ch] < threshold), "prediction"] = 0
    return postprocessed

def constant(gdf : gpd.GeoDataFrame, map_array : np.ndarray, map_meta : dict) -> np.ndarray:
    """ A static heuristic approach for filtering false positives (FP) to true negatives (TN). """
    binary_map = binarize_map(map_array, map_meta)
    data = np.array(list(gdf["eo_data"].apply(ast.literal_eval)))
    indices = calculate_indices(data)
    t, ch = np.roll(MONTHS, -START).tolist().index("Sep"), 0
    # Filter
    poslabels = gdf.class_probability == 1.0
    mean, std = np.mean(indices[poslabels,t,ch]), np.std(indices[poslabels,t,ch])
    threshold = (mean - std)
    # Reclassification
    for _, row in gdf.loc[np.where(indices[:,t,ch] < threshold)[0],:].iterrows():
        geometry = row["geometry"]
        x, y = geometry.x, geometry.y
        px, py = rowcol(map_meta["transform"], x, y)
        binary_map[px, py] = 0
    return binary_map

def calculate_indices(array : np.ndarray) -> np.ndarray:
    """ Calculates additional vegetation indices - GNDVI, EVI, and CVI """
    b2 = array[:,:,BANDS_DICT["B2"]] / 10_000
    b3 = array[:,:,BANDS_DICT["B3"]] / 10_000
    b4 = array[:,:,BANDS_DICT["B4"]] / 10_000
    b8 = array[:,:,BANDS_DICT["B8"]] / 10_000

    ndvi = (b8 - b4) / (b8 + b4)
    gndvi = (b8 - b3) / (b8 + b3)
    evi = 2.5 * ((b8 - b4) / (b8 + 6*b4 - 7.5*b2 + 1))
    cvi = b8 - (b3 - 1)
    return np.stack([ndvi, gndvi, evi, cvi], axis = -1)

def print_report(df : pd.DataFrame) -> None:
    report = classification_report(df.actual, df.prediction)
    print(report)

def plot_timeseries(
        df : Union[pd.DataFrame, gpd.GeoDataFrame],
        indices : List[pd.Series],
        labels : List[str],
        title : str
    ) -> None:

    # Extract EO data
    data = np.array(list(df["eo_data"].apply(ast.literal_eval)))[:,START:END,:]
    # Plot
    fig, axes = plt.subplots(nrows = 3, ncols = 4, figsize = (22, 14))
    x = np.arange(data.shape[1])
    for ax, (b, i) in zip(axes.flatten(), BANDS_DICT.items()):
        band = data[:,:,i]
        for index, label in zip(indices, labels):
            mean = np.mean(band[index], axis = 0)
            std = np.std(band[index], axis = 0)
            ax.plot(mean, label = f"{label} ({index.sum()})")
            ax.fill_between(x, mean - std, mean + std, alpha = 0.25)
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(np.roll(MONTHS, -START))
        ax.set_xlabel("Month")
        ax.set_ylabel(f"Mean {b}")
        ax.set_title(f"{b}", fontweight = "bold")
        ax.legend()
        ax.grid()
    fig.suptitle(title, fontsize = 22, y = 1.0)
    plt.tight_layout()

def plot_actuals(gdf : gpd.GeoDataFrame) -> None:
    crop = gdf["class_probability"] == 1.0
    fallow_weed = (gdf["Crop (Crop type)"] == "Fallow with weeds or grass")
    fallow_none = (gdf["Crop (Crop type)"] == "Fallow (no vegetation at all)")
    labels = ["Crop", "Fallow Weed", "Fallow None"]
    title = "Crop vs. Non Crop Signal"
    plot_timeseries(gdf, [crop, fallow_weed, fallow_none], labels, title)

def plot_predictions(df : pd.DataFrame) -> None:
    pospreds = df.prediction == 1
    negpreds = df.prediction == 0 
    labels = ["Positive Prediction", "Negative Prediction"]
    title = "Positive vs. Negative Predictions"
    plot_timeseries(df, [pospreds, negpreds], labels, title)

def plot_tn_fp(df : pd.DataFrame) -> None:
    negatives = df.loc[df.actual == 0, :]
    tn = negatives.prediction == 0
    fp = negatives.prediction == 1
    labels = ["True Negative", "False Positive"]
    title = "True Negative vs. False Positive Signal"
    plot_timeseries(negatives, [tn, fp], labels, title)

def plot_indices(df : pd.DataFrame) -> None:
    data = np.array(list(df["eo_data"].apply(ast.literal_eval)))[:,START:END,:]
    vegetation_indices = calculate_indices(data)
    # Indices
    fallow_weeds_idx = (df["Crop (Crop type)"] == "Fallow with weeds or grass")
    fallow_none_idx = (df["Crop (Crop type)"] == "Fallow (no vegetation at all)")
    crops_idx = (df.class_probability == 1.0)
    indices = [crops_idx, fallow_weeds_idx, fallow_none_idx]
    # Labels
    labels = ["Crop","Fallow Weed", "Fallow None"]
    vegetations = ["NDVI", "GNDVI", "EVI", "GCVI", "BSI"]
    # Plot
    fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (22, 6))
    x = np.arange(12)
    for i, ax in enumerate(axes.flatten()):
        band = vegetation_indices[:,:,i]
        for j in np.arange(3):
            idx = indices[j]
            label = labels[j]

            mean = np.mean(band[idx], axis = 0)
            std = np.std(band[idx], axis = 0)

            ax.plot(mean, label = f"{label} ({idx.sum()})")
            ax.fill_between(x, mean - std, mean + std, alpha = 0.25)
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(np.roll(MONTHS, -START))
        ax.set_xlabel("Month")
        ax.set_ylabel(f"Mean {vegetations[i]}")
        ax.set_title(f"{vegetations[i]}", fontweight = "bold")
        ax.legend()
        ax.grid()
    fig.suptitle("Crop vs. Non-Crop Signal", fontsize = 22, y = 1.0)
    plt.tight_layout()

def plot_roc(
        df : pd.DataFrame,
        value : Optional[float] = 0.50
    ) -> None:
    fpr, tpr, thresholds = roc_curve(df.actual, df.probabilities)
    auroc = auc(fpr, tpr)

    rand = np.linspace(0, 1, len(fpr))
    idx = find_nearest(thresholds, value)

    _, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
    ax.plot(fpr, tpr, label = "AUC {:.4f}".format(auroc))
    ax.plot(rand, rand, ls = "dashed")
    ax.plot(fpr[idx], tpr[idx], marker = "*", ms = 12, mec = "r", mfc = "r")

    # Pretty plot
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC", fontweight = "bold", fontsize = 16)
    ax.legend(loc = "lower right", handletextpad = 0.25, handlelength = 0)
    ax.grid()