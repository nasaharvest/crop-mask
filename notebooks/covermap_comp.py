import os
from pathlib import Path

import ee
import geemap
import geopandas as gdp
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

ee.Authenticate()
ee.Initialize()

DATA_PATH = "../data/datasets/"
TARGET_PATHS = [p for p in Path(DATA_PATH).glob("*") if p.stem in TEST_COUNTRIES]

# TEST_COUNTRIES = ["Kenya", "Togo", "Tanzania_CEO_2019"]
TEST_COUNTRIES = ["Kenya", "Togo"]
# DATASETS = ["cop", "esa", "glad", "harvest_togo", "harvest_kenya", "harvest_tanzania"]
DATASETS = ["harvest_togo", "harvest_kenya"]
HARVEST_MAPS = {
    "harvest_togo": "projects/sat-io/open-datasets/nasa-harvest/togo_cropland_binary",
    "harvest_kenya": "projects/sat-io/open-datasets/nasa-harvest/kenya_cropland_binary",
}

METRICS = [
    "dataset",
    "accuracy",
    "crop_f1",
    "crop_support",
    "noncrop_support",
    "crop_precision",
    "crop_recall",
    "noncrop_precision",
    "noncrop_recall",
]


def compare_maps() -> dict:
    """
    Returns dictionary of metrics... **INCOMPLETE**

    """
    test_data = retrieve_test_data()
    test_coll = ee.FeatureCollection(test_data["ee_pts"].tolist())
    results = {}

    for p in TARGET_PATHS:
        key = p.stem
        results[key] = pd.DataFrame(columns=METRICS)

    print(results)

    # Harvest Data
    for map in HARVEST_MAPS.keys():
        print(map)
        sampled = geemap.ee_to_gdf(ee.Image(HARVEST_MAPS[map]).sampleRegions(collection=test_coll))
        test_data[key] = pd.merge(test_data, sampled, on=["lat", "lon"], how="left")["b1"]

    print("compute_results")

    return compute_results(test_data, results)


# Remaps probabilities to binary values
def map_values(val, value_for_crop):
    if val == value_for_crop:
        return 1
    else:
        return 0


# Function used in map function to extract from feature collection
def raster_extraction(image, resolution, f_collection):
    feature = image.sampleRegions(collection=f_collection, scale=resolution)
    return feature


# Convert sklearn classification report dict to
def report_to_row(dataset, report, df):
    new_report = pd.DataFrame(
        data={
            "dataset": dataset,
            "accuracy": report["accuracy"],
            "crop_f1": report["1"]["f1-score"],
            "crop_support": report["1"]["support"],
            "noncrop_support": report["0"]["support"],
            "crop_precision": report["1"]["precision"],
            "crop_recall": report["1"]["recall"],
            "noncrop_precision": report["0"]["precision"],
            "noncrop_recall": report["0"]["recall"],
        },
        index=[0],
    )

    return pd.concat([df, new_report])


# Creates ee.Feature from longitude and latitude coordinates from a dataframe
def create_point(row):
    geom = ee.Geometry.Point(row["lon"], row["lat"])
    prop = dict(row)

    return ee.Feature(geom, prop)


# Gets test data used for comparison
def retrieve_test_data():
    test_data = pd.DataFrame(columns=["lat", "lon", "test_class", "ee_pts", "country"])
    test_set = []

    for p in TARGET_PATHS:
        # Set dict key name
        key = p.stem

        # Read in data and extract test values and points
        df = pd.read_csv(p)
        df = df.loc[df["subset"] == "testing"]
        df = df[["lat", "lon", "class_probability"]]

        # Create earth engine geometry points
        df["ee_pts"] = df.apply(create_point, axis=1)

        # Recast points as 1 or 0 (threshold = 0.5)
        df["test_class"] = df["class_probability"].apply(lambda x: 1 if x > 0.5 else 0)

        df["country"] = key

        test_set.append(df)

    test_data = pd.concat(test_set)
    test_data.reset_index(inplace=True)
    test_data.drop("index", axis=1, inplace=True)

    return test_data


# Computes evaluation
def compute_results(test_data, results):
    for country, df in test_data.groupby("country"):
        for dataset in DATASETS:
            # If country is non-empty
            if not pd.isnull(df[dataset]).all() or not np.isnan(np.unique(df[dataset])[1]):
                print(country + ": " + dataset)
                # Remove na values
                temp = df[["test_class", dataset]].dropna()
                if len(temp) > 10:
                    report = classification_report(
                        temp["test_class"], temp[dataset], output_dict=True
                    )

                results[country] = report_to_row(dataset, report, results[country])

    return results
