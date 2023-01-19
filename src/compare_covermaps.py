import os
from pathlib import Path

import cartopy.io.shapereader as shpreader
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from shapely import wkt
from shapely.geometry import GeometryCollection, MultiPolygon, Point
from sklearn.metrics import classification_report

COUNTRIES = ["Kenya", "Togo", "Tanzania"]
TEST_COUNTRIES = ["Kenya", "Togo", "Tanzania_CEO_2019"]
DATA_PATH = "../data/datasets/"
TEST_CODE = {"Kenya": "KEN", "Togo": "TGO", "Tanzania_CEO_2019": "TZA"}
DATASET_PATH = Path(DATA_PATH).glob("*")
TARGET_PATHS = [p for p in DATASET_PATH if p.stem in TEST_COUNTRIES]  # test data path
NE_GDF = gpd.read_file(
    shpreader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
)

# Takes dictionary with keys corresponding to earth engine image link and corresponding testing columns for maps
def extract_harvest(images: dict):
    harvest = []
    for map in images.keys():
        name = map.split("/")[-1]
        print(name)

        # Convert image to image collection
        sampled = ee.ImageCollection([ee.Image(map)])
        sampled = sampled.map(lambda x: raster_extraction(x, 10, images[map])).flatten()

        # Convert to gdf
        sampled = geemap.ee_to_gdf(sampled)

        # Binarize
        sampled["first"] = sampled["mode"].apply(lambda x: 1 if x >= 0.5 else 0)

        harvest.append(sampled)

    harvest = pd.concat(harvest)

    return harvest


# --- SUPPORTING FUNCTIONS ---
def raster_extraction(
    image, 
    fc, 
    resolution, 
    reducer=ee.Reducer.first(), 
    crs="EPSG:4326"
) -> ee.FeatureCollection:
    """
    Mapping function used to extract values, given a feature collection, from an earth engine image.
    """
    fc_sub = fc.filterBounds(image.geometry())

    feature = image.reduceRegions(
        collection=fc_sub,
        reducer=reducer,
        scale=resolution,
        crs=crs
    )
    
    return feature

def bufferPoints(
    radius: int, 
    bounds: bool
) -> function:
    """
    Generates function to add buffering radius to point. "bound" (bool) snap boundaries of radii to square pixel
    """
    def function(pt):
        pt = ee.Feature(pt)
        return pt.buffer(radius).bounds() if bounds else pt.buffer(radius)

    return function

# Extract raster values from earth engine image collection. 
def extract_points(
    ic: ee.ImageCollection, 
    fc: ee.FeatureCollection, 
    resolution=10, 
    projection='EPSG:4326'
) -> gpd.GeoDataFrame:
    """
    Creates geodataframe of extracted values from image collection. Assumes ic parameters are set (date, region, band, etc.). 
    """

    extracted = ic.map(lambda x: raster_extraction(x, resolution, fc, projection=projection)).flatten()
    extracted = geemap.ee_to_gdf(extracted)

    return extracted

def generate_report(
    dataset_name: str, 
    true, 
    pred
    ) -> pd.DataFrame:
    """
    Creates classification report on crop coverage binary values. Assumes 1 indicates cropland.
    """

    report = classification_report(true, pred, output_dict=True)
    return pd.DataFrame(data = {
        "dataset": dataset_name, 
        "crop_f1": report["1"]["f1-score"], 
        "accuracy": report["accuracy"], 
        "crop_support": report["1"]["support"], 
        "noncrop_support": report["0"]["support"], 
        "crop_recall": report["1"]["recall"], 
        "noncrop_recall": report["0"]["recall"],
        "crop_precision": report["1"]["precision"],
        "noncrop_precision": report["0"]["precision"] 
        }, index=[0])

def create_point(row) -> ee.Feature:
    """
    Creates ee.Feature from longitude and latitude coordinates from a dataframe
    """

    geom = ee.Geometry.Point(row.lon, row.lat)
    prop = dict(row[["lon", "lat", "binary"]])

    return ee.Feature(geom, prop)


def filter_by_bounds(
    country: str, 
    gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:

    """
    Filters out data in gdf that is not within country bounds
    """
    boundary = NE_GDF.loc[NE_GDF["adm1_code"].str.startswith(country), :].copy()

    if boundary.crs == None:
        boundary = boundary.set_crs("epsg:4326")
    if boundary.crs != "epsg:4326":
        boundary = boundary.to_crs("epsg:4326")

    boundary = GeometryCollection([x for x in boundary["geometry"]])

    mask = gdf.within(boundary)
    filtered = gdf.loc[mask].copy()

    return filtered


def generate_test_data(target_paths: str) -> gpd.GeoDataFrame:
    """
    Creates geodataframe containing all test points from labeled maps.
    """

    test_set = []

    for p in target_paths:
        # Set dict key name
        key = p.stem
        print(key)

        # Read in data and extract test values and points
        gdf = read_test(p)

        print(len(gdf))
        gdf = filter_by_bounds(TEST_CODE[key], gdf)
        print(len(gdf))

        test_set.append(gdf)

    test = gpd.GeoDataFrame(pd.concat(test_set))
    test.reset_index(inplace=True, drop=True)

    return test

def read_test(path: str) -> gpd.GeoDataFrame:
    """
    Opens and binarizes dataframe used for test set.
    """
    test = pd.read_csv(path)
    test = gpd.GeoDataFrame(test, geometry=gpd.points_from_xy(test.lon, test.lat), crs="epsg:4326")
    test = test.loc[test["subset"] == "testing"]
    test["binary"] = test["class_probability"].apply(lambda x: 1 if x >= 0.5 else 0)

    return test
