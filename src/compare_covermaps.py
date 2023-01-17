import os
import geemap
import cartopy.io.shapereader as shpreader 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rio

from pathlib import Path
from shapely import wkt
from sklearn.metrics import classification_report 
from shapely.geometry import Point, MultiPolygon, GeometryCollection

COUNTRIES = ["Kenya", "Togo", "Tanzania"]
TEST_COUNTRIES = ["Kenya", "Togo", "Tanzania_CEO_2019"]
DATA_PATH = "../data/datasets/"
TEST_CODE = {"Kenya": "KEN", "Togo": "TGO", "Tanzania_CEO_2019": "TZA"} 
DATASET_PATH = Path(DATA_PATH).glob("*")
TARGET_PATHS = [p for p in DATASET_PATH if p.stem in TEST_COUNTRIES] # test data path
NE_GDF = gpd.read_file(
    shpreader.natural_earth(
        resolution='10m', 
        category='cultural', 
        name='admin_1_states_provinces')
)

# Takes dictionary with keys corresponding to earth engine image link and corresponding testing columns for maps
def extract_harvest(images: dict):
    harvest = []
    for map in images.keys():
        name = map.split('/')[-1]
        print(name)

        # Convert image to image collection
        sampled = ee.ImageCollection([ee.Image(map)])
        sampled = sampled.map(lambda x: raster_extraction(x, 10, images[map])).flatten()

        # Convert to gdf
        sampled = geemap.ee_to_gdf(sampled)

        # Binarize 
        sampled['first'] = sampled['mode'].apply(lambda x: 1 if x>=0.5 else 0) 

        harvest.append(sampled)

    harvest = pd.concat(harvest)

    return harvest 

# --- SUPPORTING FUNCTIONS ---

# Function used in map function to extract from feature collection
def raster_extraction(image, resolution, fc, projection="EPSG:4326"):
    
    # Filter feature collection to only points within image
    fc_sub = fc.filterBounds(image.geometry())

    feature = image.reduceRegions(
        collection=fc_sub,
        reducer=ee.Reducer.mode(),
        scale=resolution,
        crs=projection
    )
    return feature

# Creates buffer for earth engine points
def bufferPoints(radius, bounds):
    def function(pt):
        pt = ee.Feature(pt)
        return pt.buffer(radius).bounds() if bounds else pt.buffer(radius)
    return function

# Convert sklearn classification report dict to 
def generate_report(dataset, report):
    return pd.DataFrame(data = {
        "dataset": dataset, 
        "accuracy": report["accuracy"], 
        "crop_f1": report["1"]["f1-score"], 
        "crop_support": report["1"]["support"], 
        "noncrop_support": report["0"]["support"], 
        "crop_precision": report["1"]["precision"], 
        "crop_recall": report["1"]["recall"], 
        "noncrop_precision": report["0"]["precision"], 
        "noncrop_recall": report["0"]["recall"]
        }, index=[0])

# Creates ee.Feature from longitude and latitude coordinates from a dataframe
def create_point(row):
    geom = ee.Geometry.Point(row.lon, row.lat)
    prop = dict(row[['lon', 'lat', 'binary']])

    return ee.Feature(geom, prop)

# Filters out data in gdf that is not within country bounds 
def filter_by_bounds(country: str, gdf: gpd.GeoDataFrame):
    boundary = NE_GDF.loc[NE_GDF['adm1_code'].str.startswith(country), :].copy()

    if boundary.crs == None:
        boundary = boundary.set_crs('epsg:4326')
    if boundary.crs != 'epsg:4326':
        boundary = boundary.to_crs('epsg:4326')

    boundary = GeometryCollection([x for x in boundary['geometry']])

    mask = gdf.within(boundary)
    filtered = gdf.loc[mask].copy()

    return filtered

# Create a testing GDF
def generate_test_data(target_paths: str):
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

def read_test(path: str):
    test = pd.read_csv(path)
    test = gpd.GeoDataFrame(test, geometry=gpd.points_from_xy(test.lon, test.lat), crs='epsg:4326')
    test = test.loc[test['subset']=='testing']
    test['binary'] = test['class_probability'].apply(lambda x: 1 if x >= 0.5 else 0)

    return test