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

def extract_harvest(images: dict):
    harvest = []
    for map in images.keys():
        name = map.split('/')[-1]
        print(name)

        #sample from earth engine
        sampled = ee.Image(map).reduceRegions(
            collection=images[map],
            reducer=ee.Reducer.first()
        )
        
        #convert to dataframe
        sampled = geemap.ee_to_gdf(sampled)
        
        #binarize 
        sampled['first'] = sampled['first'].apply(lambda x: 1 if x>=0.5 else 0) 

        harvest.append(sampled)

    harvest = pd.concat(harvest)

def extract_other(copernicus, esa, glad, test_coll):
    cop_clipped = copernicus.select("discrete_classification").filterDate("2019-01-01", "2019-12-31").map(lambda x: raster_extraction(x, 100, test_coll)).flatten()
    cop_sampled = geemap.ee_to_gdf(cop_clipped)
    cop_sampled["cop_class"] = cop_sampled["discrete_classification"].apply(lambda x: 1 if x==40 else 0)

    esa_clipped = esa.filterBounds(test_coll).map(lambda x: raster_extraction(x, 10, test_coll)).flatten()
    esa_sampled = geemap.ee_to_gdf(esa_clipped)
    esa_sampled["esa_class"] = esa_sampled["Map"].apply(lambda x: 1 if x==40 else 0)

    glad_clipped = glad.filterBounds(test_coll).map(lambda x: raster_extraction(x, 30, test_coll)).flatten()
    glad_sampled = geemap.ee_to_gdf(glad_clipped)

    


# --- SUPPORTING FUNCTIONS ---

# Retrieve country boundaries (for test set clipping)


# Remaps classes to crop/noncrop 
def map_values(val, value_for_crop):
    if val == value_for_crop:
        return 1
    else:
        return 0

# Function used in map function to extract from feature collection
def raster_extraction(image, resolution, f_collection):
    feature = image.sampleRegions(
        collection = f_collection,
        scale = resolution
    )
    return feature

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
        gdf = gpd.read_file(p)
        gdf["geometry"] = gpd.points_from_xy(gdf.lon, gdf.lat)

        if gdf.crs == None:
            gdf.set_crs("epsg:4326")
        elif gdf.crs != "epsg:4326":
            gdf.to_crs("epsg:4326")

        gdf = gdf.loc[gdf["subset"]=="testing"]
        gdf = gdf.astype({"lat":"float", "lon": "float", "class_probability": "float"})
        gdf = gdf[["lat", "lon", "class_probability", "country", "geometry"]]
        gdf["binary"] = gdf["class_probability"].apply(lambda x: 1 if x >= 0.5 else 0)
        
        print(len(gdf))
        gdf = filter_by_bounds(TEST_CODE[key], gdf)
        print(len(gdf))

        test_set.append(gdf)

    test = gpd.GeoDataFrame(pd.concat(test_set))
    test.reset_index(inplace=True, drop=True)
    
    return test