import os
from pathlib import Path

import cartopy.io.shapereader as shpreader
import ee
import geemap
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pandas as pd

from pathlib import Path
from shapely.geometry import GeometryCollection
from sklearn.metrics import classification_report

# ee.Authenticate()
ee.Initialize()

COUNTRIES = ["Kenya", "Togo", "Tanzania"]
TEST_COUNTRIES = ["Kenya", "Togo", "Tanzania_CEO_2019"]
DATA_PATH = "../data/datasets/"
TEST_CODE = {"Kenya": "KEN", "Togo": "TGO", "Tanzania_CEO_2019": "TZA"}
DATASET_PATH = Path(DATA_PATH).glob("*")
TEST_PATHS = [p for p in DATASET_PATH if p.stem in TEST_COUNTRIES]  # test data path
NE_GDF = gpd.read_file(
    shpreader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
)

HARVEST_TGO = "projects/sat-io/open-datasets/nasa-harvest/togo_cropland_binary"
HARVEST_KEN = "projects/sat-io/open-datasets/nasa-harvest/kenya_cropland_binary"
HARVEST_TZA = "users/adadebay/Tanzania_cropland_2019"
COP = "COPERNICUS/Landcover/100m/Proba-V-C3/Global"
ESA = "ESA/WorldCover/v100"
GLAD = "users/potapovpeter/Global_cropland_2019"

REDUCER = ee.Reducer.mode()
REDUCER_STR = "mode"


def extract_harvest(test):
    """
    Generates and extracts tests points from harvest covermaps. Test set is assumed to follow same format as one generated using the
    generate_test_set function.
    """
    test = test.copy()
    # Create feature collection for each country
    tgo_coll = ee.FeatureCollection(
        test.loc[test["country"] == "Togo"].apply(create_point, axis=1).to_list()
    )
    ken_coll = ee.FeatureCollection(
        test.loc[test["country"] == "Kenya"].apply(create_point, axis=1).to_list()
    )
    tza_coll = ee.FeatureCollection(
        test.loc[test["country"] == "Tanzania"].apply(create_point, axis=1).to_list()
    )

    images = {HARVEST_TGO: tgo_coll, HARVEST_KEN: ken_coll, HARVEST_TZA: tza_coll}

    harvest = []
    for map in images.keys():
        name = map.split("/")[-1]
        print("sampling " + name)

        # Convert image to image collection
        sampled = extract_points(
            ic=ee.ImageCollection(ee.Image(map)), fc=images[map], resolution=10
        )

        # Binarize
        sampled["sampled"] = sampled[REDUCER_STR].apply(lambda x: 1 if x >= 0.5 else 0)

        harvest.append(sampled)

    test["harvest"] = pd.merge(test, pd.concat(harvest), on=["lat", "lon"], how="left")["sampled"]

    return test


def extract_covermaps(test):
    """
    Generates and extracts test points from Copernicus, ESA, and GLAD covermaps. Test set is assumed to follow same format as one generated using the
    generate_test_set function.
    """
    test = test.copy()
    # Create feature collection from test set
    test_coll = ee.FeatureCollection(test.apply(create_point, axis=1).to_list())

    copernicus = (
        ee.ImageCollection(COP)
        .select("discrete_classification")
        .filterDate("2019-01-01", "2019-12-31")
    )
    esa = ee.ImageCollection(ESA)
    glad = ee.ImageCollection(GLAD)

    print("sampling copernicus")
    cop_sampled = extract_points(copernicus, test_coll, 100)
    print("sampling esa")
    esa_sampled = extract_points(esa, test_coll, 10)
    print("sampling glad")
    glad_sampled = extract_points(glad, test_coll, 30)

    test["cop"] = pd.merge(test, cop_sampled, on=["lat", "lon"], how="left")["mode"].apply(
        lambda x: 1 if x == 40 else 0
    )
    test["esa"] = pd.merge(test, esa_sampled, on=["lat", "lon"], how="left")["mode"].apply(
        lambda x: 1 if x == 40 else 0
    )
    test["glad"] = pd.merge(test, glad_sampled, on=["lat", "lon"], how="left")["mode"]

    return test


# --- SUPPORTING FUNCTIONS ---
def create_point(row) -> ee.Feature:
    """
    Creates ee.Feature from longitude and latitude coordinates from a dataframe
    """

    geom = ee.Geometry.Point(row.lon, row.lat)
    prop = dict(row[["lon", "lat", "binary"]])

    return ee.Feature(geom, prop)


def bufferPoints(radius: int, bounds: bool):
    """
    Generates function to add buffering radius to point. "bound" (bool) snap boundaries of radii to square pixel
    """

    def function(pt):
        pt = ee.Feature(pt)
        return pt.buffer(radius).bounds() if bounds else pt.buffer(radius)

    return function


def raster_extraction(
    image, fc, resolution, reducer=REDUCER, crs="EPSG:4326"
) -> ee.FeatureCollection:
    """
    Mapping function used to extract values, given a feature collection, from an earth engine image.
    """
    fc_sub = fc.filterBounds(image.geometry())

    feature = image.reduceRegions(collection=fc_sub, reducer=reducer, scale=resolution, crs=crs)

    return feature


def extract_points(
    ic: ee.ImageCollection, fc: ee.FeatureCollection, resolution=10, projection="EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Creates geodataframe of extracted values from image collection. Assumes ic parameters are set (date, region, band, etc.).
    """

    extracted = ic.map(lambda x: raster_extraction(x, fc, resolution, crs=projection)).flatten()
    extracted = geemap.ee_to_gdf(extracted)

    return extracted


def filter_by_bounds(country: str, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

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


def read_test(path: str) -> gpd.GeoDataFrame:
    """
    Opens and binarizes dataframe used for test set.
    """
    test = pd.read_csv(path)[["lat", "lon", "class_probability", "country", "subset"]]
    test = gpd.GeoDataFrame(test, geometry=gpd.points_from_xy(test.lon, test.lat), crs="epsg:4326")
    test = test.loc[test["subset"] == "testing"]
    test["binary"] = test["class_probability"].apply(lambda x: 1 if x >= 0.5 else 0)

    return test


def generate_test_data(target_paths: TEST_PATHS) -> gpd.GeoDataFrame:
    """
    Returns geodataframe containing all test points from labeled maps.
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


def generate_report(dataset_name: str, country: str, true, pred) -> pd.DataFrame:
    """
    Creates classification report on crop coverage binary values. Assumes 1 indicates cropland.
    """

    report = classification_report(true, pred, output_dict=True)
    return pd.DataFrame(
        data={
            "dataset": dataset_name,
            "country": country,
            "crop_f1": report["1"]["f1-score"],
            "accuracy": report["accuracy"],
            "crop_support": report["1"]["support"],
            "noncrop_support": report["0"]["support"],
            "crop_recall": report["1"]["recall"],
            "noncrop_recall": report["0"]["recall"],
            "crop_precision": report["1"]["precision"],
            "noncrop_precision": report["0"]["precision"],
        },
        index=[0],
    )
