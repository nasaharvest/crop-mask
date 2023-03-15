import os
from pathlib import Path

import cartopy.io.shapereader as shpreader
import ee
import geemap
import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection
from sklearn.metrics import classification_report

# ee.Authenticate()
ee.Initialize()

COUNTRIES = ["Kenya", "Togo", "Tanzania"]
DATA_PATH = "../data/datasets/"
# Country codes for Natural Earth bounding box according file name of testing set
TEST_CODE = {"Kenya": "KEN", "Togo": "TGO", "Tanzania": "TZA", "Tanzania_CEO_2019": "TZA"}
DATASET_PATH = Path(DATA_PATH).glob("*")
NE_GDF = gpd.read_file(
    shpreader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
)

# OLD
TEST_COUNTRIES = ["Kenya", "Togo", "Tanzania_CEO_2019"]
TEST_PATHS = [p for p in DATASET_PATH if p.stem in TEST_COUNTRIES]  # test data path

TEST_COUNTRIES = {
    "Kenya": DATA_PATH + "Kenya.csv",
    "Togo": DATA_PATH + "Togo.csv",
    "Tanzania": DATA_PATH + "Tanzania_CEO_2019.csv",
}

COVERMAPS = {
    "harvest_togo": "projects/sat-io/open-datasets/nasa-harvest/togo_cropland_binary",
    "harvest_kenya": "projects/sat-io/open-datasets/nasa-harvest/kenya_cropland_binary",
    "harvest_tanzania": "users/adadebay/Tanzania_cropland_2019",
    "copernicus": "COPERNICUS/Landcover/100m/Proba-V-C3/Global",
    "esa": "ESA/WorldCover/v100",
    "glad": "users/potapovpeter/Global_cropland_2019",
    "gfsad": "USGS/GFSAD1000_V1",
    "asap": "users/sbaber/asap_mask_crop_v03",
}

REDUCER = ee.Reducer.mode()
REDUCER_STR = "mode"
LAT = "lat"
LON = "lon"
CLASS_COL = "binary"
COUNTRY_COL = "country"

class TestCovermaps:
    """
    Architecture for sampling and comparing ee maps to compare against CropHarvest testing sets. Covermaps can be self specified
    or taken from TARGETS dict. If no countries are given to test in, class uses all test sets available. 
    """

    def __init__(self, test_countries: list, covermaps: list) -> None:
        self.test_countries = test_countries
        if test_countries is None:
            self.test_countries = [country for country in TEST_COUNTRIES]
        self.covermaps = covermaps 
        self.covermap_titles = [x.title for x in self.covermaps]
        self.sampled_maps = {}
        self.results = {}

    def get_test_points(self, targets=None) -> gpd.GeoDataFrame:
        """
        Returns geodataframe containing all test points from labeled maps. Modified from generate_test_data
        """
        if targets == None:
            targets = self.test_countries.copy()

        test_set = []

        for country in targets:
            gdf = read_test(TEST_COUNTRIES[country])
            gdf = filter_by_bounds(TEST_CODE[country], gdf)

            test_set.append(gdf)

        test = gpd.GeoDataFrame(pd.concat(test_set))
        test.reset_index(inplace=True, drop=True)

        return test

    def extract_covermaps(self, test_df):
        """
        Groups testing points by country then extracts from each map. If map does not include
        specified country, the map is skipped. Resulting extracted points are combined for
        each country then added to sampled_maps, where key = country and value = a dataframe of extracted
        points from maps.
        """
        for country in self.test_countries:
            test_temp = test_df.loc[test_df[COUNTRY_COL] == country].copy()

            assert not test_temp.is_empty.all(), "No testing points for " + country

            for map in self.covermaps:
                if country in map.countries:
                    print(f"[{country}] sampling " + map.title + "...")

                    map_sampled = map.extract_test(test_temp).copy()
                    test_temp = pd.merge(test_temp, map_sampled, on=[LAT, LON], how="left")

            self.sampled_maps[country] = test_temp.copy()

        return self.sampled_maps.copy()

    def evaluate(self):
        """
        Evaluates extracted maps using CropHarvest maps as baseline. Groups metrics
        by country, and countries with NaN values will have those row dropped.
        """
        if len(self.sampled_maps) == 0:
            print("No maps extracted")
            return None
        else:
            print("evaluating maps...")
            for country in self.sampled_maps:
                comparisons = []
                dataset = self.sampled_maps[country].copy()

                for map in self.covermap_titles:
                    if map in dataset.columns:
                        temp = dataset[[CLASS_COL, map]].dropna()
                        print("dataset: " + map + " | country: " + country)

                        comparisons.append(
                            generate_report(map, country, temp[CLASS_COL], temp[map])
                        )

                self.results[country] = pd.concat(comparisons)

            return self.results.copy()

class Covermap:
    def __init__(
        self,
        title: str,
        ee_asset: ee.ImageCollection,
        resolution,
        probability=None,
        p_threshold=None,
        crop_labels=None,
        countries=None,
    ) -> None:
        # TODO: Check parameters
        self.title = title
        self.ee_asset = ee_asset
        self.resolution = resolution

        assert probability ^ crop_labels, "Please specify only 1 of (probability and threshold) or crop_labels"

        if probability:
            self.probability = probability  # for harvest maps, where points are crop probability
            self.threshold = p_threshold
        else:
            self.crop_label = crop_labels

        if countries is None:
            self.countries = [c for c in TEST_COUNTRIES]
        else:
            self.countries = countries

    def extract_test(self, test) -> gpd.GeoDataFrame:
        # TODO: Test here for test data params, might not need to worry about since test sets are created through this

        # Extract from countires that are covered by map
        test_points = test.loc[test[COUNTRY_COL].isin(self.countries)].copy()

        test_coll = ee.FeatureCollection(
            test_points.apply(lambda x: create_point(x), axis=1).to_list()
        )
        sampled = extract_points(ic=self.ee_asset, fc=test_coll, resolution=self.resolution)

        if len(sampled) != len(test_points):
            print("Extracting error: length of sampled dataset is not the same as testing dataset")

        # Recast values
        if self.probability:
            mapper = lambda x: 1 if x >= self.threshold else 0
        else:
            if type(self.crop_label) is list:
                mapper = lambda x: 1 if x in self.crop_label else 0
            else:
                print("Invalid valid label format")
                return None

        sampled[self.title] = sampled[REDUCER_STR].apply(lambda x: mapper(x))

        return sampled[[LAT, LON, self.title]]

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

def create_point(row) -> ee.Feature:
    """
    Creates ee.Feature from longitude and latitude coordinates from a dataframe. Column
    values are assumed to be LAT, LON, and CLASS_COLUMN
    """

    geom = ee.Geometry.Point(row.lon, row.lat)
    prop = dict(row[[LON, LAT, CLASS_COL]])

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

TARGETS = {
    "harvest_togo": Covermap(
        "harvest_togo",
        ee.ImageCollection(
            ee.Image("projects/sat-io/open-datasets/nasa-harvest/togo_cropland_binary")
        ),
        resolution=10,
        probability=True,
        p_treshold=0.5,
        countries=["Togo"]
    ),
    "harvest_kenya": Covermap(
        "harvest_kenya",
        ee.ImageCollection(
            ee.Image("projects/sat-io/open-datasets/nasa-harvest/kenya_cropland_binary")
        ),
        resolution=10,
        probability=True,
        p_threshold=0.5,
        countries=["Kenya"]
    ),
    "harvest_tanzania": Covermap(
        "harvest_tanzania",
        ee.ImageCollection(
            ee.Image("users/adadebay/Tanzania_cropland_2019")
        ),
        resolution=10,
        probability=True,
        p_threshold=0.5,
        countries=["Tanzania"]
    ),
    "copernicus": Covermap(
        "copernicus",
        ee.ImageCollection(
            "COPERNICUS/Landcover/100m/Proba-V-C3/Global").select(
            "discrete_classification").filterDate(
            "2019-01-01", "2019-12-31"
        ),
        resolution=100,
        crop_labels=[40]
    ),
    "esa": Covermap(
        "esa",
        ee.ImageCollection(
            "ESA/WorldCover/v100"
        ),
        resolution=10,
        crop_labels=[40]
    ),
    "glad": Covermap(
        "glad",
        ee.ImageCollection(
            "users/potapovpeter/Global_cropland_2019"
        ),
        resolution=30,
        probability=True,
        probability=0.5
    ),
    "gfsad": Covermap(
        "gfsad",
        ee.ImageCollection(
            ee.Image("USGS/GFSAD1000_V1")
        ),
        resolution=1000,
        crop_label=[1, 2, 3, 4, 5]
    ),
    "asap": Covermap(
        "asap",
        ee.ImageCollection(
            ee.Image("users/sbaber/asap_mask_crop_v03")
        ),
        resolution=1000,
        probability=True,
        p_threshold=100
    ),
    "dynamicworld": Covermap(
        "dynamicworld",
        ee.ImageCollection(
            "GOOGLE/DYNAMICWORLD/V1").select(
            "crops").filterDate(
            "2019-01-01", "2019-12-31"
        ),
        resolution=10,
        probability=True,
        p_threshold=0.5
    ),
    "gfsad-gcep": Covermap(
        "gfsad-gcep",
        ee.ImageCollection(
            "projects/sat-io/open-datasets/GFSAD/GCEP30"
        ),
        resolution=30,
        crop_labels=[2]
    ),
    "gfsad-lgrip": Covermap(
        "gfsad-lgrip", 
        ee.ImageCollection(
            "projects/sat-io/open-datasets/GFSAD/LGRIP30"
        ),
        resolution=30,
        crop_labels=[2,3]
    )
}
