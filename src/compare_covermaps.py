__author__ = "Adam Yang"

from pathlib import Path

import cartopy.io.shapereader as shpreader
import ee
import geemap
import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection
from sklearn.metrics import classification_report, confusion_matrix
from area_utils import *

DATA_PATH = "../data/datasets/"
# Country codes for Natural Earth bounding box according file name of testing set
# (try looking up 3 letter codes)
TEST_CODE = {
    "Kenya": "KEN",
    "Togo": "TGO",
    "Tanzania": "TZA",
    "Tanzania_CEO_2019": "TZA",
    "Malawi": "MWI",
    "Mali": "MLI",
    "Namibia": "NAM",
    "Rwanda": "RWA",
    "Uganda": "UGA",
    "Zambia": "ZMB",
}
DATASET_PATH = Path(DATA_PATH).glob("*")
NE_GDF = gpd.read_file(
    shpreader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
)

# Hold directories of test files
TEST_COUNTRIES = {
    "Kenya": DATA_PATH + "KenyaCEO2019.csv",
    "Togo": DATA_PATH + "Togo.csv",
    "Tanzania": DATA_PATH + "Tanzania_CEO_2019.csv",
    "Malawi": DATA_PATH + "Malawi_CEO_2020.csv",
    "Mali": DATA_PATH + "MaliStratifiedCEO2019.csv",
    "Namibia": DATA_PATH + "Namibia_CEO_2020.csv",
    "Rwanda": DATA_PATH + "Rwanda.csv",
    "Uganda": DATA_PATH + "Uganda.csv",
    "Zambia": DATA_PATH + "Zambia_CEO_2019.csv",
}

REDUCER = ee.Reducer.mode()
# MAP_COLUMN = "extracted"
REDUCER_STR = "mode"
MAP_COLUMN = "mode"
LAT = "lat"
LON = "lon"
CLASS_COL = "binary"
COUNTRY_COL = "country"


class Covermap:
    """
    Object class for earth engine covermap. Covermap is defined by:
        1. Dataset title (ex: 'dynamicworld', 'gfsad-lgrip' )
        2. earth engine asset (must be image collection)
        3. resolution (int in meters)
        4. countries covered (if left blank, we assume it covers all countries in test_countries,
            as is the case with global datasets)
        5. the final parameter can either be a float defining the treshold for positive/negative
        labels (typically 0.5)
           OR a list of labels for cropland
    See examples in TARGETS.
    """

    def __init__(
        self,
        title: str,
        ee_asset: ee.ImageCollection,
        resolution: int,
        countries=None,
        probability=None,
        crop_labels=None,
    ) -> None:
        self.title = title
        self.ee_asset = ee_asset
        self.resolution = resolution

        assert (probability is None) ^ (
            crop_labels is None
        ), "Please specify only 1 of (probability and threshold) or crop_labels"

        self.probability = probability
        self.crop_labels = crop_labels

        if countries is None:
            self.countries = [c for c in TEST_COUNTRIES]
        else:
            self.countries = countries

    def extract_test(self, test: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # Extract from countires that are covered by map
        test_points = test.loc[test[COUNTRY_COL].isin(self.countries)].copy()

        test_coll = ee.FeatureCollection(
            test_points.apply(lambda x: create_point(x), axis=1).to_list()
        )
        sampled = extract_points(ic=self.ee_asset, fc=test_coll, resolution=self.resolution)

        if len(sampled) != len(test_points):
            print("Warning: length of sampled dataset ({}) != test points ({})" \
                    .format(len(sampled), len(test_points)))

        # Recast values
        if self.probability:

            def mapper(x):
                return 1 if x >= self.probability else 0

        else:
            if type(self.crop_labels) is list:

                def mapper(x):
                    return 1 if x in self.crop_labels else 0

            else:
                print("Invalid valid label format")
                return None

        sampled[self.title] = sampled[REDUCER_STR].apply(lambda x: mapper(x))

        assert len(sampled) != 0, "Empty testing set"

        return sampled[[LAT, LON, self.title]]

    def __repr__(self) -> str:
        return self.title + " " + repr(self.countries)


class TestCovermaps:
    """
    Architecture for sampling and comparing ee maps to compare against CropHarvest testing sets.
    Covermaps can be self specified or taken from TARGETS dict. If no countries are given to
    test in, class uses all test sets available.
    """

    def __init__(self, test_countries: list, covermaps: list) -> None:
        if test_countries is None:
            self.test_countries = list(TEST_COUNTRIES.keys())
        else:
            self.test_countries = test_countries
        self.covermaps = covermaps
        self.covermap_titles = [x.title for x in self.covermaps]
        self.sampled_maps: dict = {}
        self.results: dict = {}

    def get_test_points(self, targets=None) -> gpd.GeoDataFrame:
        """
        Returns geodataframe containing all test points from labeled maps.
        *Modified from generate_test_data()*
        """

        if targets is None:
            targets = self.test_countries.copy()

        test_set = []

        for country in targets:
            gdf = read_test(TEST_COUNTRIES[country], country=country)
            gdf = filter_by_bounds(TEST_CODE[country], gdf)

            test_set.append(gdf)

        test = gpd.GeoDataFrame(pd.concat(test_set))
        test.reset_index(inplace=True, drop=True)

        return test

    def extract_covermaps(self, test_df):
        """
        Groups testing points by country then extracts from each map. If map does not include
        specified country, the map is skipped. Resulting extracted points are combined for
        each country then added to sampled_maps, where key = country and value = a dataframe of
        extracted points from maps.
        """
        for country in self.test_countries:
            country_test = test_df.loc[test_df[COUNTRY_COL] == country].copy()

            assert not country_test.is_empty.all(), "No testing points for " + country

            for map in self.covermaps:
                if country in map.countries:
                    print(f"[{country}] sampling " + map.title + "...")

                    map_sampled = map.extract_test(country_test).copy()
                    country_test = pd.merge(country_test, map_sampled, on=[LAT, LON], how="left")
                    country_test.drop_duplicates(
                        inplace=True
                    )  # TODO find why points get duplicated

            self.sampled_maps[country] = country_test

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

                self.results[country] = pd.concat(comparisons).set_index(["dataset"])

            return self.results.copy()

    def __repr__(self) -> str:
        return repr(self.covermap_titles) + " " + repr(self.test_countries)


# Supporting funcs


def generate_test_data(target_paths) -> gpd.GeoDataFrame:
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
    Generates function to add buffering radius to point. "bound" (bool) snap boundaries of radii to
    square pixel
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
    im = image.reproject(crs=crs, scale=resolution)
    feature = im.reduceRegions(collection=fc_sub, reducer=reducer, scale=resolution, crs=crs)

    return feature


def extract_points(
    ic: ee.ImageCollection, fc: ee.FeatureCollection, resolution=10, projection="EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Creates geodataframe of extracted values from image collection. Assumes ic parameters are set
    (date, region, band, etc.).
    """

    extracted = ic.map(lambda x: raster_extraction(x, fc, resolution, crs=projection)).flatten()
    extracted = geemap.ee_to_gdf(extracted)

    return extracted


def filter_by_bounds(country_code: str, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filters out data in gdf that is not within country bounds
    """
    boundary = NE_GDF.loc[NE_GDF["adm1_code"].str.startswith(country_code), :].copy()

    if boundary.crs is None:
        boundary = boundary.set_crs("epsg:4326")
    if boundary.crs != "epsg:4326":
        boundary = boundary.to_crs("epsg:4326")

    boundary = GeometryCollection([x for x in boundary["geometry"]])

    mask = gdf.within(boundary)
    filtered = gdf.loc[mask].copy()

    return filtered


def read_test(path: str, country=None) -> gpd.GeoDataFrame:
    """
    Opens and binarizes dataframe used for test set.
    """

    try:
        test = pd.read_csv(path)[["lat", "lon", "class_probability", "country", "subset"]]
    except KeyError:
        test = pd.read_csv(path)[["lat", "lon", "class_probability", "subset"]]
        if country is None:
            raise Exception("No country given in dataset. Input country string needed")
        test["country"] = country

    test = gpd.GeoDataFrame(test, geometry=gpd.points_from_xy(test.lon, test.lat), crs="epsg:4326")

    # Use only consensus points, and use entire Kenya map
    if country == "Kenya":
        test = test[test["class_probability"].isin([0, 1])]
    else:
        test = test[test["class_probability"].isin([0, 1])].loc[test["subset"] == "testing"]
    test.rename(columns={"class_probability": CLASS_COL}, inplace=True)
    test[CLASS_COL] = test[CLASS_COL].astype(int)

    return test


def generate_report(dataset_name: str, country: str, true, pred) -> pd.DataFrame:
    """
    Creates classification report on crop coverage binary values. Assumes 1 indicates cropland.
    """
    report = classification_report(true, pred, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
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
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        },
        index=[0],
    ).round(2)


# Maps
TARGETS = {
    "copernicus": Covermap(
        "copernicus",
        ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global")
        .select("discrete_classification")
        .filterDate("2019-01-01", "2019-12-31"),
        resolution=100,
        crop_labels=[40],
    ),
    "worldcover": Covermap(
        "worldcover", ee.ImageCollection("ESA/WorldCover/v100"), resolution=10, crop_labels=[40]
    ),
    "glad": Covermap(
        "glad",
        ee.ImageCollection("users/potapovpeter/Global_cropland_2019"),
        resolution=30,
        probability=0.5,
    ),
    # "gfsad": Covermap(
    #     "gfsad",
    #     ee.ImageCollection(ee.Image("USGS/GFSAD1000_V1")),
    #     resolution=1000,
    #     crop_labels=[1, 2, 3, 4, 5],
    # ),
    "asap": Covermap(
        "asap",
        ee.ImageCollection(ee.Image("users/sbaber/asap_mask_crop_v03")),
        resolution=1000,
        crop_labels=list(range(10, 190)),
    ),
    "dynamicworld": Covermap(
        "dynamicworld",
        ee.ImageCollection(
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filter(ee.Filter.date("2019-01-01", "2020-01-01"))
            .select(["label"])
            .mode()
        ),
        resolution=10,
        crop_labels=[4],
    ),
    "gfsad-gcep": Covermap(
        "gfsad-gcep",
        ee.ImageCollection("projects/sat-io/open-datasets/GFSAD/GCEP30"),
        resolution=30,
        crop_labels=[2],
    ),
    "gfsad-lgrip": Covermap(
        "gfsad-lgrip",
        ee.ImageCollection("projects/sat-io/open-datasets/GFSAD/LGRIP30"),
        resolution=30,
        crop_labels=[2, 3],
    ),
    "digital-earth-africa": Covermap(
        "digital-earth-africa",
        ee.ImageCollection("projects/sat-io/open-datasets/DEAF/CROPLAND-EXTENT/filtered"),
        resolution=10,
        crop_labels=[1],
    ),
    "esa-cci-africa": Covermap(
        "esa-cci-africa",
        ee.ImageCollection(
            ee.Image("projects/sat-io/open-datasets/ESA/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v10")
        ),
        resolution=20,
        crop_labels=[4],
    ),
    "globcover-v23": Covermap(
        "globcover-v23",
        ee.ImageCollection(
            ee.Image("projects/sat-io/open-datasets/ESA/GLOBCOVER_L4_200901_200912_V23")
        ),
        resolution=300,
        crop_labels=[11, 14, 20, 30],
    ),
    "globcover-v22": Covermap(
        "globcover-v22",
        ee.ImageCollection(
            ee.Image("projects/sat-io/open-datasets/ESA/GLOBCOVER_200412_200606_V22_Global_CLA")
        ),
        resolution=300,
        crop_labels=[11, 14, 20, 30],
    ),
    "harvest-crop-maps": Covermap(
        "harvest-crop-maps",
        ee.ImageCollection("projects/bsos-geog-harvest1/assets/harvest-crop-maps"),
        resolution=10,
        probability=0.5,
        countries=["Togo", "Kenya", "Malawi"],
    ),
    "esri-lulc": Covermap(
        "esri-lulc",
        ee.ImageCollection(
            "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
        ).filter(ee.Filter.date("2019-01-01", "2020-01-01")),
        resolution=10,
        crop_labels=[5],
    ),
    "nabil-etal-2021": Covermap(
        "nabil-etal-2021",
        ee.ImageCollection.fromImages(
            [ee.Image("projects/sat-io/open-datasets/landcover/AF_Cropland_mask_30m_2016_v3")]
        ),
        resolution=30,
        crop_labels=[2],
    ),
}
