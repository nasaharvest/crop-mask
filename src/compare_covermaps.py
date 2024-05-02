__author__ = "Adam Yang, Hannah Kerner"

from pathlib import Path
import re

import cartopy.io.shapereader as shpreader
import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection
from sklearn.metrics import confusion_matrix

from src.area_utils import (
    compute_acc,
    compute_area_error_matrix,
    compute_p_i,
    compute_u_j,
    compute_var_acc,
    compute_var_p_i,
    compute_var_u_j,
)

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
    "Senegal": "SEN",
    "Hawaii": "USA",
    "BlueNile2020": "SDN",
    "BlueNile2019": "SDN",
    "AlGadaref2019": "SDN",
    "GedarefDarfurAlJazirah2022": "SDN",
    "BureJimma2019": "ETH",
    "BureJimma2020": "ETH",
    "Tigray2021": "ETH",
    "Tigray2020": "ETH",
    "Zambia2019": "ZMB",
}
DATASET_PATH = Path(DATA_PATH).glob("*")
NE_GDF = gpd.read_file(
    shpreader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
)

# Hold directories of test files
TEST_COUNTRIES = {
    "Kenya": DATA_PATH + "KenyaCEO2019.csv",
    "Togo": DATA_PATH + "Togo_2019.csv",
    "Tanzania": DATA_PATH + "Tanzania_CEO_2019.csv",
    "Malawi": DATA_PATH + "Malawi_CEO_2020.csv",
    "Mali": DATA_PATH + "MaliStratifiedCEO2019.csv",
    "Namibia": DATA_PATH + "Namibia_CEO_2020.csv",
    "Rwanda": DATA_PATH + "Rwanda_2019.csv",
    "Uganda": DATA_PATH + "Uganda_2019.csv",
    "Zambia": DATA_PATH + "Zambia_CEO_2019.csv",
    "Senegal": DATA_PATH + "Senegal_CEO_2022.csv",
    "Hawaii": DATA_PATH + "Hawaii_CEO_2020.csv",
    "BlueNile2020": DATA_PATH + "SudanBlueNileCEO2020.csv",
    "BlueNile2019": DATA_PATH + "Sudan_Blue_Nile_CEO_2019.csv",
    "AlGadaref2019": DATA_PATH + "SudanAlGadarefCEO2019.csv",
    "GedarefDarfurAlJazirah2022": DATA_PATH + "SudanGedarefDarfurAlJazirah2022.csv",
    "BureJimma2019": DATA_PATH + "Ethiopia_Bure_Jimma_2019.csv",
    "BureJimma2020": DATA_PATH + "Ethiopia_Bure_Jimma_2020.csv",
    "Tigray2021": DATA_PATH + "Ethiopia_Tigray_2021.csv",
    "Tigray2020": DATA_PATH + "Ethiopia_Tigray_2020.csv",
    "Zambia2019": DATA_PATH + "Zambia_CEO_2019.csv",
}

REDUCER = ee.Reducer.mode()
# MAP_COLUMN = "extracted"
REDUCER_STR = "mode"
MAP_COLUMN = "mode"
LAT = "lat"
LON = "lon"
CLASS_COL = "binary"
COUNTRY_COL = "country"
MAX_PIXELS = 1e18


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
        ee_asset_str: str,
        resolution: int,
        countries=None,
        probability=None,
        crop_labels=None,
        years_covered=[],
    ) -> None:
        self.title = title
        self.title_safe = title.replace("-", "_")
        self.ee_asset_str = ee_asset_str
        self.ee_asset = eval(ee_asset_str.replace("\n", "").replace(" ", ""))
        self.resolution = resolution
        self.years_covered = years_covered

        assert (probability is None) ^ (
            crop_labels is None
        ), "Please specify only 1 of (probability and threshold) or crop_labels"

        self.probability = probability
        self.crop_labels = crop_labels

        if countries is None:
            self.countries = [c for c in TEST_COUNTRIES.keys()]
        else:
            self.countries = countries

    def ee_script(self, country: str, include_export: bool = True, include_prefix: bool = True):
        script = ""
        if include_prefix:
            script += f"""
var palettes = require('users/gena/packages:palettes');
var classVis = {{palette: palettes.cmocean.Speed[7].slice(0,-2)}}
var aoi = ee.FeatureCollection("FAO/GAUL/2015/level0")
    .filter(ee.Filter.eq('ADM0_NAME', '{country}'));
Map.centerObject(aoi, 7);\n
"""
        script += f"var {self.title_safe} = {self.ee_asset_str}"
        script += ".filterBounds(aoi).mosaic().clip(aoi);"
        script += f"\n{self.title_safe} = "

        if self.crop_labels is None:
            script += f"{self.title_safe}.gte({self.probability})"

        # Check if crop_labels are an ordered list
        elif len(self.crop_labels) > 2 and self.crop_labels == list(
            range(self.crop_labels[0], self.crop_labels[-1] + 1)
        ):
            script += f"{self.title_safe}.gte({self.crop_labels[0]})"
            script += f".and({self.title_safe}.lte({self.crop_labels[-1]}))"
        else:
            script += f"{self.title_safe}.eq({self.crop_labels[0]})"
            if len(self.crop_labels) > 1:
                for crop_value in self.crop_labels[1:]:
                    script += f".or({self.title_safe}.eq({crop_value}))"
        script += ".rename('crop')"

        script += f"\nMap.addLayer({self.title_safe}, classVis, 'Cropland from {self.title}');"

        if include_export:
            script += f"""
Export.image.toCloudStorage({{
    image: {self.title_safe},
    description: "{country}_{self.title_safe}",
    bucket: 'crop-mask-preds-merged',
    fileNamePrefix: '{country}_{self.title_safe}',
    region: aoi,
    scale: 10,
    crs: "EPSG:4326",
    maxPixels: {MAX_PIXELS},
    skipEmptyTiles: true
}});"""
        return script

    def extract_test(self, test: gpd.GeoDataFrame, year: int) -> gpd.GeoDataFrame:
        # Extract from countries that are covered by map
        test_points = test.loc[test[COUNTRY_COL].isin(self.countries)].copy()

        test_coll = ee.FeatureCollection(
            test_points.apply(lambda x: create_point(x), axis=1).to_list()
        )

        if len(self.years_covered) > 1:
            year_diff = [abs(y - year) for y in self.years_covered]
            nearest_year_idx = min(enumerate(year_diff), key=lambda x: x[1])[0]
            nearest_year = self.years_covered[nearest_year_idx]
            self.ee_asset_str = self.ee_asset_str.replace("2019-01-01", "%s-01-01" % nearest_year)
            self.ee_asset_str = self.ee_asset_str.replace("2019-12-31", "%s-12-31" % nearest_year)
            self.ee_asset = eval(self.ee_asset_str.replace("\n", "").replace(" ", ""))
            print("using closest map year (%s) to test year (%s)" % (nearest_year, year))

        sampled = extract_points(ic=self.ee_asset, fc=test_coll, resolution=self.resolution)
        if len(sampled) != len(test_points):
            print(
                "Warning: length of sampled dataset ({}) != test points ({})".format(
                    len(sampled), len(test_points)
                )
            )

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

    def get_binary_image(self, aoi, projection="EPSG:4326"):
        """
        Creates a binary image for a covermap in given country.
        """
        image = self.ee_asset.mosaic().clip(aoi).reproject(crs=projection, scale=self.resolution)

        # Convert the image to binary classes: 1 (crop), 0 (noncrop)
        if self.crop_labels is None:
            binary_image = image.gte(self.probability)

        # Check if crop_labels are an ordered list
        elif len(self.crop_labels) > 2 and self.crop_labels == list(
            range(self.crop_labels[0], self.crop_labels[-1] + 1)
        ):
            binary_image = image.gte(self.crop_labels[0]).And(image.lte(self.crop_labels[-1]))
        else:
            binary_image = image.eq(self.crop_labels[0])
            if len(self.crop_labels) > 1:
                for crop_value in self.crop_labels[1:]:
                    binary_image = binary_image.Or(image.eq(crop_value))

        return binary_image.rename("crop")

    def compute_map_area(
        self, country: str, dataset_name, projection="EPSG:4326", tile_grid=[1, 1], export=False
    ):
        aoi = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
            ee.Filter.eq("ADM0_NAME", country)
        )

        binary_image = self.get_binary_image(aoi=aoi, projection=projection)

        if tile_grid == [1, 1]:
            crop_px_sum = binary_image.reduceRegion(
                reducer=ee.Reducer.sum().unweighted(),
                geometry=aoi.geometry(),
                scale=self.resolution,
                maxPixels=MAX_PIXELS,
                bestEffort=True,
            ).get("crop")
            noncrop_px_sum = (
                binary_image.Not()
                .reduceRegion(
                    reducer=ee.Reducer.sum().unweighted(),
                    geometry=aoi.geometry(),
                    scale=self.resolution,
                    maxPixels=MAX_PIXELS,
                    bestEffort=True,
                )
                .get("crop")
            )

            if export:
                # export the computation and retrieve result later
                export_task = ee.batch.Export.table.toDrive(
                    collection=ee.FeatureCollection(
                        [ee.Feature(None, {"crop_sum": crop_px_sum, "noncrop_sum": noncrop_px_sum})]
                    ),
                    description=f"Crop_NonCrop_Area_Sum_Export-{country}-{dataset_name}",
                    fileFormat="CSV",
                )
                export_task.start()
                print(f"Export task started for {dataset_name}, {country}. Returning null for now.")
                a_j = np.array([None, None])

            else:
                # do computation in this client session
                a_j = np.array([noncrop_px_sum.getInfo(), crop_px_sum.getInfo()])

        else:
            tile_geometries = create_tile_geometries(aoi)
            # Initialize sums
            crop_sum_total = 0
            noncrop_sum_total = 0
            # Iterate over each tile
            for tile_geometry in tile_geometries:
                crop_sum = (
                    compute_tile_sum(binary_image, tile_geometry, scale=self.resolution)
                    .get("crop")
                    .getInfo()
                )
                noncrop_sum = (
                    compute_tile_sum(binary_image.Not(), tile_geometry, scale=self.resolution)
                    .get("crop")
                    .getInfo()
                )
                # Update total sums
                crop_sum_total += crop_sum
                noncrop_sum_total += noncrop_sum

            a_j = np.array([noncrop_sum_total, crop_sum_total])

        return a_j


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

    def get_test_years(self):
        """
        Returns a dictionary containing the test years for each test file.
        """
        test_years = {}
        for country in TEST_COUNTRIES.keys():
            match = re.search(r"\d{4}", TEST_COUNTRIES[country])
            if match:
                test_years[country] = match.group(0)
            else:
                test_years[country] = None
        return test_years

    def extract_covermaps(self, test_df):
        """
        Groups testing points by country then extracts from each map. If map does not include
        specified country, the map is skipped. Resulting extracted points are combined for
        each country then added to sampled_maps, where key = country and value = a dataframe of
        extracted points from maps.
        """
        test_years = self.get_test_years()

        for country in self.test_countries:
            country_test = test_df.loc[test_df[COUNTRY_COL] == country].copy()

            assert not country_test.is_empty.all(), "No testing points for " + country

            for map in self.covermaps:
                if country in map.countries:
                    print(f"[{country}] sampling " + map.title + "...")

                    map_sampled = map.extract_test(country_test, test_years[country]).copy()
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


def compute_tile_sum(binary_image, tile_geometry, scale):
    """
    Compute the sum of binary values within a tile.
    """
    sum_dict = binary_image.reduceRegion(
        reducer=ee.Reducer.sum().unweighted(),
        geometry=tile_geometry,
        scale=scale,
        maxPixels=MAX_PIXELS,
    )
    return sum_dict


def create_tile_geometries(aoi, rows=10, columns=10):
    """
    Generate a list of tile geometries as a grid over the specified AOI,
    where the AOI is given as an ee.FeatureCollection.

    Parameters:
    aoi (ee.FeatureCollection): The area of interest as a feature collection.
    rows (int): The number of rows in the grid.
    columns (int): The number of columns in the grid.

    Returns:
    list of ee.Geometry: The geometries representing each tile in the grid.
    """
    # Convert the FeatureCollection to a Geometry that represents the union of all features
    aoi_geometry = aoi.geometry()

    # Get the bounds of the AOI geometry
    bounds = aoi_geometry.bounds()
    boundsInfo = bounds.getInfo()  # Get information about the bounds
    coords = boundsInfo["coordinates"][0]  # Extract the coordinates of the bounds

    # Extract the min and max coordinates
    x_min = coords[0][0]
    y_min = coords[0][1]
    x_max = coords[2][0]
    y_max = coords[2][1]

    # Calculate the width and height of each tile
    width = (x_max - x_min) / columns
    height = (y_max - y_min) / rows

    tile_geometries = []

    # Create the grid of tiles
    for i in range(columns):
        for j in range(rows):
            x1 = x_min + width * i
            y1 = y_min + height * j
            x2 = x1 + width
            y2 = y1 + height
            tile = ee.Geometry.Rectangle([x1, y1, x2, y2], "EPSG:4326", False)
            tile_geometries.append(tile)

    return tile_geometries


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


def extract_points(
    ic: ee.ImageCollection, fc: ee.FeatureCollection, resolution=10, projection="EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Creates geodataframe of extracted values from image collection. Assumes ic parameters are set
    (date, region, band, etc.).
    """
    image = ic.filterBounds(fc).mosaic().reproject(crs=projection, scale=resolution)
    extracted = image.reduceRegions(
        collection=fc, scale=resolution, crs=projection, reducer=REDUCER
    )
    return geemap.ee_to_gdf(extracted)


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
    test = pd.read_csv(path)[["lat", "lon", "class_probability", "subset"]]
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


def compute_f1(precision, recall):
    """
    Calculate the F1 score from precision and recall.
    """
    if precision + recall == 0:
        return 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def compute_std_f1(label, recall_i, precision_i, std_rec_i, std_prec_i):
    """
    Propagates standard deviation of precision and recall to get
    standard deviation of F1 score.
    """
    r = recall_i[label]
    p = precision_i[label]
    dr = std_rec_i[label]
    dp = std_prec_i[label]
    expr1 = 2 * (r * dp + p * dr) / (p + r)
    expr2 = ((2 * p * r) * (dp + dr)) / ((p + r) * (p + r))
    df = expr1 + expr2
    return df


def get_ensemble_area(country: str, covermaps, tile_grid=[1, 1], export=False):
    """
    Creates ensemble image and calculates areas.
    """
    # Create the binary map version of each map
    aoi = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq("ADM0_NAME", country))

    binary_images = []
    for covermap in covermaps:
        binary_images.append(covermap.get_binary_image(aoi))

    # Create an image collection of the maps and sum
    binary_coll = ee.ImageCollection(binary_images)
    sum_image = binary_coll.reduce(ee.Reducer.sum()).clip(aoi)

    # Threshold to binary based on majority vote
    majority_thresh = np.ceil(len(binary_images) / 2)
    ensemble_image = sum_image.gte(majority_thresh).rename("crop")

    # Calculate the total pixels in each class
    min_scale = min([c.resolution for c in covermaps])

    if tile_grid == [1, 1]:
        crop_px_sum = ensemble_image.reduceRegion(
            reducer=ee.Reducer.sum().unweighted(),
            geometry=aoi.geometry(),
            scale=min_scale,
            maxPixels=MAX_PIXELS,
            bestEffort=True,
        ).get("crop")
        noncrop_px_sum = (
            ensemble_image.Not()
            .reduceRegion(
                reducer=ee.Reducer.sum().unweighted(),
                geometry=aoi.geometry(),
                scale=min_scale,
                maxPixels=MAX_PIXELS,
                bestEffort=True,
            )
            .get("crop")
        )

        if export:
            # export the computation and retrieve result later
            export_task = ee.batch.Export.table.toDrive(
                collection=ee.FeatureCollection(
                    [ee.Feature(None, {"crop_sum": crop_px_sum, "noncrop_sum": noncrop_px_sum})]
                ),
                description=f"Crop_NonCrop_Ensemble_Area_Sum_Export-{country}",
                fileFormat="CSV",
            )
            export_task.start()
            print(
                f"Export task started for ensemble map of {country}. Returning null area for now."
            )
            a_j = np.array([None, None])

        else:
            # do computation in this client session
            a_j = np.array([noncrop_px_sum.getInfo(), crop_px_sum.getInfo()])

    else:
        tile_geometries = create_tile_geometries(aoi)

        # Initialize sums
        crop_sum_total = 0
        noncrop_sum_total = 0

        # Iterate over each tile
        for tile_geometry in tile_geometries:
            crop_sum = (
                compute_tile_sum(ensemble_image, tile_geometry, scale=min_scale)
                .get("crop")
                .getInfo()
            )
            noncrop_sum = (
                compute_tile_sum(ensemble_image.Not(), tile_geometry, scale=min_scale)
                .get("crop")
                .getInfo()
            )

            # Update total sums
            crop_sum_total += crop_sum
            noncrop_sum_total += noncrop_sum

        a_j = np.array([noncrop_sum_total, crop_sum_total])

    return a_j


def generate_report(
    dataset_name: str, country: str, true, pred, a_j, area_weighted
) -> pd.DataFrame:
    """
    Creates classification report on crop coverage binary values.
    Assumes 1 indicates cropland.
    """
    cm = confusion_matrix(true, pred)
    tn, fp, fn, tp = cm.ravel()
    total_px = a_j.sum()
    w_j = a_j / total_px
    if area_weighted:
        # weight by mapped area for each class
        am = compute_area_error_matrix(cm, w_j)
    else:
        # unweighted metrics
        w_j = np.ones_like(w_j)
        am = cm / (tn + fp + fn + tp)
    tn_area, fp_area, fn_area, tp_area = am.ravel()
    u_j = compute_u_j(am)
    p_i = compute_p_i(am)
    return pd.DataFrame(
        data={
            "dataset": dataset_name,
            "country": country,
            "crop_f1": compute_f1(precision=u_j[1], recall=p_i[1]),
            "std_crop_f1": compute_std_f1(
                label=1,
                recall_i=p_i,
                precision_i=u_j,
                std_rec_i=np.sqrt(compute_var_p_i(p_i=p_i, u_j=u_j, a_j=a_j, cm=cm)),
                std_prec_i=np.sqrt(compute_var_u_j(u_j=u_j, cm=cm)),
            ),
            "accuracy": compute_acc(am),
            "std_acc": np.sqrt(compute_var_acc(w_j=w_j, u_j=u_j, cm=cm)),
            "crop_recall_pa": p_i[1],
            "std_crop_pa": np.sqrt(compute_var_p_i(p_i=p_i, u_j=u_j, a_j=a_j, cm=cm))[1],
            "noncrop_recall_pa": p_i[0],
            "std_noncrop_pa": np.sqrt(compute_var_p_i(p_i=p_i, u_j=u_j, a_j=a_j, cm=cm))[0],
            "crop_precision_ua": u_j[1],
            "std_crop_ua": np.sqrt(compute_var_u_j(u_j=u_j, cm=cm))[1],
            "noncrop_precision_ua": u_j[0],
            "std_noncrop_ua": np.sqrt(compute_var_u_j(u_j=u_j, cm=cm))[0],
            "crop_support": tp + fn,
            "noncrop_support": tn + fp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "tn_area": tn_area,
            "fp_area": fp_area,
            "fn_area": fn_area,
            "tp_area": tp_area,
        },
        index=[0],
    ).round(2)


# Maps
TARGETS = {
    "copernicus": Covermap(
        "copernicus",
        """ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global")
        .select("discrete_classification")
        .filterDate("2019-01-01", "2019-12-31")""",
        resolution=100,
        crop_labels=[40],
        years_covered=[2015, 2016, 2017, 2018, 2019],
    ),
    "worldcover-v100": Covermap(
        "worldcover-v100",
        'ee.ImageCollection("ESA/WorldCover/v100")',
        resolution=10,
        crop_labels=[40],
        years_covered=[2020],
    ),
    "worldcover-v200": Covermap(
        "worldcover-v200",
        'ee.ImageCollection("ESA/WorldCover/v200")',
        resolution=10,
        crop_labels=[40],
        years_covered=[2021],
    ),
    "worldcereal-v100": Covermap(
        "worldcereal-v100",
        """ee.ImageCollection(
            ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
            .filter('product == "temporarycrops"')
            .select("classification")
            .mosaic()
        )""",
        resolution=10,
        crop_labels=[100],
        years_covered=[2020],
    ),
    "glad": Covermap(
        "glad",
        'ee.ImageCollection("users/potapovpeter/Global_cropland_2019")',
        resolution=30,
        probability=0.5,
        years_covered=[2019],
    ),
    # "gfsad": Covermap(
    #     "gfsad",
    #     ee.ImageCollection(ee.Image("USGS/GFSAD1000_V1")),
    #     resolution=1000,
    #     crop_labels=[1, 2, 3, 4, 5],
    #     years_covered=[2010],
    # ),
    "asap": Covermap(
        "asap",
        'ee.ImageCollection(ee.Image("users/sbaber/asap_mask_crop_v03").unmask())',
        resolution=1000,
        probability=100,
        countries=[country for country in TEST_COUNTRIES.keys() if country != "Hawaii"],
        years_covered=[2017],
    ),
    "dynamicworld": Covermap(
        "dynamicworld",
        """ee.ImageCollection(
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filter(ee.Filter.date("2019-01-01", "2019-12-31"))
            .select(["label"])
            .mode()
        )""",
        resolution=10,
        crop_labels=[4],
        years_covered=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    ),
    "gfsad-gcep": Covermap(
        "gfsad-gcep",
        'ee.ImageCollection("projects/sat-io/open-datasets/GFSAD/GCEP30")',
        resolution=30,
        crop_labels=[2],
        years_covered=[2015],
    ),
    "gfsad-lgrip": Covermap(
        "gfsad-lgrip",
        'ee.ImageCollection("projects/sat-io/open-datasets/GFSAD/LGRIP30")',
        resolution=30,
        crop_labels=[2, 3],
        years_covered=[2015],
    ),
    "digital-earth-africa": Covermap(
        "digital-earth-africa",
        """ee.ImageCollection([ee.Image(0)
            .where(ee.ImageCollection("projects/sat-io/open-datasets/DEAF/CROPLAND-EXTENT/filtered")
            .mosaic()
            .eq(1), 1)]
        )""",
        resolution=10,
        crop_labels=[1],
        countries=[country for country in TEST_COUNTRIES.keys() if country != "Hawaii"],
        years_covered=[2019],
    ),
    "esa-cci-africa": Covermap(
        "esa-cci-africa",
        """ee.ImageCollection(
            ee.Image("projects/sat-io/open-datasets/ESA/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v10")
        )""",
        resolution=20,
        crop_labels=[4],
        countries=[country for country in TEST_COUNTRIES.keys() if country != "Hawaii"],
        years_covered=[2016],
    ),
    "globcover-v23": Covermap(
        "globcover-v23",
        """ee.ImageCollection(
            ee.Image("projects/sat-io/open-datasets/ESA/GLOBCOVER_L4_200901_200912_V23")
        )""",
        resolution=300,
        crop_labels=[11, 14, 20, 30],
        years_covered=[2009],
    ),
    "globcover-v22": Covermap(
        "globcover-v22",
        """ee.ImageCollection(
            ee.Image("projects/sat-io/open-datasets/ESA/GLOBCOVER_200412_200606_V22_Global_CLA")
        )""",
        resolution=300,
        crop_labels=[11, 14, 20, 30],
        years_covered=[2005],
    ),
    "esri-lulc": Covermap(
        "esri-lulc",
        """ee.ImageCollection(
            "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
        ).filter(ee.Filter.date("2019-01-01", "2019-12-31"))""",
        resolution=10,
        crop_labels=[5],
        years_covered=[2017, 2018, 2019, 2020, 2021, 2022],
    ),
    "nabil-etal-2021": Covermap(
        "nabil-etal-2021",
        """ee.ImageCollection.fromImages(
            [ee.Image("projects/sat-io/open-datasets/landcover/AF_Cropland_mask_30m_2016_v3")]
        )""",
        resolution=30,
        crop_labels=[2],
        countries=[country for country in TEST_COUNTRIES.keys() if country != "Hawaii"],
        years_covered=[2016],
    ),
    "harvest-crop-maps": Covermap(
        "harvest-crop-maps",
        'ee.ImageCollection("projects/bsos-geog-harvest1/assets/harvest-crop-maps")',
        resolution=10,
        probability=0.5,
        countries=["Togo", "Kenya", "Malawi"],
    ),
    "harvest-dev": Covermap(
        "harvest-dev",
        """ee.ImageCollection.fromImages(
            [
                ee.Image("users/abaansah/Namibia_North_2020_V3"),
                ee.Image("users/adadebay/Zambia_cropland_2019"),
                ee.Image("users/izvonkov/Hawaii_skip_era5_v4"),
                ee.Image(
                    "users/adadebay/Uganda_2019_skip_ERA5_min_lat--1-63"
                    "_min_lon-29-3_max_lat-4-3_max_lon-35-17_dates-2019-02-01_2020-02-"
                ),
                ee.Image("users/abaansah/Sudan_Al_Gadaref_2020_Feb"),
                ee.Image("users/abaansah/Sudan_Al_Gadaref_2019_Feb"),
                ee.Image("users/izvonkov/Sudan_Blue_Nile_2020"),
                ee.Image("users/izvonkov/Sudan_Blue_Nile_2019_v3"),
                ee.Image("users/izvonkov/Ethiopia_Tigray_2020_threshold-3-5"),
                ee.Image("users/izvonkov/Ethiopia_Tigray_2021_threshold-3-5"),
                ee.Image("users/adadebay/Tanzania_cropland_2019"),
                ee.Image("users/eutzschn/Ethiopia_Bure_Jimma_2020_v1"),
                ee.Image("users/izvonkov/Ethiopia_Bure_Jimma_2019_v1"),
                ee.Image(
                    "users/izvonkov/Rwanda_2019_skip_era5_min_lat--3"
                    "-035_min_lon-28-43_max_lat--0-76_max_lon-31-013"
                    "_dates-2019-02-01_202"
                )
            ]
        )""",
        resolution=10,
        probability=0.5,
        countries=[
            "Tanzania",
            "Namibia",
            "Uganda",
            "Zambia",
            "Hawaii",
            "BlueNile2020",
            "BlueNile2019",
            "AlGadaref2019",
            "BureJimma2019",
            "BureJimma2020",
            "Tigray2021",
            "Tigray2020",
            "Rwanda",
        ],
    ),
}
