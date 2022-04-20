from cropharvest.eo import EarthEngineExporter
from cropharvest.eo.eo import get_cloud_tif_list
from cropharvest.engineer import Engineer
from datetime import datetime
from dataclasses import dataclass
from google.cloud import storage
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
import tempfile

from .processor import Processor
from src.utils import (
    data_dir,
    features_dir,
    memoize,
    distance,
    distance_point_from_center,
    find_nearest,
)
from src.ETL.boundingbox import BoundingBox
from src.ETL.constants import (
    ALREADY_EXISTS,
    COUNTRY,
    CROP_PROB,
    CROP_TYPE,
    FEATURE_FILENAME,
    FEATURE_PATH,
    LAT,
    LON,
    START,
    END,
    SOURCE,
    NUM_LABELERS,
    LABELER_NAMES,
    LABEL_DUR,
    SUBSET,
    DATASET,
    TIF_PATHS,
    TIF_BUCKET,
)
from src.ETL.data_instance import CropDataInstance

unexported_file = data_dir / "unexported.txt"
unexported = pd.read_csv(unexported_file, sep="\n", header=None)[0].tolist()

missing_data_file = data_dir / "missing_data.txt"
missing_data = pd.read_csv(missing_data_file, sep="\n", header=None)[0].tolist()

duplicates_data_file = data_dir / "duplicates.txt"
duplicates_data = pd.read_csv(duplicates_data_file, sep="\n", header=None)[0].tolist()

bucket = storage.Client().bucket(TIF_BUCKET)
temp_dir = tempfile.gettempdir()


@memoize
def generate_bbox_from_paths() -> Dict[Path, BoundingBox]:
    cloud_tif_paths = [Path(p) for p in get_cloud_tif_list(TIF_BUCKET)]
    return {
        p: BoundingBox.from_path(p)
        for p in tqdm(cloud_tif_paths, desc="Generating BoundingBoxes from paths")
    }


def get_tif_paths(path_to_bbox, lat, lon, start_date, end_date, pbar):
    candidate_paths = []
    for p, bbox in path_to_bbox.items():
        if bbox.contains(lat, lon) and f"dates={start_date}_{end_date}" in p.stem:
            candidate_paths.append(p)
    pbar.update(1)
    return candidate_paths


def match_labels_to_tifs(labels: pd.DataFrame) -> pd.Series:
    bbox_for_labels = BoundingBox(
        min_lon=labels[LON].min(),
        min_lat=labels[LAT].min(),
        max_lon=labels[LON].max(),
        max_lat=labels[LAT].max(),
    )
    # Get all tif paths and bboxes
    path_to_bbox = {
        p: bbox for p, bbox in generate_bbox_from_paths().items() if bbox_for_labels.overlaps(bbox)
    }

    # Match labels to tif files
    # Faster than going through bboxes
    with tqdm(total=len(labels), desc="Matching labels to tif paths") as pbar:
        tif_paths = np.vectorize(get_tif_paths, otypes=[np.ndarray])(
            path_to_bbox,
            labels[LAT],
            labels[LON],
            labels[START],
            labels[END],
            pbar,
        )
    return tif_paths


def find_matching_point(
    start: str, tif_paths: List[Path], label_lon: float, label_lat: float
) -> Tuple[np.ndarray, float, float, str]:
    """
    Given a label coordinate (y) this functions finds the associated satellite data (X)
    by looking through one or multiple tif files.
    Each tif file contains satellite data for a grid of coordinates.
    So the function finds the closest grid coordinate to the label coordinate.
    Additional value is given to a grid coordinate that is close to the center of the tif.
    """
    start_date = datetime.strptime(start, "%Y-%m-%d")
    tif_slope_tuples = []
    for p in tif_paths:
        blob = bucket.blob(str(p))
        local_path = f"{temp_dir}/{p.name}"
        blob.download_to_filename(local_path)
        tif_slope_tuples.append(
            Engineer.load_tif(local_path, start_date=start_date, num_timesteps=None)
        )
        Path(local_path).unlink()

    if len(tif_slope_tuples) > 1:
        min_distance_from_point = np.inf
        min_distance_from_center = np.inf
        for i, tif_slope_tuple in enumerate(tif_slope_tuples):
            tif, slope = tif_slope_tuple
            lon, lon_idx = find_nearest(tif.x, label_lon)
            lat, lat_idx = find_nearest(tif.y, label_lat)
            distance_from_point = distance(label_lat, label_lon, lat, lon)
            distance_from_center = distance_point_from_center(lat_idx, lon_idx, tif)
            if (distance_from_point < min_distance_from_point) or (
                distance_from_point == min_distance_from_point
                and distance_from_center < min_distance_from_center
            ):
                closest_lon = lon
                closest_lat = lat
                min_distance_from_center = distance_from_center
                min_distance_from_point = distance_from_point
                labelled_np = tif.sel(x=lon).sel(y=lat).values
                average_slope = slope
                source_file = tif_paths[i].name
    else:
        tif, slope = tif_slope_tuples[0]
        closest_lon = find_nearest(tif.x, label_lon)[0]
        closest_lat = find_nearest(tif.y, label_lat)[0]
        labelled_np = tif.sel(x=closest_lon).sel(y=closest_lat).values
        average_slope = slope
        source_file = tif_paths[0].name

    labelled_np = Engineer.calculate_ndvi(labelled_np)
    labelled_np = Engineer.remove_bands(labelled_np)
    labelled_np = Engineer.fillna(labelled_np, average_slope)

    return labelled_np, closest_lon, closest_lat, source_file


def create_pickled_labeled_dataset(labels):
    for label in tqdm(labels.to_dict(orient="records"), desc="Creating pickled instances"):
        (labelled_array, tif_lon, tif_lat, tif_file) = find_matching_point(
            start=label[START],
            tif_paths=label[TIF_PATHS],
            label_lon=label[LON],
            label_lat=label[LAT],
        )

        if labelled_array is None:
            with open(missing_data_file, "a") as f:
                f.write("\n" + label[FEATURE_FILENAME])
            continue

        instance = CropDataInstance(
            labelled_array=labelled_array,
            instance_lat=tif_lat,
            instance_lon=tif_lon,
            source_file=tif_file,
        )
        save_path = Path(label[FEATURE_PATH])
        save_path.parent.mkdir(exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(instance, f)


def get_label_timesteps(labels):
    diff = pd.to_datetime(labels[END]) - pd.to_datetime(labels[START])
    return (diff / np.timedelta64(1, "M")).round().astype(int)


def load_all_features_as_df() -> pd.DataFrame:
    features = []
    files = list(features_dir.glob("*.pkl"))
    print("------------------------------")
    print("Loading all features...")
    non_duplicated_files = []
    for p in files:
        if p.stem not in duplicates_data:
            non_duplicated_files.append(p)
            with p.open("rb") as f:
                features.append(pickle.load(f))
    df = pd.DataFrame([feat.__dict__ for feat in features])
    df["filename"] = non_duplicated_files
    return df


@dataclass
class LabeledDataset:
    dataset: str = ""
    country: str = ""

    # Process parameters
    processors: Tuple[Processor, ...] = ()

    def __post_init__(self):
        self.raw_labels_dir = data_dir / "raw" / self.dataset
        self.labels_path = data_dir / "processed" / (self.dataset + ".csv")
        self._cached_labels_csv = None

    def summary(self, df=None):
        if df is None:
            df = self.load_labels(allow_processing=False, fail_if_missing_features=False)
        text = f"{self.dataset} "
        timesteps = get_label_timesteps(df).unique()
        text += f"(Timesteps: {','.join([str(int(t)) for t in timesteps])})\n"
        text += "----------------------------------------------------------------------------\n"
        train_val_test_counts = df[SUBSET].value_counts()
        for subset, labels_in_subset in train_val_test_counts.items():
            features_in_subset = df[df[SUBSET] == subset][ALREADY_EXISTS].sum()
            if labels_in_subset != features_in_subset:
                text += (
                    f"\u2716 {subset}: {labels_in_subset} labels, "
                    + f"but {features_in_subset} features\n"
                )
            else:
                crop_percentage = (
                    df[df[SUBSET] == subset][CROP_PROB] > 0.5
                ).sum() / labels_in_subset
                text += f"\u2714 {subset} amount: {labels_in_subset}, crop: {crop_percentage:.1%}\n"

        return text

    def process_labels(self):
        df = pd.DataFrame({})
        already_processed = []
        if self.labels_path.exists():
            df = pd.read_csv(self.labels_path)
            already_processed = df[SOURCE].unique()

        # Go through processors and create new labels if necessary
        new_labels = [
            p.process(self.raw_labels_dir)
            for p in self.processors
            if p.filename not in str(already_processed)
        ]

        if len(new_labels) == 0:
            return df

        df = pd.concat([df] + new_labels)

        # Combine duplicate labels
        df[NUM_LABELERS] = 1
        df = df.groupby([LON, LAT, START, END], as_index=False, sort=False).agg(
            {
                SOURCE: lambda sources: ",".join(sources.unique()),
                CROP_PROB: "mean",
                NUM_LABELERS: "sum",
                SUBSET: "first",
                CROP_TYPE: "first",
                LABEL_DUR: lambda dur: ",".join(dur),
                LABELER_NAMES: lambda name: ",".join(name),
            }
        )
        df[COUNTRY] = self.country
        df[DATASET] = self.dataset

        df[FEATURE_FILENAME] = (
            "lat="
            + df[LAT].round(8).astype(str)
            + "_lon="
            + df[LON].round(8).astype(str)
            + "_date="
            + df[START].astype(str)
            + "_"
            + df[END].astype(str)
        )

        df = df.reset_index(drop=True)
        df.to_csv(self.labels_path, index=False)
        return df

    def load_labels(
        self, allow_processing: bool = False, fail_if_missing_features: bool = False
    ) -> pd.DataFrame:
        if allow_processing:
            labels = self.process_labels()
            self._cached_labels_csv = labels
        elif self._cached_labels_csv is not None:
            labels = self._cached_labels_csv
        elif self.labels_path.exists():
            labels = pd.read_csv(self.labels_path)
            self._cached_labels_csv = labels
        else:
            raise FileNotFoundError(f"{self.labels_path} does not exist")
        labels = labels[labels[CROP_PROB] != 0.5]
        unexported_labels = labels[FEATURE_FILENAME].isin(unexported)
        missing_data_labels = labels[FEATURE_FILENAME].isin(missing_data)
        duplicate_labels = labels[FEATURE_FILENAME].isin(duplicates_data)
        labels = labels[~unexported_labels & ~missing_data_labels & ~duplicate_labels].copy()
        labels["feature_dir"] = str(features_dir)
        labels[FEATURE_PATH] = labels["feature_dir"] + "/" + labels[FEATURE_FILENAME] + ".pkl"
        labels[ALREADY_EXISTS] = np.vectorize(lambda p: Path(p).exists())(labels[FEATURE_PATH])
        if fail_if_missing_features and not labels[ALREADY_EXISTS].all():
            raise FileNotFoundError(
                f"{self.dataset} has missing features: {labels[FEATURE_FILENAME].to_list()}"
            )
        return labels

    def create_features(self, disable_gee_export: bool = False) -> str:
        """
        Features are the (X, y) pairs that are used to train the model.
        In this case,
        - X is the satellite data for a lat lon coordinate over a 12 month time series
        - y is the crop/non-crop label for that coordinate

        To create the features:
        1. Obtain the labels
        2. Check if the features already exist
        3. Use the label coordinates to match to the associated satellite data (X)
        4. If the satellite data is missing, download it using Google Earth Engine
        5. Create the features (X, y)
        """
        print("------------------------------")
        print(self.dataset)

        # -------------------------------------------------
        # STEP 1: Obtain the labels
        # -------------------------------------------------
        labels = self.load_labels(allow_processing=True)

        # -------------------------------------------------
        # STEP 2: Check if features already exist
        # -------------------------------------------------
        labels_with_no_features = labels[~labels[ALREADY_EXISTS]].copy()
        if len(labels_with_no_features) == 0:
            return self.summary(labels)

        # -------------------------------------------------
        # STEP 3: Match labels to tif files (X)
        # -------------------------------------------------
        labels_with_no_features[TIF_PATHS] = match_labels_to_tifs(labels_with_no_features)
        tifs_found = labels_with_no_features[TIF_PATHS].str.len() > 0

        labels_with_no_tifs = labels_with_no_features.loc[~tifs_found].copy()
        labels_with_tifs_but_no_features = labels_with_no_features.loc[tifs_found]

        # -------------------------------------------------
        # STEP 4: If no matching tif, download it
        # -------------------------------------------------
        if len(labels_with_no_tifs) > 0:
            print(f"{len(labels_with_no_tifs )} labels not matched")
            if not disable_gee_export:
                labels_with_no_tifs[START] = pd.to_datetime(labels_with_no_tifs[START]).dt.date
                labels_with_no_tifs[END] = pd.to_datetime(labels_with_no_tifs[END]).dt.date
                EarthEngineExporter(
                    check_ee=True,
                    check_gcp=True,
                    dest_bucket=TIF_BUCKET,
                ).export_for_labels(labels=labels_with_no_tifs)

        # -------------------------------------------------
        # STEP 5: Create the features (X, y)
        # -------------------------------------------------
        if len(labels_with_tifs_but_no_features) > 0:
            create_pickled_labeled_dataset(labels=labels_with_tifs_but_no_features)
            labels[ALREADY_EXISTS] = np.vectorize(lambda p: Path(p).exists())(labels[FEATURE_PATH])
        return self.summary(labels)
