from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm
import logging
import pandas as pd
import numpy as np

from .engineer import Engineer
from .processor import Processor
from .ee_exporter import LabelExporter, RegionExporter, Season
from src.utils import get_data_dir, get_tifs_dir, memoize
from src.ETL.ee_boundingbox import BoundingBox
from src.ETL.constants import (
    ALREADY_EXISTS,
    COUNTRY,
    CROP_PROB,
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
)

logger = logging.getLogger(__name__)

default_data_folder: Path = get_data_dir()

unexported_file = default_data_folder / "unexported.txt"
unexported = pd.read_csv(unexported_file, sep="\n", header=None)[0].tolist()


@memoize
def generate_bbox_from_paths() -> Dict[Path, BoundingBox]:
    return {p: BoundingBox.from_path(p) for p in tqdm(get_tifs_dir().glob("**/*.tif"))}


class DataDir(Enum):
    RAW_DIR = "raw_dir"
    RAW_LABELS_DIR = "raw_labels_dir"
    LABELS_PATH = "labels_path"
    FEATURES_DIR = "features_dir"


@dataclass
class LabeledDataset:
    dataset: str = ""
    country: str = ""

    # Process parameters
    processors: Tuple[Processor, ...] = ()
    days_per_timestep: int = 30

    def __post_init__(self):
        self.raw_labels_dir = self.get_path(DataDir.RAW_LABELS_DIR)
        self.labels_path = self.get_path(DataDir.LABELS_PATH)
        self.feature_dir = self.get_path(DataDir.FEATURES_DIR)
        self.feature_dir.mkdir(exist_ok=True, parents=True)

    def get_path(self, data_dir: DataDir, root_data_folder: Path = default_data_folder):
        if data_dir == DataDir.RAW_DIR:
            return root_data_folder / "raw"
        if data_dir == DataDir.RAW_LABELS_DIR:
            return root_data_folder / "raw" / self.dataset
        if data_dir == DataDir.LABELS_PATH:
            return root_data_folder / "processed" / (self.dataset + ".csv")
        if data_dir == DataDir.FEATURES_DIR:
            return root_data_folder / "features" / self.dataset

    @staticmethod
    def merge_sources(sources):
        return ",".join(sources.unique())

    def process_labels(self):
        df = pd.DataFrame({})
        already_processed = []
        if self.labels_path.exists():
            df = pd.read_csv(self.labels_path)
            already_processed = df[SOURCE].unique()

        # Go through processors and create new labels if necessary
        new_labels = [
            p.process(self.raw_labels_dir, self.days_per_timestep)
            for p in self.processors
            if p.filename not in str(already_processed)
        ]

        if len(new_labels) == 0:
            return df

        df = pd.concat([df] + new_labels)

        # Combine duplicate labels
        df[NUM_LABELERS] = 1
        df = df.groupby([LON, LAT, START, END], as_index=False, sort=False).agg(
            {SOURCE: self.merge_sources, CROP_PROB: "mean", NUM_LABELERS: "sum", SUBSET: "first",
            LABEL_DUR: "mean"}
        )
        df[COUNTRY] = self.country
        df[DATASET] = self.dataset

        df["filename"] = (
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

    @staticmethod
    def get_tif_paths(path_to_bbox, lat, lon, start_date, end_date, pbar):
        candidate_paths = []
        for p, bbox in path_to_bbox.items():
            if bbox.contains(lat, lon) and p.stem.endswith(f"dates={start_date}_{end_date}"):
                candidate_paths.append(p)
        pbar.update(1)
        return candidate_paths

    def do_label_and_feature_amounts_match(self, labels: pd.DataFrame):
        train_val_test_counts = labels[SUBSET].value_counts()
        for subset, labels_in_subset in train_val_test_counts.items():
            features_in_subset = len(list((self.feature_dir / subset).glob("*.pkl")))
            if labels_in_subset != features_in_subset:
                print(
                    f"\u2716 {subset}: {labels_in_subset} labels, but {features_in_subset} features"
                )
            else:
                print(f"\u2714 {subset} amount: {labels_in_subset}")

    def prune_features_with_no_label(self, features_with_label: List[str]):
        for f in list(self.feature_dir.glob("**/*.pkl")):
            if str(f) not in features_with_label:
                f.unlink()

    def generate_feature_paths(self, labels: pd.DataFrame) -> pd.Series:
        labels["feature_dir"] = str(self.feature_dir)
        return labels["feature_dir"] + "/" + labels[SUBSET] + "/" + labels["filename"] + ".pkl"

    def match_labels_to_tifs(self, labels: pd.DataFrame) -> pd.Series:
        bbox_for_labels = BoundingBox(
            min_lon=labels[LON].min(),
            min_lat=labels[LAT].min(),
            max_lon=labels[LON].max(),
            max_lat=labels[LAT].max(),
        )
        # Get all tif paths and bboxes
        path_to_bbox = {
            p: bbox
            for p, bbox in generate_bbox_from_paths().items()
            if bbox_for_labels.overlaps(bbox)
        }

        # Match labels to tif files
        # Faster than going through bboxes
        with tqdm(total=len(labels), desc="Matching labels to tif paths") as pbar:
            tif_paths = np.vectorize(self.get_tif_paths, otypes=[np.ndarray])(
                path_to_bbox,
                labels[LAT],
                labels[LON],
                labels[START],
                labels[END],
                pbar,
            )
        return tif_paths

    def create_features(self, disable_gee_export: bool = False):
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
        labels = self.process_labels()

        # set aside conflicting labels that are eliminated
        eliminated = labels[labels[CROP_PROB] == 0.5]
        print("eliminated amount: "+ str(len(eliminated)))

        labels = labels[labels[CROP_PROB] != 0.5]        
        labels = labels[~labels["filename"].isin(unexported)]
        labels[FEATURE_PATH] = self.generate_feature_paths(labels)

        # -------------------------------------------------
        # STEP 2: Check if features already exist
        # -------------------------------------------------
        self.prune_features_with_no_label(labels[FEATURE_PATH].to_list())
        labels[ALREADY_EXISTS] = np.vectorize(lambda p: Path(p).exists())(labels[FEATURE_PATH])
        labels_with_no_features = labels[~labels[ALREADY_EXISTS]].copy()
        if len(labels_with_no_features) == 0:
            self.do_label_and_feature_amounts_match(labels)
            return

        # -------------------------------------------------
        # STEP 3: Match labels to tif files (X)
        # -------------------------------------------------
        labels_with_no_features[TIF_PATHS] = self.match_labels_to_tifs(labels_with_no_features)
        tifs_found = labels_with_no_features[TIF_PATHS].str.len() > 0

        labels_with_no_tifs = labels_with_no_features.loc[~tifs_found]
        labels_with_tifs_but_no_features = labels_with_no_features.loc[tifs_found]

        # -------------------------------------------------
        # STEP 4: If no matching tif, download it
        # -------------------------------------------------
        if len(labels_with_no_tifs) > 0:
            print(f"{len(labels_with_no_tifs)} labels not matched")
            if not disable_gee_export:
                LabelExporter().export(labels=labels_with_no_tifs)

        # -------------------------------------------------
        # STEP 5: Create the features (X, y)
        # -------------------------------------------------
        if len(labels_with_tifs_but_no_features) > 0:
            Engineer().create_pickled_labeled_dataset(labels=labels_with_tifs_but_no_features)

        self.do_label_and_feature_amounts_match(labels)


@dataclass
class UnlabeledDataset:
    sentinel_dataset: str
    season: Season

    def export_earth_engine_data(self):
        RegionExporter(sentinel_dataset=self.sentinel_dataset).export(
            season=self.season, metres_per_polygon=None
        )
