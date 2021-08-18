from pathlib import Path
from unittest import TestCase
import glob
import pandas as pd
import pickle
import sys

from typing import List
from pprint import pprint

sys.path.append("..")

from utils import get_dvc_dir  # noqa: E402
from src.ETL.constants import (
    CROP_PROB,
    DEST_TIF,
    SUBSET,
    GEOWIKI_UNEXPORTED,
    UGANDA_UNEXPORTED,
)  # noqa: E402
from src.ETL.dataset import LabeledDataset, DataDir  # noqa: E402
from src.datasets_labeled import labeled_datasets  # noqa: E402


class IntegrationTestLabeledData(TestCase):
    """Tests that the features look right"""

    @classmethod
    def setUpClass(cls):
        get_dvc_dir("processed")
        get_dvc_dir("raw")
        get_dvc_dir("features")

    @staticmethod
    def get_files(
        directory: Path,
        extension=None,
    ) -> List[str]:
        if not directory.exists():
            return []
        files = glob.glob(str(directory) + "/**", recursive=True)
        if extension:
            files = [f.replace(str(directory) + "/", "") for f in files if f.endswith(extension)]
        return files

    @staticmethod
    def load_labels(d: LabeledDataset) -> pd.DataFrame:
        labels = pd.read_csv(d.get_path(DataDir.LABELS_PATH))

        if d.dataset == "geowiki_landcover_2017":
            labels = labels[~labels.index.isin(GEOWIKI_UNEXPORTED)]
        elif d.dataset == "Uganda":
            labels = labels[~labels.index.isin(UGANDA_UNEXPORTED)]

        return labels

    def test_each_label_has_tif(self):
        all_labels_have_tifs = True
        print("\n")
        for d in labeled_datasets:
            print("-------------------------------")
            print(f"{d.dataset}")

            labels = self.load_labels(d)
            parent_tif_dir = d.get_path(DataDir.RAW_IMAGES_DIR)
            expected = [f for f in labels[DEST_TIF]]
            expected_old = [str(Path(f).name) for f in labels[DEST_TIF]]
            actual = self.get_files(parent_tif_dir, extension=".tif")
            if len(actual) == 0:
                all_labels_have_tifs = False
                print(f"\u2716 WARNING: 0 tifs found")
                continue

            difference = list(set(expected) - set(actual))
            if len(difference) == 0:
                print(f"\u2714 Labels == tifs")
                continue

            difference_old = list(set(expected_old) - set(actual))
            if len(difference_old) == 0:
                print(f"\u2714 Labels == tifs (organized with previous convention)")
            elif len(difference_old) < len(difference):
                all_labels_have_tifs = False
                print(f"\u2716 Difference: {len(difference_old)}")
                pprint(difference_old[:5])
            else:
                all_labels_have_tifs = False
                print(f"\u2716 Difference: {len(difference)}")
                pprint(difference[:5])

        self.assertTrue(all_labels_have_tifs, "Check logs for which labels or tifs are missing.")

    def test_label_feature_subset_amounts(self):
        all_subsets_correct_size = True
        for d in labeled_datasets:
            labels = self.load_labels(d)
            train_val_test_counts = labels[labels[CROP_PROB] != 0.5][SUBSET].value_counts()

            print("------------------------------")
            print(d.dataset)
            for subset in ["training", "validation", "testing"]:
                labels_in_subset = 0
                if subset in train_val_test_counts:
                    labels_in_subset = train_val_test_counts[subset]
                features_in_subset = len(
                    self.get_files(d.get_path(DataDir.FEATURES_DIR) / subset, extension=".pkl")
                )
                if labels_in_subset != features_in_subset:
                    all_subsets_correct_size = False
                    print(
                        f"\u2716 {subset}: {labels_in_subset} labels, but {features_in_subset} features"
                    )
                else:
                    print(f"\u2714 {subset} amount")

        self.assertTrue(
            all_subsets_correct_size, "Check logs for which subsets have different sizes."
        )

    def test_features_for_duplicates(self):
        no_duplicates = True
        print("\n")
        for d in labeled_datasets:
            print("------------------------------")
            print(d.dataset)

            features = []
            for subset in ["training", "validation", "testing"]:
                features_dir = d.get_path(DataDir.FEATURES_DIR)
                if (features_dir / subset).exists():
                    for p in (features_dir / subset).iterdir():
                        with p.open("rb") as f:
                            features.append(pickle.load(f))
            features_df = pd.DataFrame([feat.__dict__ for feat in features])
            cols_to_check = ["label_lon", "label_lat", "start_date_str", "end_date_str"]
            duplicates = features_df[features_df.duplicated(subset=cols_to_check)]
            num_dupes = len(duplicates)

            if num_dupes > 0:
                no_duplicates = False
                print(f"\u2716 Duplicates: {num_dupes}")
                # duplicates.to_csv(f"../data/test/duplicates/{d.dataset}.csv", index=False)
            else:
                print(f"\u2714 Duplicates: {num_dupes}")

        self.assertTrue(no_duplicates, "Check logs for duplicates.")
