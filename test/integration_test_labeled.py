from unittest import TestCase
from typing import List
import pandas as pd
import pickle
import sys

sys.path.append("..")

from utils import get_dvc_dir  # noqa: E402
from src.constants import CROP_PROB, SUBSET  # noqa: E402
from data.datasets_labeled import labeled_datasets  # noqa: E402


class IntegrationTestLabeledData(TestCase):
    """Tests that the features look right"""

    datasets_to_skip: List[str] = ["Kenya"]

    @classmethod
    def setUpClass(cls):
        get_dvc_dir("processed")
        get_dvc_dir("raw")
        get_dvc_dir("features")

    @staticmethod
    def get_file_count(directory):
        count = 0
        if directory.exists():
            for p in directory.iterdir():
                if p.is_file():
                    count += 1
        return count

    def test_each_label_has_tif(self):
        for d in labeled_datasets:
            if d.dataset in self.datasets_to_skip:
                continue
            labels = pd.read_csv(d.labels_path)
            tif_file_count = self.get_file_count(d.raw_images_dir)
            self.assertEqual(
                len(labels),
                tif_file_count,
                f"Amount of {d.dataset} labels ({len(labels)}) and resulting "
                f"{d.dataset} tif files ({tif_file_count}) is not the same",
            )
            print(f"{d.dataset} - each label has a tif file")

    def test_label_feature_subset_amounts(self):
        for d in labeled_datasets:
            if d.dataset in self.datasets_to_skip:
                continue

            labels = pd.read_csv(d.labels_path)
            train_val_test_counts = labels[labels[CROP_PROB] != 0.5][SUBSET].value_counts()
            for subset in ["training", "validation", "testing"]:
                labels_in_subset = 0
                if subset in train_val_test_counts:
                    labels_in_subset = train_val_test_counts[subset]
                features_in_subset = self.get_file_count(d.features_dir / subset)
                self.assertEqual(
                    labels_in_subset,
                    features_in_subset,
                    f"{d.dataset} {subset} labels ({labels_in_subset}) and features "
                    f"({features_in_subset}) are not equal in size",
                )
            print(f"{d.dataset} - label distribution exactly matches features")

    def test_features_for_duplicates(self):
        for d in labeled_datasets:
            if d.dataset in self.datasets_to_skip:
                continue

            features = []
            for subset in ["training", "validation", "testing"]:
                if (d.features_dir / subset).exists():
                    for p in (d.features_dir / subset).iterdir():
                        with p.open("rb") as f:
                            features.append(pickle.load(f))
            features_df = pd.DataFrame([feat.__dict__ for feat in features])
            cols_to_check = ["label_lon", "label_lat", "start_date_str", "end_date_str"]
            num_dupes = len(features_df[features_df.duplicated(subset=cols_to_check)])
            self.assertEqual(num_dupes, 0, f"{d.dataset} features contain {num_dupes} duplicates.")
