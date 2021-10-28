from pathlib import Path
from unittest import TestCase
import glob
import pandas as pd
import pickle
import os
import sys
import unittest

from typing import List

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.utils import get_dvc_dir, get_data_dir  # noqa: E402
from src.ETL.constants import CROP_PROB, SUBSET  # noqa: E402
from src.ETL.dataset import LabeledDataset, DataDir  # noqa: E402
from src.datasets_labeled import labeled_datasets  # noqa: E402


class IntegrationTestLabeledData(TestCase):
    """Tests that the features look right"""

    @classmethod
    def setUpClass(cls):
        get_dvc_dir("processed")
        get_dvc_dir("raw")
        get_dvc_dir("features")
        unexported_file = get_data_dir() / "unexported.txt"
        cls.unexported = pd.read_csv(unexported_file, sep="\n", header=None)[0].tolist()

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

    @classmethod
    def load_labels(cls, d: LabeledDataset) -> pd.DataFrame:
        labels = pd.read_csv(d.get_path(DataDir.LABELS_PATH))
        labels = labels[~labels["filename"].isin(cls.unexported)]
        return labels

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
                    self.get_files(d.get_path(DataDir.FEATURES_DIR2) / subset, extension=".pkl")
                )
                if labels_in_subset != features_in_subset:
                    all_subsets_correct_size = False
                    print(
                        f"\u2716 {subset}: {labels_in_subset} labels, "
                        + f"but {features_in_subset} features"
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
                features_dir = d.get_path(DataDir.FEATURES_DIR2)
                if (features_dir / subset).exists():
                    for p in (features_dir / subset).glob("*.pkl"):
                        with p.open("rb") as f:
                            features.append(pickle.load(f))
            features_df = pd.DataFrame([feat.__dict__ for feat in features])
            cols_to_check = ["label_lon", "label_lat", "start_date_str", "end_date_str"]
            duplicates = features_df[features_df.duplicated(subset=cols_to_check)]
            num_dupes = len(duplicates)

            if num_dupes > 0:
                no_duplicates = False
                print(f"\u2716 Duplicates: {num_dupes}")
            else:
                print(f"\u2714 Duplicates: {num_dupes}")

        self.assertTrue(no_duplicates, "Check logs for duplicates.")

    def test_features_for_closeness(self):
        no_mismatches = True
        print("\n")
        for d in labeled_datasets:
            print("------------------------------")
            print(d.dataset)

            features = []
            for subset in ["training", "validation", "testing"]:
                features_dir = d.get_path(DataDir.FEATURES_DIR2)
                if (features_dir / subset).exists():
                    for p in (features_dir / subset).glob("*.pkl"):
                        with p.open("rb") as f:
                            features.append(pickle.load(f))
            features_df = pd.DataFrame([feat.__dict__ for feat in features])
            label_tif_mismatch = features_df[
                (features_df["label_lon"] - features_df["instance_lon"]) > 0.0001
            ]
            num_mismatched = len(label_tif_mismatch)

            if num_mismatched > 0:
                no_mismatches = False
                print(f"\u2716 Mismatches: {num_mismatched}")
            else:
                print(f"\u2714 Mismatches: {num_mismatched}")

        self.assertTrue(no_mismatches, "Check logs for mismatched.")


if __name__ == "__main__":
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=2)
    unittest.main(testRunner=runner)
