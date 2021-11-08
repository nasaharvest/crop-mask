from pathlib import Path
from unittest import TestCase
import pandas as pd
import pickle
import os
import sys
import unittest

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.utils import get_dvc_dir, get_data_dir  # noqa: E402
from src.ETL.constants import CROP_PROB, SUBSET  # noqa: E402
from src.ETL.dataset import DataDir  # noqa: E402
from src.datasets_labeled import labeled_datasets  # noqa: E402


class IntegrationTestLabeledData(TestCase):
    """Tests that the features look right"""

    @classmethod
    def setUpClass(cls):
        get_dvc_dir("processed")
        get_dvc_dir("features")
        unexported_file = get_data_dir() / "unexported.txt"
        cls.unexported = pd.read_csv(unexported_file, sep="\n", header=None)[0].tolist()

    @classmethod
    def load_labels(cls, p: Path) -> pd.DataFrame:
        labels = pd.read_csv(p)
        labels = labels[~labels["filename"].isin(cls.unexported)]
        labels = labels[labels[CROP_PROB] != 0.5]
        return labels

    @staticmethod
    def load_features_as_df(features_dir: Path) -> pd.DataFrame:
        features = []
        files = []
        for subset in ["training", "validation", "testing"]:
            if (features_dir / subset).exists():
                for p in (features_dir / subset).glob("*.pkl"):
                    with p.open("rb") as f:
                        features.append(pickle.load(f))
                        files.append(p)
        df = pd.DataFrame([feat.__dict__ for feat in features])
        df["filename"] = files
        return df

    def test_label_feature_subset_amounts(self):
        all_subsets_correct_size = True
        for d in labeled_datasets:
            labels = self.load_labels(d.get_path(DataDir.LABELS_PATH))
            train_val_test_counts = labels[SUBSET].value_counts()

            print("------------------------------")
            print(d.dataset)
            for subset in ["training", "validation", "testing"]:
                labels_in_subset = 0
                if subset in train_val_test_counts:
                    labels_in_subset = train_val_test_counts[subset]
                features_in_subset = len(
                    list((d.get_path(DataDir.FEATURES_DIR) / subset).glob("**/*.pkl"))
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

            features_df = self.load_features_as_df(d.get_path(DataDir.FEATURES_DIR))
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

            features_df = self.load_features_as_df(d.get_path(DataDir.FEATURES_DIR))
            label_tif_mismatch = features_df[
                ((features_df["label_lon"] - features_df["instance_lon"]) > 0.0001)
                | ((features_df["label_lat"] - features_df["instance_lat"]) > 0.0001)
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
