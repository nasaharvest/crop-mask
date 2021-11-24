from unittest import TestCase
import pandas as pd
import pickle
import os
import sys
import unittest

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.utils import features_dir, memoize  # noqa: E402
from src.datasets_labeled import labeled_datasets  # noqa: E402


class IntegrationTestLabeledData(TestCase):
    """Tests that the features look right"""

    @staticmethod
    @memoize
    def load_features_as_df() -> pd.DataFrame:
        features = []
        files = list(features_dir.glob("*.pkl"))
        print("Loading features...")
        for p in files:
            with p.open("rb") as f:
                features.append(pickle.load(f))
        df = pd.DataFrame([feat.__dict__ for feat in features])
        df["filename"] = files
        return df

    def test_label_feature_subset_amounts(self):
        all_subsets_correct_size = True
        for d in labeled_datasets:
            print("------------------------------")
            print(d.dataset)
            labels = d.load_labels()
            if not d.do_label_and_feature_amounts_match(labels):
                all_subsets_correct_size = False

        self.assertTrue(
            all_subsets_correct_size, "Check logs for which subsets have different sizes."
        )

    def test_features_for_duplicates(self):
        features_df = self.load_features_as_df()
        cols_to_check = ["label_lon", "label_lat", "start_date_str", "end_date_str"]
        duplicates = features_df[features_df.duplicated(subset=cols_to_check)]
        num_dupes = len(duplicates)
        self.assertTrue(num_dupes == 0, f"Found {num_dupes} duplicates")

    def test_features_for_closeness(self):
        features_df = self.load_features_as_df()
        label_tif_mismatch = features_df[
            ((features_df["label_lon"] - features_df["instance_lon"]) > 0.0001)
            | ((features_df["label_lat"] - features_df["instance_lat"]) > 0.0001)
        ]
        num_mismatched = len(label_tif_mismatch)
        self.assertTrue(num_mismatched == 0, "Found {num_mismatched} mismatched labels+tifs.")


if __name__ == "__main__":
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=2)
    unittest.main(testRunner=runner)
