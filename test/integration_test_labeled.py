from pathlib import Path
from unittest import TestCase
import pandas as pd
import pickle
import os
import sys
import unittest

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.ETL.constants import ALREADY_EXISTS, LAT, LON, FEATURE_PATH
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
        cols_to_check = ["instance_lon", "instance_lat", "source_file"]
        duplicates = features_df[features_df.duplicated(subset=cols_to_check)]
        num_dupes = len(duplicates)
        self.assertTrue(num_dupes == 0, f"Found {num_dupes} duplicates")

    def test_features_for_emptiness(self):
        features_df = self.load_features_as_df()
        is_empty = features_df["labelled_array"].isnull()
        num_empty_features = len(features_df[is_empty])
        self.assertTrue(num_empty_features == 0, f"Found {num_empty_features} empty features")
        # Code to delete empty features features:
        # features_df[is_empty].filename.apply(lambda p: Path(p).unlink())

    def test_features_for_closeness(self):
        total_num_mismatched = 0
        print("")  # Ensure output starts on new line
        for d in labeled_datasets:

            labels = d.load_labels()
            labels = labels[labels[ALREADY_EXISTS]].copy()

            if len(labels) == 0:
                print(f"\\ {d.dataset}:\t\tNo features")
                continue

            def load_feature(p):
                with Path(p).open("rb") as f:
                    return pickle.load(f)

            features = labels[FEATURE_PATH].apply(load_feature)

            labels["instance_lon"] = features.apply(lambda f: f.instance_lon)
            labels["instance_lat"] = features.apply(lambda f: f.instance_lat)

            # features_df = self.load_features_as_df()
            label_tif_mismatch = labels[
                ((labels[LON] - labels["instance_lon"]) > 0.0001)
                | ((labels[LAT] - labels["instance_lat"]) > 0.0001)
            ]
            num_mismatched = len(label_tif_mismatch)
            if num_mismatched > 0:
                mark = "\u2716"
            else:
                mark = "\u2714"
            print(f"{mark} {d.dataset}:\t\tMismatches: {num_mismatched}")
        self.assertTrue(
            total_num_mismatched == 0, "Found {total_num_mismatched} mismatched labels+tifs."
        )


if __name__ == "__main__":
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=2)
    unittest.main(testRunner=runner)
