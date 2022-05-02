from datetime import date
from dateutil.relativedelta import relativedelta
from pathlib import Path
from unittest import TestCase
import numpy as np
import pandas as pd
import pickle
import os
import sys
import unittest

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from src.ETL.constants import (  # noqa: E402
    ALREADY_EXISTS,
    FEATURE_FILENAME,
    LAT,
    LON,
    FEATURE_PATH,
    START,
    END,
    SUBSET,
)

from src.utils import data_dir  # noqa: E402
from src.ETL.data_instance import CropDataInstance  # noqa: E402
from src.ETL.dataset import get_label_timesteps, load_all_features_as_df  # noqa: E402
from src.datasets_labeled import labeled_datasets  # noqa: E402

unexported_file = data_dir / "unexported.txt"
unexported = pd.read_csv(unexported_file, sep="\n", header=None)[0].tolist()

duplicates_data_file = data_dir / "duplicates.txt"
duplicates_data = pd.read_csv(duplicates_data_file, sep="\n", header=None)[0].tolist()


def load_feature(p):
    with Path(p).open("rb") as f:
        return pickle.load(f)


class IntegrationTestLabeledData(TestCase):
    """Tests that the features look right"""

    @staticmethod
    def load_labels(is_print=False):
        print("")
        datasets = {}
        for d in labeled_datasets:
            try:
                datasets[d.dataset] = d.load_labels()
                if is_print:
                    print(d.summary(datasets[d.dataset]))
            except FileNotFoundError:
                continue
        return datasets

    def test_features_with_no_labels(self):
        feature_name_list = []
        for _, labels in self.load_labels().items():
            feature_name_list += labels[FEATURE_FILENAME].tolist()

        features_df = load_all_features_as_df()
        features_df_stems = features_df.filename.apply(lambda p: p.stem)
        features_with_no_label = features_df[~features_df_stems.isin(feature_name_list)]
        amount = len(features_with_no_label)
        self.assertTrue(amount == 0, f'Found {amount} features with no labels')

    def test_each_pickle_file_is_crop_data_instance(self):
        each_pickle_file_is_crop_data_instance = True
        for name, labels in self.load_labels().items():
            labels = labels[labels[ALREADY_EXISTS]].copy()
            all_features = labels[FEATURE_PATH].apply(load_feature)
            good_features = [feat for feat in all_features if isinstance(feat, CropDataInstance)]

            if len(good_features) == len(all_features):
                mark = "\u2714"
            else:
                mark = "\u2716"
                each_pickle_file_is_crop_data_instance = False
            print(f"{mark} {name} has {len(good_features)} features out of {len(all_features)}.")
        self.assertTrue(
            each_pickle_file_is_crop_data_instance,
            "Not all pickle files are crop data instances, check logs for details.",
        )

    def test_label_feature_subset_amounts(self):
        # If this test is failing and there are no activate exports,
        # temporarily set the add_to_unexported_file to True
        # to update the unexported list
        add_to_unexported_file = False

        all_subsets_correct_size = True

        newly_unexported = []
        for _, labels in self.load_labels(is_print=True).items():
            if not labels[ALREADY_EXISTS].all():
                labels[ALREADY_EXISTS] = np.vectorize(lambda p: Path(p).exists())(
                    labels[FEATURE_PATH]
                )
            train_val_test_counts = labels[SUBSET].value_counts()
            for subset, labels_in_subset in train_val_test_counts.items():
                features_in_subset = labels[labels[SUBSET] == subset][ALREADY_EXISTS].sum()
                if labels_in_subset != features_in_subset:
                    all_subsets_correct_size = False
                    if add_to_unexported_file:
                        labels_with_no_feature = labels[
                            (labels[SUBSET] == subset) & ~labels[ALREADY_EXISTS]
                        ]
                        assert len(labels_with_no_feature) == (
                            labels_in_subset - features_in_subset
                        )
                        newly_unexported += labels_with_no_feature[FEATURE_FILENAME].tolist()

        if add_to_unexported_file and not all_subsets_correct_size:
            with unexported_file.open("w") as f:
                f.write("\n".join(unexported + newly_unexported))

        self.assertTrue(
            all_subsets_correct_size, "Check logs for which subsets have different sizes."
        )

    def test_features_for_duplicates(self):
        # If this test is failing you can temporarily set remove_duplicates to True
        # and rerun create_features.py
        add_to_duplicates_file = False
        features_df = load_all_features_as_df()
        cols_to_check = ["instance_lon", "instance_lat", "source_file"]
        duplicates = features_df[features_df.duplicated(subset=cols_to_check)]
        num_dupes = len(duplicates)
        if add_to_duplicates_file and num_dupes > 0:
            feature_filenames = duplicates.filename.apply(lambda p: Path(p).stem).tolist()
            with duplicates_data_file.open("w") as f:
                f.write("\n".join(duplicates_data + feature_filenames))
        self.assertTrue(num_dupes == 0, f"Found {num_dupes} duplicates")

    def test_features_for_emptiness(self):
        features_df = load_all_features_as_df()
        is_empty = features_df["labelled_array"].isnull()
        num_empty_features = len(features_df[is_empty])
        self.assertTrue(
            num_empty_features == 0,
            f"Found {num_empty_features} empty features, run create_features.py to solve this.",
        )

    def test_all_features_have_18_bands(self):
        features_df = load_all_features_as_df()
        is_empty = features_df["labelled_array"].isnull()
        band_amount = features_df[~is_empty]["labelled_array"].apply(lambda f: f.shape[-1]).unique()
        self.assertEqual(band_amount.tolist(), [18], "Found {band_amount} bands")

    def test_all_features_start_with_january_first(self):
        features_df = load_all_features_as_df()
        starts_with_jan_first = features_df.filename.str.contains("_01_01")
        self.assertTrue(starts_with_jan_first.all(), "Not all features start with January 1st")

    def test_label_and_feature_ranges_match(self):
        all_label_and_feature_ranges_match = True
        for name, labels in self.load_labels().items():
            labels = labels[labels[ALREADY_EXISTS]].copy()
            if len(labels) == 0:
                continue
            features = labels[FEATURE_PATH].apply(load_feature)
            features_df = pd.DataFrame([feat.__dict__ for feat in features])
            feature_month_amount = features_df["labelled_array"].apply(lambda f: f.shape[0])
            label_month_amount = get_label_timesteps(labels).reset_index(drop=True)
            label_ranges = label_month_amount.value_counts().to_dict()
            feature_ranges = feature_month_amount.value_counts().to_dict()
            if (feature_month_amount == label_month_amount).all():
                mark = "\u2714"
                last_word = "match"
            else:
                mark = "\u2716"
                last_word = "mismatch"
                all_label_and_feature_ranges_match = False
            # Code to delete:
            # labels.reset_index(drop=True)[feature_month_amount != label_month_amount]
            # [FEATURE_PATH].apply(lambda p: Path(p).unlink())
            print(
                f"{mark} {name} label {label_ranges} and "
                + f"feature {feature_ranges} ranges {last_word}"
            )
        self.assertTrue(
            all_label_and_feature_ranges_match, "Check logs for which subsets have different sizes."
        )

    def test_labels_have_start_before_end_date(self):
        all_labels_have_consistent_dates = True
        for name, labels in self.load_labels().items():
            consistent_dates = pd.to_datetime(labels[START]) < pd.to_datetime(labels[END])
            if consistent_dates.all():
                mark = "\u2714"
                last_word = "consistent dates"
            else:
                mark = "\u2716"
                last_word = f"{(~consistent_dates).sum()} inconsistent dates"
                all_labels_have_consistent_dates = False
            print(f"{mark} {name} label has {last_word}")
        self.assertTrue(
            all_labels_have_consistent_dates, "Check logs for which labels have inconsistent dates."
        )

    def test_all_older_features_have_24_months(self):
        current_cutoff_date = date.today().replace(day=1) + relativedelta(months=-4)
        two_years_before_cutoff = pd.Timestamp(current_cutoff_date + relativedelta(months=-24))

        all_older_features_have_24_months = True

        for name, labels in self.load_labels().items():
            cutoff = pd.to_datetime(labels[START]) < two_years_before_cutoff
            labels = labels[labels[ALREADY_EXISTS] & cutoff].copy()
            if len(labels) == 0:
                continue
            features = labels[FEATURE_PATH].apply(load_feature)
            features_df = pd.DataFrame([feat.__dict__ for feat in features])
            is_empty = features_df["labelled_array"].isnull()
            month_amount = (
                features_df[~is_empty]["labelled_array"].apply(lambda f: f.shape[0]).unique()
            )

            if month_amount.tolist() == [24]:
                mark = "\u2714"
            else:
                all_older_features_have_24_months = False
                mark = "\u2716"
            print(f"{mark} {name} \t\t{month_amount.tolist()}")

        self.assertTrue(
            all_older_features_have_24_months, "Not all older features have 24 months, check logs."
        )

    def test_features_for_closeness(self):
        total_num_mismatched = 0
        for name, labels in self.load_labels().items():
            labels = labels[labels[ALREADY_EXISTS]].copy()

            if len(labels) == 0:
                print(f"\\ {name}:\t\tNo features")
                continue

            features = labels[FEATURE_PATH].apply(load_feature)

            labels["instance_lon"] = features.apply(lambda f: f.instance_lon)
            labels["instance_lat"] = features.apply(lambda f: f.instance_lat)

            label_tif_mismatch = labels[
                ((labels[LON] - labels["instance_lon"]) > 0.0001)
                | ((labels[LAT] - labels["instance_lat"]) > 0.0001)
            ]
            num_mismatched = len(label_tif_mismatch)
            if num_mismatched > 0:
                mark = "\u2716"
            else:
                mark = "\u2714"
            print(f"{mark} {name}:\t\tMismatches: {num_mismatched}")
        self.assertTrue(
            total_num_mismatched == 0, f"Found {total_num_mismatched} mismatched labels+tifs."
        )

    def test_label_coordinate_duplication(self):
        """For now this test is just a status report"""
        all_dfs = []
        for name, labels in self.load_labels().items():
            labels["name"] = name
            all_dfs.append(labels)

        big_df = pd.concat(all_dfs)
        duplicates = big_df[big_df.duplicated(subset=[LON, LAT], keep=False)]
        duplicates["start_year"] = pd.to_datetime(duplicates[START]).dt.year.astype(str)
        df = duplicates.groupby([LON, LAT], as_index=False, sort=False).agg(
            {
                "name": lambda names: ",".join(names.unique()),
                SUBSET: lambda subs: ",".join(subs.unique()),
                "start_year": lambda start_years: ",".join(start_years),
            }
        )
        print("------------------------------------------------------")
        print("Label coordinate spill over")
        print("------------------------------------------------------")
        print(df[["name", SUBSET, "start_year"]].value_counts())


if __name__ == "__main__":
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=2)
    unittest.main(testRunner=runner)
