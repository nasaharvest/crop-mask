"""
Combines the earth observation data with the labels to create (x, y) training data
"""
from pathlib import Path

import os
import pandas as pd
import sys

# Change the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("..")

from src.ETL.dataset import load_all_features_as_df  # noqa: E402
from src.datasets_labeled import labeled_datasets  # noqa: E402


def check_empty_features(features_df, remove=False):
    """
    Some exported tif data may have nan values
    """
    empties = features_df[features_df["labelled_array"].isnull()]
    num_empty = len(empties)
    if num_empty > 0:
        print(f"\u2716 Found {num_empty} empty features")
        if remove:
            empties.filename.apply(lambda p: Path(p).unlink())
        return True
    else:
        print("\u2714 Found no empty features")
        return False


def check_duplicates(features_df: pd.DataFrame, remove=False):
    """
    Can happen when not all tifs have been downloaded and different labels are matched to same tif
    """
    cols_to_check = ["instance_lon", "instance_lat", "source_file"]
    duplicates = features_df[features_df.duplicated(subset=cols_to_check)]
    num_dupes = len(duplicates)
    if num_dupes > 0:
        print(f"\u2716 Found {num_dupes} duplicates")
        if remove:
            duplicates.filename.apply(lambda p: Path(p).unlink())
        return True
    else:
        print("\u2714 No duplicates found")
        return False


if __name__ == "__main__":

    datasets_to_process = [
        "Ethiopia_Tigray_2020",
        "Ethiopia_Tigray_2021",
        "geowiki_landcover_2017",
        "digitalearthafrica_eastern",
        "Ethiopia",
        "Kenya",
        "Mali",
        "Rwanda",
        "Togo",
    ]

    for d in labeled_datasets:
        if d.dataset in datasets_to_process:
            d.create_features()

    features_df = load_all_features_as_df()
    check_empty_features(features_df)
    check_duplicates(features_df)
