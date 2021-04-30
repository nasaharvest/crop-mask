from unittest import TestCase
from pathlib import Path
import pandas as pd

import sys

sys.path.append("..")

from utils import get_dvc_dir  # noqa: E402
from src.constants import CROP_PROB, SUBSET
from src.dataset_config import labeled_datasets  # noqa: E402


class IntegrationTestFeatures(TestCase):
    """Tests that the features look right"""

    temp_dir: Path = Path("")

    @staticmethod
    def get_file_count(directory):
        count = 0
        if directory.exists():
            for p in directory.iterdir():
                if p.is_file():
                    count += 1
        return count


    def test_label_to_ee_file_counts(self):
        get_dvc_dir("processed")
        get_dvc_dir("raw")

        for d in labeled_datasets:
            if d.dataset in ['geowiki_landcover_2017', "Kenya"]:
                continue
            labels = pd.read_csv(d.labels_path)
            tif_file_count = self.get_file_count(d.raw_images_dir)
            self.assertEqual(len(labels), tif_file_count,
                             f"Amount of {d.dataset} labels and {d.dataset} tif files is not the same")


    def test_feature_counts(self):
        get_dvc_dir("processed")
        get_dvc_dir("features")

        for d in labeled_datasets:
            if d.dataset in ['geowiki_landcover_2017', 'Kenya']:
                continue

            labels = pd.read_csv(d.labels_path)
            train_val_test_counts = labels[labels[CROP_PROB] != 0.5][SUBSET].value_counts()
            for subset in ['training', 'validation', 'testing']:
                labels_in_subset = 0
                if subset in train_val_test_counts:
                    labels_in_subset = train_val_test_counts[subset]
                features_in_subset = self.get_file_count(d.features_dir / subset)
                self.assertEqual(labels_in_subset, features_in_subset,
                                 f"{d.dataset} {subset} labels and features are not equal in size")
