from pathlib import Path
from unittest import TestCase
import glob
import pandas as pd
import pickle
import sys

sys.path.append("..")

from utils import get_dvc_dir  # noqa: E402
from src.ETL.constants import CROP_PROB, SUBSET, GEOWIKI_UNEXPORTED  # noqa: E402
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
    def get_file_count(directory: Path, extension=None):
        if not directory.exists():
            return 0
        files = glob.glob(str(directory) + "/**", recursive=True)
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return len(files)

    @staticmethod
    def load_labels(d: LabeledDataset) -> pd.DataFrame:
        labels = pd.read_csv(d.get_path(DataDir.LABELS_PATH))

        if d.dataset == "geowiki_landcover_2017":
            labels = labels[~labels.index.isin(GEOWIKI_UNEXPORTED)]

        return labels

    def test_each_label_has_tif(self):
        for d in labeled_datasets:

            labels = self.load_labels(d)
            label_count = len(labels)

            tif_file_count = self.get_file_count(
                d.get_path(DataDir.RAW_IMAGES_DIR), extension=".tif"
            )
            self.assertEqual(
                label_count,
                tif_file_count,
                f"Amount of {d.dataset} labels ({label_count}) and resulting "
                f"{d.dataset} tif files ({tif_file_count}) is not the same",
            )
            print(f"{d.dataset} - each label has a tif file")

    def test_label_feature_subset_amounts(self):
        for d in labeled_datasets:

            # geowiki has 202 examples that are not associated with any labels
            if d.dataset == "geowiki_landcover_2017":
                continue

            labels = self.load_labels(d)
            train_val_test_counts = labels[labels[CROP_PROB] != 0.5][SUBSET].value_counts()
            for subset in ["training", "validation", "testing"]:
                labels_in_subset = 0
                if subset in train_val_test_counts:
                    labels_in_subset = train_val_test_counts[subset]
                features_in_subset = self.get_file_count(d.features_dir / subset, extension=".pkl")
                self.assertEqual(
                    labels_in_subset,
                    features_in_subset,
                    f"{d.dataset} {subset} labels ({labels_in_subset}) and features "
                    f"({features_in_subset}) are not equal in size",
                )
            print(f"{d.dataset} - label distribution exactly matches features")

    def test_features_for_duplicates(self):
        for d in labeled_datasets:
            if d.dataset == "geowiki_landcover_2017":
                continue

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
                duplicates.to_csv(f"../data/test/duplicates/{d.dataset}.csv", index=False)
            self.assertEqual(num_dupes, 0, f"{d.dataset} features contain {num_dupes} duplicates.")
            print(f"{d.dataset} - features have no duplicates")
