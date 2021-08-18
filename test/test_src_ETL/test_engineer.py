from datetime import datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import pandas as pd
import pickle
import shutil
import tempfile
import xarray as xr

from src.ETL.constants import CROP_PROB, DEST_TIF, LAT, LON, SUBSET, START, END
from src.ETL.data_instance import CropDataInstance
from src.ETL.engineer import Engineer


class TestEngineer(TestCase):
    """Tests for Engineer"""

    engineer: Engineer

    @classmethod
    def setUpClass(cls):
        temp_path = Path(tempfile.gettempdir())
        sentinel_files_path = temp_path / "tifs"
        sentinel_files_path.mkdir(parents=True, exist_ok=True)
        save_dir = temp_path / "mock_save_dir"
        save_dir.mkdir(parents=True, exist_ok=True)

        cls.engineer = Engineer(
            sentinel_files_path=sentinel_files_path,
            labels_path=Path("mock_labels_path"),
            save_dir=save_dir,
            nan_fill=0,
            max_nan_ratio=0,
            add_ndvi=False,
            add_ndwi=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.engineer.save_dir)
        shutil.rmtree(cls.engineer.sentinel_files_path)

    def test_find_nearest(self):
        val = Engineer._find_nearest([1.0, 2.0, 3.0, 4.0, 5.0], 4.0)
        self.assertEqual(val, 4.0)

        val = Engineer._find_nearest(xr.DataArray([1.0, 2.0, 3.0, -4.0, -5.0]), -1.0)
        self.assertEqual(val, 1.0)

    @patch("src.ETL.engineer.load_tif")
    def test_create_labeled_data_instance(self, mock_load_tif):
        mock_load_tif.return_value = xr.DataArray(
            attrs={"x": np.array([15, 45]), "y": np.array([25, 55])},
            dims=["x", "y"],
            data=np.zeros((55, 45)),
        )

        feature_path = self.engineer.save_dir / "mock_feature_file.pkl"

        args = (
            "mock_file",
            str(feature_path),
            0.0,
            "training",
            "2020-01-01",
            "2021-01-01",
            20,
            30,
            "mock_file",
        )

        self.engineer._create_labeled_data_instance(args)

        with feature_path.open("rb") as f:
            actual_data_instance = pickle.load(f)

        expected_data_instance = CropDataInstance(
            crop_probability=0.0,
            instance_lat=30,
            instance_lon=20,
            label_lat=30,
            label_lon=20,
            labelled_array=0.0,
            data_subset="training",
            source_file="mock_file",
            start_date_str="2020-01-01",
            end_date_str="2021-01-01",
        )
        self.assertEqual(expected_data_instance, actual_data_instance)

    @patch("src.ETL.engineer.load_tif")
    @patch("src.ETL.engineer.pd.read_csv")
    def test_create_pickled_labeled_dataset(self, mock_read_csv, mock_load_tif):
        tif_filenames = ["00_2020-01-01_2021-01-01.tif", "01_2020-01-01_2021-01-01.tif"]
        for f in tif_filenames:
            (self.engineer.sentinel_files_path / f).touch()

        mock_read_csv.return_value = pd.DataFrame(
            {
                LON: [20, 40],
                LAT: [30, 50],
                CROP_PROB: [0.0, 1.0],
                SUBSET: ["training", "validation"],
                START: ["2020-01-01", "2020-01-01"],
                END: ["2021-01-01", "2021-01-01"],
                DEST_TIF: tif_filenames,
            }
        )

        mock_load_tif.return_value = xr.DataArray(
            attrs={"x": np.array([15, 45]), "y": np.array([25, 55])},
            dims=["x", "y"],
            data=np.zeros((55, 45)),
        )
        self.engineer.create_pickled_labeled_dataset()

        feature_files = list(self.engineer.save_dir.glob("**/*"))

        pkl_file_stems = [f.stem for f in feature_files if f.suffix == ".pkl"]

        for tif_file in tif_filenames:
            self.assertIn(Path(tif_file).stem, pkl_file_stems)
