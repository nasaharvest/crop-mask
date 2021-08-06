from datetime import datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import pandas as pd
import shutil
import tempfile
import xarray as xr

from src.ETL.constants import CROP_PROB, LAT, LON, SUBSET, START, END
from src.ETL.data_instance import CropDataInstance
from src.ETL.engineer import Engineer


class TestEngineer(TestCase):
    """Tests for Engineer"""

    engineer: Engineer

    @classmethod
    @patch("src.ETL.engineer.pd.read_csv")
    @patch("src.ETL.engineer.Path.glob")
    def setUpClass(cls, mock_glob, mock_read_csv):
        geospatial_file = tempfile.NamedTemporaryFile(suffix="00_2020-01-01_2021-01-01.tif")
        mock_glob.return_value = [Path(geospatial_file.name)]
        mock_read_csv.return_value = pd.DataFrame(
            {
                LON: [20, 40],
                LAT: [30, 50],
                CROP_PROB: [0.0, 1.0],
                SUBSET: ["training", "validation"],
                START: ["2020-01-01", "2020-01-01"],
                END: ["2021-01-01", "2021-01-01"],
            }
        )
        cls.engineer = Engineer(
            sentinel_files_path=Path("mock_sentinel_path"),
            labels_path=Path("mock_labels_path"),
            save_dir=Path("mock_features_path"),
            nan_fill=0,
            max_nan_ratio=0,
            add_ndvi=False,
            add_ndwi=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.engineer.save_dir)

    @staticmethod
    def generate_data_kwargs():
        return {
            "path_to_file": Path("mock_file"),
            "calculate_normalizing_dict": False,
            "start_date": datetime(2020, 1, 1, 0, 0, 0),
            "end_date": datetime(2021, 1, 1, 0, 0, 0),
            "days_per_timestep": 30,
        }

    def test_find_nearest(self):
        val, idx = Engineer._find_nearest([1.0, 2.0, 3.0, 4.0, 5.0], 4.0)
        self.assertEqual(val, 4.0)
        self.assertEqual(idx, 3)

        val, idx = Engineer._find_nearest(xr.DataArray([1.0, 2.0, 3.0, -4.0, -5.0]), -1.0)
        self.assertEqual(val, 1.0)
        self.assertEqual(idx, 0)

    def test_update_normalizing_values(self):
        norm_dict = {"n": 0}
        array = np.array([[1, 2, 3], [2, 3, 4]])
        Engineer._update_normalizing_values(norm_dict, array)
        self.assertEqual(norm_dict["n"], array.shape[0])
        self.assertTrue(np.allclose(norm_dict["mean"], np.array([1.5, 2.5, 3.5])))
        self.assertTrue(np.allclose(norm_dict["M2"], np.array([0.5, 0.5, 0.5])))

        array2 = np.array([[3, 4, 5], [4, 5, 6]])
        Engineer._update_normalizing_values(norm_dict, array2)
        self.assertEqual(norm_dict["n"], array.shape[0] + array2.shape[0])
        self.assertTrue(np.allclose(norm_dict["mean"], np.array([2.5, 3.5, 4.5])))
        self.assertTrue(np.allclose(norm_dict["M2"], np.array([5.0, 5.0, 5.0])))

    def test_update_batch_normalizing_values(self):
        norm_dict = {"n": 0}
        array = np.array([[[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6]]])
        self.engineer._update_batch_normalizing_values(norm_dict, array)
        self.assertEqual(norm_dict["n"], array.shape[0] * array.shape[1])
        self.assertTrue(np.allclose(norm_dict["mean"], np.array([2.5, 3.5, 4.5])))
        self.assertTrue(np.allclose(norm_dict["M2"], np.array([5.0, 5.0, 5.0])))

    def test_calculate_normalizing_dict(self):
        norm_dict = {"n": 4, "mean": np.array([2.5, 3.5, 4.5]), "M2": np.array([5.0, 5.0, 5.0])}
        normalizing_dict = self.engineer._calculate_normalizing_dict(norm_dict)
        self.assertTrue(np.allclose(normalizing_dict["mean"], norm_dict["mean"]))
        self.assertTrue(
            np.allclose(normalizing_dict["std"], np.array([1.29099445, 1.29099445, 1.29099445]))
        )
        empty_norm_dict = {}
        normalizing_dict = self.engineer._calculate_normalizing_dict(empty_norm_dict)
        self.assertIsNone(normalizing_dict)

    @patch("src.ETL.engineer.load_tif")
    def test_create_labeled_data_instance_no_overlap(self, mock_load_tif):
        mock_load_tif.return_value = xr.DataArray(
            attrs={"x": np.array([25, 35]), "y": np.array([35, 45])}
        )
        kwargs = self.generate_data_kwargs()
        data_instance = self.engineer._create_labeled_data_instance(**kwargs)
        self.assertIsNone(data_instance)

    @patch("src.ETL.engineer.load_tif")
    def test_create_labeled_data_instance(self, mock_load_tif):
        mock_load_tif.return_value = xr.DataArray(
            attrs={"x": np.array([15, 45]), "y": np.array([25, 55])},
            dims=["x", "y"],
            data=np.zeros((55, 45)),
        )
        kwargs = self.generate_data_kwargs()
        actual_data_instance = self.engineer._create_labeled_data_instance(**kwargs)
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
    def test_create_pickled_labeled_dataset(self, mock_load_tif):
        mock_load_tif.return_value = xr.DataArray(
            attrs={"x": np.array([15, 45]), "y": np.array([25, 55])},
            dims=["x", "y"],
            data=np.zeros((55, 45)),
        )
        kwargs = self.generate_data_kwargs()
        for k in ["path_to_file", "start_date", "end_date"]:
            del kwargs[k]
        self.engineer.create_pickled_labeled_dataset(**kwargs)
