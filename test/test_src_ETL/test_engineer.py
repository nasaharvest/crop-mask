from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import pandas as pd
import xarray as xr

from src.ETL.constants import CROP_PROB, FEATURE_PATH, LAT, LON, SUBSET, START, END, TIF_PATHS
from src.ETL.data_instance import CropDataInstance
from src.ETL.engineer import Engineer


class TestEngineer(TestCase):
    """Tests for Engineer"""

    def test_find_nearest(self):
        val, idx = Engineer._find_nearest([1.0, 2.0, 3.0, 4.0, 5.0], 4.0)
        self.assertEqual(val, 4.0)
        self.assertEqual(idx, 3)

        val, idx = Engineer._find_nearest(xr.DataArray([1.0, 2.0, 3.0, -4.0, -5.0]), -1.0)
        self.assertEqual(val, 1.0)
        self.assertEqual(idx, 0)

    def test_distance(self):
        self.assertEqual(Engineer._distance(0, 0, 0.01, 0.01), 1.5725337265584898)
        self.assertEqual(Engineer._distance(0, 0, 0.01, 0), 1.1119492645167193)
        self.assertEqual(Engineer._distance(0, 0, 0, 0.01), 1.1119492645167193)

    def test_distance_point_from_center(self):
        tif = xr.DataArray(attrs={"x": np.array([25, 35, 45]), "y": np.array([35, 45, 55])})
        self.assertEqual(Engineer._distance_point_from_center(0, 0, tif), 2.0)
        self.assertEqual(Engineer._distance_point_from_center(0, 1, tif), 1.0)
        self.assertEqual(Engineer._distance_point_from_center(1, 1, tif), 0.0)
        self.assertEqual(Engineer._distance_point_from_center(2, 1, tif), 1.0)

    @patch("src.ETL.engineer.load_tif")
    def test_find_matching_point_from_one(self, mock_load_tif):
        mock_load_tif.return_value = xr.DataArray(
            dims=["x", "y"],
            data=np.zeros((55, 45)),
        )
        labelled_np, closest_lon, closest_lat, source_file = Engineer()._find_matching_point(
            start="2020-10-10", tif_paths=[Path("mock_file")], label_lon=25, label_lat=35
        )
        self.assertEqual(closest_lon, 25)
        self.assertEqual(closest_lat, 35)
        self.assertEqual(source_file, "mock_file")
        self.assertEqual(labelled_np, np.array(0.0))

    @patch("src.ETL.engineer.load_tif")
    def test_find_matching_point_from_multiple(self, mock_load_tif):
        tif_paths = [Path("mock1"), Path("mock2"), Path("mock3")]

        def side_effect(path, days_per_timestep, start_date):
            idx = [i for i, p in enumerate(tif_paths) if p == path][0]
            return xr.DataArray(
                dims=["x", "y"],
                data=np.zeros((10 + idx, 10 + idx)) + idx,
            )

        mock_load_tif.side_effect = side_effect
        labelled_np, closest_lon, closest_lat, source_file = Engineer()._find_matching_point(
            start="2020-10-10", tif_paths=tif_paths, label_lon=8, label_lat=8
        )
        self.assertEqual(closest_lon, 8)
        self.assertEqual(closest_lat, 8)
        self.assertEqual(source_file, "mock3")
        self.assertEqual(labelled_np, np.array(2.0))

    @patch("src.ETL.engineer.Path.open")
    @patch("src.ETL.engineer.Engineer._find_matching_point")
    @patch("src.ETL.engineer.process_bands")
    @patch("src.ETL.engineer.pickle.dump")
    def test_create_pickled_labeled_dataset(
        self, mock_dump, mock_process_bands, mock_find_matching_point, mock_open
    ):
        mock_find_matching_point.return_value = (
            None,
            0.1,
            0.1,
            "mock_file",
        )

        mock_process_bands.return_value = np.array([0.0])

        mock_labels = pd.DataFrame(
            {
                LON: [20, 40],
                LAT: [30, 50],
                CROP_PROB: [0.0, 1.0],
                SUBSET: ["training", "validation"],
                START: ["2020-01-01", "2020-01-01"],
                END: ["2021-01-01", "2021-01-01"],
                TIF_PATHS: [[Path("tif1")], [Path("tif2"), Path("tif3")]],
                FEATURE_PATH: ["feature1", "feature2"],
            }
        )

        Engineer().create_pickled_labeled_dataset(mock_labels)

        instances = [
            CropDataInstance(
                crop_probability=0.0,
                instance_lat=0.1,
                instance_lon=0.1,
                label_lat=30,
                label_lon=20,
                labelled_array=np.array([0.0]),
                data_subset="training",
                source_file="mock_file",
                start_date_str="2020-01-01",
                end_date_str="2021-01-01",
            ),
            CropDataInstance(
                crop_probability=1.0,
                instance_lat=0.1,
                instance_lon=0.1,
                label_lat=50,
                label_lon=40,
                labelled_array=np.array([0.0]),
                data_subset="validation",
                source_file="mock_file",
                start_date_str="2020-01-01",
                end_date_str="2021-01-01",
            ),
        ]

        self.assertEqual(mock_dump.call_count, 2)
        self.assertEqual(mock_dump.call_args_list[0][0][0], instances[0])
        self.assertEqual(mock_dump.call_args_list[1][0][0], instances[1])
