from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import pandas as pd
import xarray as xr

from src.ETL.constants import CROP_PROB, FEATURE_PATH, LAT, LON, START, END, TIF_PATHS
from src.ETL.data_instance import CropDataInstance
from src.ETL.dataset import find_matching_point, create_pickled_labeled_dataset


class TestDataset(TestCase):
    """Tests for dataset"""

    @patch("src.ETL.dataset.storage")
    @patch("src.ETL.dataset.Engineer.load_tif")
    def test_find_matching_point_from_one(self, mock_load_tif, mock_storage):
        mock_data = xr.DataArray(data=np.ones((24, 19, 17, 17)), dims=("time", "band", "y", "x"))
        mock_load_tif.return_value = mock_data, 0.0
        labelled_np, closest_lon, closest_lat, source_file = find_matching_point(
            start="2020-10-10", tif_paths=[Path("mock")], label_lon=5, label_lat=5
        )
        self.assertEqual(closest_lon, 5)
        self.assertEqual(closest_lat, 5)
        self.assertEqual(source_file, "mock")
        self.assertEqual(labelled_np.shape, (24, 18))
        expected = np.ones((24, 18))
        expected[:, -1] = 0  # NDVI is 0
        self.assertTrue((labelled_np == expected).all())

    @patch("src.ETL.dataset.storage")
    @patch("src.ETL.dataset.Engineer.load_tif")
    def test_find_matching_point_from_multiple(self, mock_load_tif, mock_storage):
        tif_paths = [Path("mock1"), Path("mock2"), Path("mock3")]

        def side_effect(path, start_date, num_timesteps):
            idx = [i for i, p in enumerate(tif_paths) if p.stem == Path(path).stem][0]
            return (
                xr.DataArray(
                    dims=("time", "band", "y", "x"),
                    data=np.ones((24, 19, 17, 17)) * idx,
                ),
                0.0,
            )

        mock_load_tif.side_effect = side_effect
        labelled_np, closest_lon, closest_lat, source_file = find_matching_point(
            start="2020-10-10", tif_paths=tif_paths, label_lon=8, label_lat=8
        )
        self.assertEqual(closest_lon, 8)
        self.assertEqual(closest_lat, 8)
        self.assertEqual(source_file, "mock1")
        expected = np.ones((24, 18)) * 0.0
        self.assertTrue((labelled_np == expected).all())

    @patch("src.ETL.dataset.Path.open")
    @patch("src.ETL.dataset.find_matching_point")
    @patch("src.ETL.dataset.pickle.dump")
    def test_create_pickled_labeled_dataset(self, mock_dump, mock_find_matching_point, mock_open):
        mock_find_matching_point.return_value = (
            np.array([0.0]),
            0.1,
            0.1,
            "mock_file",
        )

        mock_labels = pd.DataFrame(
            {
                LON: [20, 40],
                LAT: [30, 50],
                CROP_PROB: [0.0, 1.0],
                START: ["2020-01-01", "2020-01-01"],
                END: ["2021-01-01", "2021-01-01"],
                TIF_PATHS: [[Path("tif1")], [Path("tif2"), Path("tif3")]],
                FEATURE_PATH: ["feature1", "feature2"],
            }
        )

        create_pickled_labeled_dataset(mock_labels)

        instances = [
            CropDataInstance(
                instance_lat=0.1,
                instance_lon=0.1,
                labelled_array=np.array([0.0]),
                source_file="mock_file",
            ),
            CropDataInstance(
                instance_lat=0.1,
                instance_lon=0.1,
                labelled_array=np.array([0.0]),
                source_file="mock_file",
            ),
        ]

        self.assertEqual(mock_dump.call_count, 2)
        self.assertEqual(mock_dump.call_args_list[0][0][0], instances[0])
        self.assertEqual(mock_dump.call_args_list[1][0][0], instances[1])
