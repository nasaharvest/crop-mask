from datetime import datetime
from unittest import TestCase

import numpy as np

from src.inference import Inference
from src.utils import data_dir

tif_name = "min_lat=14.9517_min_lon=-86.2507_max_lat=14.9531_max_lon=-86.2493_dates=2017-01-01_2018-12-31_all.tif"


class TestInference(TestCase):
    def test_start_date_from_str(self):
        actual_start_date = Inference.start_date_from_str(tif_name)
        expected_start_date = datetime(2017, 1, 1, 0, 0)
        self.assertEqual(actual_start_date, expected_start_date)

    def test_tif_to_np(self):
        x_np, flat_lat, flat_lon = Inference._tif_to_np(
            data_dir / tif_name, start_date=datetime(2017, 1, 1, 0, 0)
        )
        self.assertEqual(x_np.shape, (289, 24, 18))
        self.assertEqual(flat_lat.shape, (289,))
        self.assertEqual(flat_lon.shape, (289,))

    def test_combine_predictions(self):
        flat_lat = np.array([14.95313164, 14.95313164, 14.95313164, 14.95313164, 14.95313164])
        flat_lon = np.array([-86.25070894, -86.25061911, -86.25052928, -86.25043945, -86.25034962])
        batch_predictions = np.array(
            [[0.43200156], [0.55286014], [0.5265], [0.5236109], [0.4110847]]
        )
        xr_predictions = Inference._combine_predictions(
            flat_lat=flat_lat, flat_lon=flat_lon, batch_predictions=batch_predictions
        )

        # Check size
        self.assertEqual(xr_predictions.dims["lat"], 1)
        self.assertEqual(xr_predictions.dims["lon"], 5)

        # Check coords
        self.assertTrue((xr_predictions.lat.values == flat_lat[0:1]).all())
        self.assertTrue((xr_predictions.lon.values == flat_lon).all())

        # Check all predictions between 0 and 1
        self.assertTrue(xr_predictions.min() >= 0)
        self.assertTrue(xr_predictions.max() <= 1)
