from unittest import TestCase

import numpy as np
import xarray as xr

from src.utils import find_nearest, distance, distance_point_from_center


class TestUtils(TestCase):
    def test_find_nearest(self):
        val, idx = find_nearest([1.0, 2.0, 3.0, 4.0, 5.0], 4.0)
        self.assertEqual(val, 4.0)
        self.assertEqual(idx, 3)

        val, idx = find_nearest(xr.DataArray([1.0, 2.0, 3.0, -4.0, -5.0]), -1.0)
        self.assertEqual(val, 1.0)
        self.assertEqual(idx, 0)

    def test_distance(self):
        self.assertEqual(distance(0, 0, 0.01, 0.01), 1.5725337265584898)
        self.assertEqual(distance(0, 0, 0.01, 0), 1.1119492645167193)
        self.assertEqual(distance(0, 0, 0, 0.01), 1.1119492645167193)

    def test_distance_point_from_center(self):
        tif = xr.DataArray(attrs={"x": np.array([25, 35, 45]), "y": np.array([35, 45, 55])})
        self.assertEqual(distance_point_from_center(0, 0, tif), 2.0)
        self.assertEqual(distance_point_from_center(0, 1, tif), 1.0)
        self.assertEqual(distance_point_from_center(1, 1, tif), 0.0)
        self.assertEqual(distance_point_from_center(2, 1, tif), 1.0)
