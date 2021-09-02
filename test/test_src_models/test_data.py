from unittest import TestCase

import numpy as np

from src.models.data import CropDataset


class TestData(TestCase):
    def test_update_normalizing_values(self):
        norm_dict = {"n": 0}
        array = np.array([[1, 2, 3], [2, 3, 4]])
        CropDataset._update_normalizing_values(norm_dict, array)
        self.assertEqual(norm_dict["n"], array.shape[0])
        self.assertTrue(np.allclose(norm_dict["mean"], np.array([1.5, 2.5, 3.5])))
        self.assertTrue(np.allclose(norm_dict["M2"], np.array([0.5, 0.5, 0.5])))

        array2 = np.array([[3, 4, 5], [4, 5, 6]])
        CropDataset._update_normalizing_values(norm_dict, array2)
        self.assertEqual(norm_dict["n"], array.shape[0] + array2.shape[0])
        self.assertTrue(np.allclose(norm_dict["mean"], np.array([2.5, 3.5, 4.5])))
        self.assertTrue(np.allclose(norm_dict["M2"], np.array([5.0, 5.0, 5.0])))

    def test_calculate_normalizing_dict(self):
        norm_dict = {"n": 4, "mean": np.array([2.5, 3.5, 4.5]), "M2": np.array([5.0, 5.0, 5.0])}
        normalizing_dict = CropDataset._calculate_normalizing_dict(norm_dict)
        self.assertTrue(np.allclose(normalizing_dict["mean"], norm_dict["mean"]))
        self.assertTrue(
            np.allclose(normalizing_dict["std"], np.array([1.29099445, 1.29099445, 1.29099445]))
        )
        empty_norm_dict = {}
        normalizing_dict = CropDataset._calculate_normalizing_dict(empty_norm_dict)
        self.assertIsNone(normalizing_dict)
