from dataclasses import dataclass
from unittest import TestCase, skipIf
from pathlib import Path

import numpy as np
import tempfile
import pickle

from src.models.data import CropDataset

try:
    import pytorch_lightning
    TORCH_LIGHTNING_INSTALLED = True
except ImportError:
    TORCH_LIGHTNING_INSTALLED = False


@dataclass
class TempInstance:
    labelled_array: np.ndarray


class TestData(TestCase):
    @skipIf(not TORCH_LIGHTNING_INSTALLED, reason="No pytorch-lightning installed")
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

    @skipIf(not TORCH_LIGHTNING_INSTALLED, reason="No pytorch-lightning installed")
    def test_calculate_normalizing_dict(self):
        tempdir = tempfile.gettempdir()
        file_paths = [Path(tempdir, f"{i}.pkl") for i in range(3)]
        labelled_arrays = [
            np.array([[1 + i, 2 + i, 3 + i], [2 + i, 3 + i, 4 + i]]) for i in range(3)
        ]
        for labelled_array, file_path in zip(labelled_arrays, file_paths):
            with file_path.open("wb") as f:
                pickle.dump(TempInstance(labelled_array), f)

        normalizing_dict = CropDataset._calculate_normalizing_dict(file_paths)
        [p.unlink() for p in file_paths]
        # self.assertTrue(np.allclose(normalizing_dict["mean"], norm_dict["mean"]))
        self.assertTrue(np.allclose(normalizing_dict["mean"], np.array([2.5, 3.5, 4.5])))
        self.assertTrue(
            np.allclose(normalizing_dict["std"], np.array([1.04880885, 1.04880885, 1.04880885]))
        )
