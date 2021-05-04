from unittest import TestCase
from pathlib import Path
import numpy as np
import tempfile
import shutil
import xarray as xr

import sys

sys.path.append("..")

from scripts.predict import run_inference  # noqa: E402
from utils import get_dvc_dir  # noqa: E402


class IntegrationTestPredict(TestCase):
    """Tests the predict script"""

    temp_dir: Path = Path("")

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    @staticmethod
    def load_first_file_in_dir(dir_path: Path) -> xr.Dataset:
        first_file = next(dir_path.rglob("*.nc"))
        return xr.load_dataset(first_file)

    def test_all_models(self):
        test_data_dir = get_dvc_dir("test")
        model_dir = get_dvc_dir("models")

        for model in model_dir.rglob("*.ckpt"):
            model_name = model.stem
            print(f"Testing model: {model_name}")

            expected_dir = Path(test_data_dir / "expected")
            predict_dir = Path(self.temp_dir / model_name)
            predict_dir.mkdir(parents=True, exist_ok=True)

            run_inference(
                local_path_to_tif_files=str(test_data_dir / "input"),
                model_name=model_name,
                data_dir=str(Path(__file__).parent.parent / "data"),
                predict_dir=str(predict_dir),
                predict_with_forecaster=True,
                predict_without_forecaster=False,
            )

            expected = self.load_first_file_in_dir(expected_dir)
            predicted = self.load_first_file_in_dir(predict_dir)

            self.assertEqual(
                expected, predicted, "Expected and actual prediction are not the same."
            )

            xr.testing.assert_allclose(expected, predicted, atol=1)

            mean_difference = np.mean(np.abs(expected - predicted)).data_vars["prediction_0"]
            print(f"Expected - predicted, mean difference: {mean_difference}")
            self.assertLessEqual(mean_difference, 0.5)
