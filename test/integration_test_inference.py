from unittest import TestCase

import torch

from src.inference import Inference
from src.utils import get_dvc_dir, data_dir
from src.models.model import Model

model_name = "Ethiopia_Tigray_2020"
tif_file = (
    "min_lat=14.9517_min_lon=-86.2507_max_lat=14.9531_"
    + "max_lon=-86.2493_dates=2017-01-01_2018-12-31_all.tif"
)


class TestIntegrationInference(TestCase):
    def test_inference_run_with_ckpt_model(self):
        model_dir = get_dvc_dir("models")
        model = Model.load_from_checkpoint(model_dir / f"{model_name}.ckpt")
        model.eval()

        inference = Inference(model=model)
        xr_predictions = inference.run(local_path=data_dir / tif_file)

        # Check size
        self.assertEqual(xr_predictions.dims["lat"], 17)
        self.assertEqual(xr_predictions.dims["lon"], 17)

        # Check all predictions between 0 and 1
        self.assertTrue(xr_predictions.min() >= 0)
        self.assertTrue(xr_predictions.max() <= 1)

    def test_inference_run_with_jit_model(self):

        model_dir = get_dvc_dir("models")
        model = torch.jit.load(str(model_dir / f"{model_name}.pt"))
        model.eval()

        inference = Inference(model=model)
        xr_predictions = inference.run(local_path=data_dir / tif_file)

        # Check size
        self.assertEqual(xr_predictions.dims["lat"], 17)
        self.assertEqual(xr_predictions.dims["lon"], 17)

        # Check all predictions between 0 and 1
        self.assertTrue(xr_predictions.min() >= 0)
        self.assertTrue(xr_predictions.max() <= 1)
