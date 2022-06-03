from cropharvest.engineer import Engineer
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import re
import torch

from typing import Dict, List, Optional, Tuple


class Inference:
    """
    Class for running inference using either a pytorch checkpoint model or
    a pytorch jit model on a single tif file.
    """

    def __init__(self, model, device: Optional[torch.device] = None):
        self.model = model
        self.device = device
        self.normalizing_dict: Dict[str, np.ndarray] = {
            k: np.array(v) for k, v in self.model.normalizing_dict_jit.items()
        }
        self.batch_size: int = self.model.batch_size

    @staticmethod
    def start_date_from_str(uri: str) -> datetime:
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", str(uri))
        if len(dates) < 2:
            raise ValueError(f"{uri} should have start and end date")
        return datetime.strptime(dates[0], "%Y-%m-%d")

    @staticmethod
    def _tif_to_np(
        local_path: Path,
        start_date: datetime,
        normalizing_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        da, slope = Engineer.load_tif(local_path, start_date=start_date, num_timesteps=None)

        # Process remote sensing data
        x_np = da.values
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
        x_np = np.moveaxis(x_np, -1, 0)
        x_np = Engineer.calculate_ndvi(x_np)
        x_np = Engineer.remove_bands(x_np)
        x_np = Engineer.fillna(x_np, slope)
        if normalizing_dict is not None:
            x_np = (x_np - normalizing_dict["mean"]) / normalizing_dict["std"]

        # Get lat lons
        lon, lat = np.meshgrid(da.x.values, da.y.values)
        flat_lat, flat_lon = (
            np.squeeze(lat.reshape(-1, 1), -1),
            np.squeeze(lon.reshape(-1, 1), -1),
        )
        return x_np, flat_lat, flat_lon

    @staticmethod
    def _combine_predictions(
        flat_lat: np.ndarray, flat_lon: np.ndarray, batch_predictions: List[np.ndarray]
    ) -> xr.Dataset:
        print("HANDLER: Combining predictions")
        all_preds = np.concatenate(batch_predictions, axis=0)
        if len(all_preds.shape) == 1:
            all_preds = np.expand_dims(all_preds, axis=-1)

        data_dict: Dict[str, np.ndarray] = {"lat": flat_lat, "lon": flat_lon}
        for i in range(all_preds.shape[1]):
            prediction_label = f"prediction_{i}"
            data_dict[prediction_label] = all_preds[:, i]
        return pd.DataFrame(data=data_dict).set_index(["lat", "lon"]).to_xarray()

    def _on_single_batch(self, batch_x_np: np.ndarray) -> np.ndarray:
        batch_x = torch.from_numpy(batch_x_np).float()
        if self.device is not None:
            batch_x = batch_x.to(self.device)
        with torch.no_grad():
            _, batch_preds_local = self.model.forward(batch_x)
        return batch_preds_local.cpu().numpy()

    def run(
        self,
        local_path: Path,
        start_date: Optional[datetime] = None,
        dest_path: Optional[Path] = None,
    ) -> xr.Dataset:
        if start_date is None:
            start_date = self.start_date_from_str(str(local_path))
        x_np, flat_lat, flat_lon = self._tif_to_np(local_path, start_date, self.normalizing_dict)
        batches = [
            x_np[i : i + self.batch_size] for i in range(0, (x_np.shape[0] - 1), self.batch_size)
        ]
        batch_predictions = [self._on_single_batch(b) for b in batches]
        combined_pred = self._combine_predictions(flat_lat, flat_lon, batch_predictions)
        if dest_path is not None:
            combined_pred.to_netcdf(dest_path)
        return combined_pred
