from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

from src.engineer.base import BaseEngineer

from typing import cast, Dict, Optional, Tuple


@dataclass
class TestInstance:
    x: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def tif_to_np(
    path_to_dataset: Path,
    add_ndvi: bool,
    add_ndwi: bool,
    nan: float,
    normalizing_dict: Optional[Dict[str, np.ndarray]],
    days_per_timestep: int,
) -> TestInstance:

    _, start_date, _ = cast(
        Tuple[str, datetime, datetime],
        BaseEngineer.process_filename(path_to_dataset.name, include_extended_filenames=True),
    )

    x = BaseEngineer.load_tif(
        path_to_dataset, days_per_timestep=days_per_timestep, start_date=start_date
    )

    lon, lat = np.meshgrid(x.x.values, x.y.values)
    flat_lat, flat_lon = (
        np.squeeze(lat.reshape(-1, 1), -1),
        np.squeeze(lon.reshape(-1, 1), -1),
    )

    x_np = x.values
    x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
    x_np = np.moveaxis(x_np, -1, 0)

    if add_ndvi:
        x_np = BaseEngineer.calculate_ndvi(x_np, num_dims=3)
    if add_ndwi:
        x_np = BaseEngineer.calculate_ndwi(x_np, num_dims=3)

    x_np = BaseEngineer.maxed_nan_to_num(x_np, nan=nan)

    if normalizing_dict is not None:
        x_np = (x_np - normalizing_dict["mean"]) / normalizing_dict["std"]

    return TestInstance(x=x_np, lat=flat_lat, lon=flat_lon)


def preds_to_xr(predictions: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> xr.Dataset:

    data_dict: Dict[str, np.ndarray] = {"lat": lats, "lon": lons}

    for prediction_idx in range(predictions.shape[1]):
        prediction_label = f"prediction_{prediction_idx}"
        data_dict[prediction_label] = predictions[:, prediction_idx]

    return pd.DataFrame(data=data_dict).set_index(["lat", "lon"]).to_xarray()
