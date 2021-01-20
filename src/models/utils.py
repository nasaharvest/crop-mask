from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

from src.utils import process_filename, load_tif
from src.band_calculations import process_bands

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
        process_filename(path_to_dataset.name, include_extended_filenames=True),
    )

    x = load_tif(
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

    x_np = process_bands(x_np, nan_fill=nan, add_ndvi=add_ndvi, add_ndwi=add_ndwi, num_dims=3)

    if normalizing_dict is not None:
        x_np = (x_np - normalizing_dict["mean"]) / normalizing_dict["std"]

    return TestInstance(x=x_np, lat=flat_lat, lon=flat_lon)


def preds_to_xr(predictions: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> xr.Dataset:

    data_dict: Dict[str, np.ndarray] = {"lat": lats, "lon": lons}

    for prediction_idx in range(predictions.shape[1]):
        prediction_label = f"prediction_{prediction_idx}"
        data_dict[prediction_label] = predictions[:, prediction_idx]

    return pd.DataFrame(data=data_dict).set_index(["lat", "lon"]).to_xarray()
