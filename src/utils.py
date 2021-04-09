import torch
import numpy as np
import logging
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from pathlib import Path
import xarray as xr
import pandas as pd

from .constants import BANDS

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def process_filename(
    filename: str, include_extended_filenames: bool
) -> Optional[Tuple[str, datetime, datetime]]:
    r"""
    Given an exported sentinel file, process it to get the start
    and end dates of the data. This assumes the filename ends with '.tif'
    """
    date_format = "%Y-%m-%d"
    if filename[-4:] != ".tif":
        raise ValueError(f"Filename: {filename} must end with .tif")

    filename_components = filename[:-4].split("_")
    if len(filename_components) < 3:
        raise ValueError(
            f"Filename: {filename} must have an identifier, start, and end date separated by '_'"
        )

    identifier = filename_components[-3]
    start_date_str = filename_components[-2]
    end_date_str = filename_components[-1]

    start_date = datetime.strptime(start_date_str, date_format)

    try:
        end_date = datetime.strptime(end_date_str, date_format)
        return identifier, start_date, end_date

    except ValueError:
        if include_extended_filenames:
            end_list = end_date_str.split("-")
            end_year, end_month, end_day = (
                end_list[0],
                end_list[1],
                end_list[2],
            )

            # if we allow extended filenames, we want to
            # differentiate them too
            id_number = end_list[3]
            identifier = f"{identifier}-{id_number}"

            return (
                identifier,
                start_date,
                datetime(int(end_year), int(end_month), int(end_day)),
            )
        else:
            logger.warning(f"Unexpected filename {filename} - skipping")
            return None


def load_tif(filepath: Path, start_date: datetime, days_per_timestep: int) -> xr.DataArray:
    r"""
    The sentinel files exported from google earth have all the timesteps
    concatenated together. This function loads a tif files and splits the
    timesteps
    """

    # this mirrors the eo-learn approach
    # also, we divide by 10,000, to remove the scaling factor
    # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
    da = xr.open_rasterio(filepath).rename("FEATURES") / 10000

    da_split_by_time: List[xr.DataArray] = []

    bands_per_timestep = len(BANDS)
    num_bands = len(da.band)

    assert (
        num_bands % bands_per_timestep == 0
    ), f"Total number of bands not divisible by the expected bands per timestep"

    cur_band = 0
    while cur_band + bands_per_timestep <= num_bands:
        time_specific_da = da.isel(band=slice(cur_band, cur_band + bands_per_timestep))
        time_specific_da["band"] = range(bands_per_timestep)
        da_split_by_time.append(time_specific_da)
        cur_band += bands_per_timestep

    timesteps = [
        start_date + timedelta(days=days_per_timestep) * i for i in range(len(da_split_by_time))
    ]

    combined = xr.concat(da_split_by_time, pd.Index(timesteps, name="time"))
    combined.attrs["band_descriptions"] = BANDS

    return combined
