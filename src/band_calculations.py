import warnings
import numpy as np
from .constants import BANDS
from typing import Optional


def _calculate_difference_index(
    input_array: np.ndarray, num_dims: int, band_1: str, band_2: str
) -> np.ndarray:
    if num_dims == 2:
        band_1_np = input_array[:, BANDS.index(band_1)]
        band_2_np = input_array[:, BANDS.index(band_2)]
    elif num_dims == 3:
        band_1_np = input_array[:, :, BANDS.index(band_1)]
        band_2_np = input_array[:, :, BANDS.index(band_2)]
    else:
        raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        # suppress the following warning
        # RuntimeWarning: invalid value encountered in true_divide
        # for cases where near_infrared + red == 0
        # since this is handled in the where condition
        ndvi = np.where(
            (band_1_np + band_2_np) > 0,
            (band_1_np - band_2_np) / (band_1_np + band_2_np),
            0,
        )
    return np.append(input_array, np.expand_dims(ndvi, -1), axis=-1)


def _calculate_ndvi(input_array: np.ndarray, num_dims: int = 2) -> np.ndarray:
    r"""
    Given an input array of shape [timestep, bands] or [batches, timesteps, bands]
    where bands == len(BANDS), returns an array of shape
    [timestep, bands + 1] where the extra band is NDVI,
    (b08 - b04) / (b08 + b04)
    """

    return _calculate_difference_index(input_array, num_dims, "B8", "B4")


def _calculate_ndwi(input_array: np.ndarray, num_dims: int = 2) -> np.ndarray:
    r"""
    Given an input array of shape [timestep, bands] or [batches, timesteps, bands]
    where bands == len(BANDS), returns an array of shape
    [timestep, bands + 1] where the extra band is NDVI,
    (b03 - b8A) / (b3 + b8a)
    """
    return _calculate_difference_index(input_array, num_dims, "B3", "B8A")


def _maxed_nan_to_num(
    array: np.ndarray, nan: float, max_ratio: Optional[float] = None
) -> Optional[np.ndarray]:
    if max_ratio is not None:
        num_nan = np.count_nonzero(np.isnan(array))
        if (num_nan / array.size) > max_ratio:
            return None
    return np.nan_to_num(array, nan=nan)


def process_bands(
    x_np,
    nan_fill: float,
    max_nan_ratio: Optional[float] = None,
    add_ndvi: bool = False,
    add_ndwi: bool = False,
    num_dims: int = 2,
) -> Optional[np.ndarray]:
    if add_ndvi:
        x_np = _calculate_ndvi(x_np, num_dims=num_dims)
    if add_ndwi:
        x_np = _calculate_ndwi(x_np, num_dims=num_dims)

    x_np = _maxed_nan_to_num(x_np, nan=nan_fill, max_ratio=max_nan_ratio)

    return x_np
