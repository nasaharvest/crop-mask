import numpy as np
import pandas as pd
import rasterio
import re
import sys
import torch
import tempfile
import warnings
import xarray as xr

from datetime import datetime, timedelta
from pathlib import Path
from rasterio.session import GSSession
from google.cloud import storage
from ts.torch_handler.base_handler import BaseHandler
from typing import cast, Dict, List, Optional, Tuple


BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]

creds_path = "/root/.config/gcloud/creds.json"


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """
    batch_size = 64
    bands_to_remove = ["B1", "B10"]

    def __init__(self):
        super().__init__()
        self.gs_session = GSSession(creds_path)
        storage_client = storage.Client.from_service_account_json(creds_path)
        self.dest_bucket_name = "crop-mask-preds-unmerged"
        self.dest_bucket = storage_client.get_bucket(self.dest_bucket_name)
        self.normalizing_dict = None

    @staticmethod
    def load_tif_gcs(gs_session, gcs_path: str, start_date: datetime, days_per_timestep: int) -> xr.DataArray:
        r"""
        The sentinel files exported from google earth have all the timesteps
        concatenated together. This function loads a tif files and splits the
        timesteps
        """

        # this mirrors the eo-learn approach
        # also, we divide by 10,000, to remove the scaling factor
        # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
        print(f"Loading {gcs_path} from gcloud storage")
        with rasterio.Env(gs_session):
            da = xr.open_rasterio(gcs_path).rename("FEATURES") / 10000
        print(f"Successfully loaded {gcs_path} from gcloud storage")
        da_split_by_time: List[xr.DataArray] = []

        bands_per_timestep = len(BANDS)
        num_bands = len(da.band)

        assert (
                num_bands % bands_per_timestep == 0
        ), "Total number of bands not divisible by the expected bands per timestep"

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
        print("Returning tif converted into xr DataArray")
        return combined

    @staticmethod
    def combine_predictions(x, predictions):
        print("Combining predictions")
        all_preds = np.concatenate(predictions, axis=0)
        if len(all_preds.shape) == 1:
            all_preds = np.expand_dims(all_preds, axis=-1)

        lon, lat = np.meshgrid(x.x.values, x.y.values)
        flat_lat, flat_lon = (
            np.squeeze(lat.reshape(-1, 1), -1),
            np.squeeze(lon.reshape(-1, 1), -1),
        )
        data_dict: Dict[str, np.ndarray] = {"lat": flat_lat, "lon": flat_lon}
        for i in range(all_preds.shape[1]):
            prediction_label = f"prediction_{i}"
            data_dict[prediction_label] = all_preds[:, i]
        return pd.DataFrame(data=data_dict).set_index(["lat", "lon"]).to_xarray()

    @classmethod
    def remove_bands(cls, x: np.ndarray) -> np.ndarray:
        """
        Expects the input to be of shape [timesteps, bands]
        """
        indices_to_remove: List[int] = []
        for band in cls.bands_to_remove:
            indices_to_remove.append(BANDS.index(band))

        bands_index = 1 if len(x.shape) == 2 else 2
        indices_to_keep = [i for i in range(x.shape[bands_index]) if i not in indices_to_remove]
        if len(x.shape) == 2:
            # timesteps, bands
            return x[:, indices_to_keep]
        else:
            # batches, timesteps, bands
            return x[:, :, indices_to_keep]

    @staticmethod
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

    @staticmethod
    def _maxed_nan_to_num(
            array: np.ndarray, nan: float, max_ratio: Optional[float] = None
    ) -> Optional[np.ndarray]:
        if max_ratio is not None:
            num_nan = np.count_nonzero(np.isnan(array))
            if (num_nan / array.size) > max_ratio:
                return None
        return np.nan_to_num(array, nan=nan)

    @staticmethod
    def process_bands(
            x_np,
            nan_fill: float,
            max_nan_ratio: Optional[float] = None,
            add_ndvi: bool = False,
            add_ndwi: bool = False,
            num_dims: int = 2,
    ) -> Optional[np.ndarray]:
        print("Processing bands")
        if add_ndvi:
            x_np = ModelHandler._calculate_difference_index(x_np, num_dims, "B8", "B4")
        if add_ndwi:
            x_np = ModelHandler._calculate_difference_index(x_np, num_dims, "B3", "B8A")
        x_np = ModelHandler._maxed_nan_to_num(x_np, nan=nan_fill, max_ratio=max_nan_ratio)
        return x_np

    def initialize(self, context):
        super().initialize(context)
        self.normalizing_dict = {k: np.array(v) for k,v in self.model.normalizing_dict_jit.items()}
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        sys.path.append(model_dir)

    def preprocess(self, data) -> Tuple[str, xr.DataArray]:
        print(data)
        print("Starting preprocessing")
        try:
            uri = next(q["uri"].decode() for q in data if "uri" in q)
        except Exception:
            raise ValueError("'uri' not input query")

        dates = re.findall(r"\d{4}-\d{2}-\d{2}", Path(uri).stem)
        if len(dates) != 2:
            raise ValueError(f"{uri} should have start and end date")
        start_date_str, _ = dates
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        x = self.load_tif_gcs(self.gs_session, uri, days_per_timestep=30, start_date=start_date)
        print("Completed preprocessing")
        return uri, x

    def inference_on_single_batch(self, batch_x_np):
        batch_x_np = self.remove_bands(batch_x_np)
        batch_x = torch.from_numpy(batch_x_np).float()

        if self.device is not None:
            batch_x = batch_x.to(self.device)

        with torch.no_grad():
            _, batch_preds = self.model.forward(batch_x)
            # back to the CPU, if necessary
            batch_preds = batch_preds.cpu()

        return cast(torch.Tensor, batch_preds).numpy()

    def inference(self, data, *args, **kwargs) -> Tuple[str, xr.Dataset]:
        print("Starting inference")
        uri, x = data
        x_np = x.values
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
        x_np = np.moveaxis(x_np, -1, 0)
        x_np = self.process_bands(x_np, nan_fill=0, add_ndvi=True, add_ndwi=False, num_dims=3)
        if self.normalizing_dict is not None:
            x_np = (x_np - self.normalizing_dict["mean"]) / self.normalizing_dict["std"]
        print("Splitting into batches")
        batches = [x_np[i: i + self.batch_size] for i in range(0, (x_np.shape[0] - 1), self.batch_size)]
        print(f"Doing inference on {len(batches)} batches")
        predictions = [self.inference_on_single_batch(b) for b in batches]
        combined_pred = self.combine_predictions(x, predictions)
        print("Completed inference")
        return uri, combined_pred

    def postprocess(self, data) -> List[Dict[str, str]]:
        print("Starting postprocessing")
        uri, preds = data
        uri_as_path = Path(uri)
        local_dest_path = Path(tempfile.gettempdir() + f"/pred_{uri_as_path.stem}.nc")
        cloud_dest_path_str = f"{uri_as_path.parent.name}/{local_dest_path.name}"

        preds.to_netcdf(local_dest_path)
        dest_blob = self.dest_bucket.blob(cloud_dest_path_str)

        dest_blob.upload_from_filename(str(local_dest_path))
        print(f"Uploaded to gs://{self.dest_bucket_name}/{cloud_dest_path_str}")
        return [{
            "src_uri": uri,
            "dest_uri": f"gs://{self.dest_bucket_name}/{cloud_dest_path_str}"
        }]
