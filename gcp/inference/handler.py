import numpy as np
import pandas as pd
import re
import sys
import torch
import tempfile
import time
import xarray as xr

from cropharvest.engineer import Engineer
from datetime import datetime
from pathlib import Path
from google.cloud import storage
from ts.torch_handler.base_handler import BaseHandler
from typing import cast, Dict, List, Tuple

temp_dir = tempfile.gettempdir()
dest_bucket_name = "crop-mask-preds"


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    batch_size = 64

    def __init__(self):
        print("HANDLER: Starting up handler")
        super().__init__()
        self.normalizing_dict = None

    @staticmethod
    def combine_predictions(x, predictions):
        print("HANDLER: Combining predictions")
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

    def download_file(self, uri: str):
        uri_as_path = Path(uri)
        bucket_name = uri_as_path.parts[1]
        file_name = "/".join(uri_as_path.parts[2:])
        bucket = storage.Client().bucket(bucket_name)
        retries = 3
        blob = bucket.blob(file_name)
        for i in range(retries + 1):
            if blob.exists():
                print(f"HANDLER: Verified {uri} exists.")
                break
            if i == retries:
                raise ValueError(f"HANDLER ERROR: {uri} does not exist.")

            print(f"HANDLER: {uri} does not yet exist, sleeping for 5 seconds and trying again.")
            time.sleep(5)
        local_path = f"{tempfile.gettempdir()}/{uri_as_path.name}"
        blob.download_to_filename(local_path)
        if not Path(local_path).exists():
            raise FileExistsError(f"HANDLER: {uri} from storage was not downloaded")
        print(f"HANDLER: Verified file downloaded to {local_path}")
        return local_path

    def get_start_date(self, uri):
        uri_as_path = Path(uri)
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", uri_as_path.stem)
        if len(dates) != 2:
            raise ValueError(f"{uri} should have start and end date")
        start_date_str, _ = dates
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        return start_date

    def initialize(self, context):
        super().initialize(context)
        self.normalizing_dict = {k: np.array(v) for k, v in self.model.normalizing_dict_jit.items()}
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        sys.path.append(model_dir)

    def preprocess(self, data) -> Tuple[str, xr.DataArray]:
        print(data)
        print("HANDLER: Starting preprocessing")
        try:
            uri = next(q["uri"].decode() for q in data if "uri" in q)
        except Exception:
            raise ValueError("'uri' not found.")

        start_date = self.get_start_date(uri)
        local_path = self.download_file(uri)
        da, slope = Engineer.load_tif(local_path, start_date=start_date)

        Path(local_path).unlink()
        print("HANDLER: Completed preprocessing")
        return uri, da, slope

    def inference_on_single_batch(self, batch_x_np):
        batch_x = torch.from_numpy(batch_x_np).float()

        if self.device is not None:
            batch_x = batch_x.to(self.device)

        with torch.no_grad():
            _, batch_preds = self.model.forward(batch_x)
            # back to the CPU, if necessary
            batch_preds = batch_preds.cpu()

        return cast(torch.Tensor, batch_preds).numpy()

    def inference(self, data, *args, **kwargs) -> Tuple[str, xr.Dataset]:
        print("HANDLER: Starting inference")
        uri, da, slope = data
        x_np = da.values
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
        x_np = np.moveaxis(x_np, -1, 0)
        x_np = Engineer.calculate_ndvi(x_np)
        x_np = Engineer.remove_bands(x_np)
        x_np = Engineer.fillna(x_np, slope)
        if self.normalizing_dict is not None:
            x_np = (x_np - self.normalizing_dict["mean"]) / self.normalizing_dict["std"]

        print("HANDLER: Splitting into batches")
        batches = [
            x_np[i : i + self.batch_size] for i in range(0, (x_np.shape[0] - 1), self.batch_size)
        ]
        print(f"HANDLER: Doing inference on {len(batches)} batches")
        predictions = [self.inference_on_single_batch(b) for b in batches]
        combined_pred = self.combine_predictions(da, predictions)
        print("HANDLER: Completed inference")
        return uri, combined_pred

    def postprocess(self, data) -> List[Dict[str, str]]:
        print("HANDLER: Starting postprocessing")
        uri, preds = data
        uri_as_path = Path(uri)

        local_dest_path = Path(tempfile.gettempdir() + f"/pred_{uri_as_path.stem}.nc")
        preds.to_netcdf(local_dest_path)

        cloud_dest_parent = "/".join(uri_as_path.parts[2:-1])
        cloud_dest_path_str = f"{cloud_dest_parent}/{local_dest_path.name}"
        dest_bucket = storage.Client().get_bucket(dest_bucket_name)
        dest_blob = dest_bucket.blob(cloud_dest_path_str)

        dest_blob.upload_from_filename(str(local_dest_path))
        print(f"HANDLER: Uploaded to gs://{dest_bucket_name}/{cloud_dest_path_str}")
        return [{"src_uri": uri, "dest_uri": f"gs://{dest_bucket_name}/{cloud_dest_path_str}"}]
