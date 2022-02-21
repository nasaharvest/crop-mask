import sys
import tempfile
import time

from pathlib import Path
from google.cloud import storage  # type: ignore
from ts.torch_handler.base_handler import BaseHandler
from typing import Tuple

from inference import Inference

temp_dir = tempfile.gettempdir()
dest_bucket_name = "crop-mask-preds"


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    The basehandler calls initialize() on server start up and
    preprocess(), inference(), and postprocess() on each request.
    """

    def __init__(self):
        print("HANDLER: Starting up handler")
        super().__init__()

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

    def initialize(self, context):
        super().initialize(context)
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        sys.path.append(model_dir)
        self.inference_module = Inference(model=self.model)

    def preprocess(self, data) -> str:
        print(data)
        print("HANDLER: Starting preprocessing")
        try:
            uri = next(q["uri"].decode() for q in data if "uri" in q)
        except Exception:
            raise ValueError("'uri' not found.")

        return uri

    def inference(self, data, *args, **kwargs) -> Tuple[str, str]:
        uri = data
        local_path = self.download_file(uri)
        uri_as_path = Path(uri)
        local_dest_path = Path(tempfile.gettempdir() + f"/pred_{uri_as_path.stem}.nc")

        print("HANDLER: Starting inference")
        start_date = self.inference_module.start_date_from_str(uri)
        print(f"HANDLER: Start date: {start_date}")
        self.inference_module.run(
            local_path=local_path, start_date=start_date, dest_path=local_dest_path
        )
        print("HANDLER: Completed inference")

        cloud_dest_parent = "/".join(uri_as_path.parts[2:-1])
        cloud_dest_path_str = f"{cloud_dest_parent}/{local_dest_path.name}"
        dest_bucket = storage.Client().get_bucket(dest_bucket_name)
        dest_blob = dest_bucket.blob(cloud_dest_path_str)
        dest_blob.upload_from_filename(str(local_dest_path))
        dest_uri = f"gs://{dest_bucket_name}/{cloud_dest_path_str}"
        print(f"HANDLER: Uploaded to {dest_uri}")
        return uri, dest_uri

    def postprocess(self, data):
        uri, dest_uri = data
        return [{"src_uri": uri, "dest_uri": dest_uri}]
