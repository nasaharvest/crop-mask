import sys
import tempfile
import time

from pathlib import Path
from google.cloud import storage
from ts.torch_handler.base_handler import BaseHandler

from src.inference import Inference

temp_dir = tempfile.gettempdir()
dest_bucket_name = "crop-mask-preds"


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
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

    def inference(self, data, *args, **kwargs):
        print(data)
        print("HANDLER: Starting preprocessing")
        try:
            uri = next(q["uri"].decode() for q in data if "uri" in q)
        except Exception:
            raise ValueError("'uri' not found.")

        local_path = self.download_file(uri)
        uri_as_path = Path(uri)
        local_dest_path = Path(tempfile.gettempdir() + f"/pred_{uri_as_path.stem}.nc")

        self.inference_module.run(local_path=local_path, dest_path=local_dest_path)
        print("HANDLER: Completed inference")

        cloud_dest_parent = "/".join(uri_as_path.parts[2:-1])
        cloud_dest_path_str = f"{cloud_dest_parent}/{local_dest_path.name}"
        dest_bucket = storage.Client().get_bucket(dest_bucket_name)
        dest_blob = dest_bucket.blob(cloud_dest_path_str)

        dest_blob.upload_from_filename(str(local_dest_path))
        print(f"HANDLER: Uploaded to gs://{dest_bucket_name}/{cloud_dest_path_str}")
        return [{"src_uri": uri, "dest_uri": f"gs://{dest_bucket_name}/{cloud_dest_path_str}"}]
