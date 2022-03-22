from pathlib import Path

import logging
import time
import os
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hello_gcs(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    logger.info(event)
    bucket_name = event["bucket"]
    blob_name = event["name"]
    src_path = f"gs://{bucket_name}/{blob_name}"
    logger.info(src_path)

    available_models = os.environ.get("MODELS").split(" ")
    model_name = Path(blob_name).parts[0]
    logger.info(f"Extracted model_name: {model_name}")
    if model_name not in available_models:
        return ValueError(f"{model_name} not available in {available_models}")

    host = os.environ.get("INFERENCE_HOST")
    url = f"{host}/predictions/{model_name}"
    logger.info(url)
    data = {"uri": src_path}
    for attempt in range(3):
        logger.info("Sending request")
        response = requests.post(url, data=data)
        logger.info("Received response")
        logger.info(response.status_code)
        if response.status_code == 200:
            logger.info(response.json())
            break
        logger.error(f"Failed response: {response.raw}")
        time.sleep(5)
