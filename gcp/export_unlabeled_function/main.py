import datetime
import logging
import json
import tempfile
import ee
import os

from flask import abort, Request
from google.cloud import secretmanager
from google.cloud import pubsub_v1
from google.cloud import firestore
from pathlib import Path

from src.ETL.ee_boundingbox import BoundingBox
from src.ETL.ee_exporter import Season, RegionExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = firestore.Client()


def get_ee_credentials():
    logger.info("Fetching credentials")
    filename = tempfile.gettempdir() + "/creds.json"
    if not Path(filename).exists():
        client = secretmanager.SecretManagerServiceClient()
        name = client.secret_version_path("670160663426", "google_application_credentials", "2")
        response = client.access_secret_version(name=name)
        payload = response.payload.data.decode("UTF-8")
        output_dict = json.loads(payload)
        with open(filename, "w") as outfile:
            json.dump(output_dict, outfile)

    service_account = "nasa-harvest@appspot.gserviceaccount.com"
    logger.info(f"Setting ee credentials for {service_account}")
    credentials = ee.ServiceAccountCredentials(service_account, filename)
    logger.info("Credentials set")
    return credentials


def export_unlabeled(request: Request):
    """
    Args:
         request (Request): Event payload.
    """
    dest_bucket = os.environ.get("DEST_BUCKET")

    request_json = request.get_json(silent=True)
    logger.info(request_json)

    bbox_keys = ["min_lon", "max_lon", "min_lat", "max_lat"]
    for key in ["model_name", "dataset_name", "season"] + bbox_keys:
        if key not in request_json:
            abort(400, description=f"{key} is missing from request_json: {request_json}")

    model_name = request_json["model_name"]
    sentinel_dataset = request_json["dataset_name"]

    try:
        season = Season(request_json["season"])
    except ValueError as e:
        logger.exception(e)
        abort(400, description=str(e))

    file_dimensions = request_json.get("file_dimensions", None)
    try:
        credentials = get_ee_credentials()
        bbox_args = {k: v for k, v in request_json.items() if k in bbox_keys}
        bbox = BoundingBox(**bbox_args)
        ids = RegionExporter(
            dest_bucket=dest_bucket,
            model_name=model_name,
            sentinel_dataset=sentinel_dataset,
            credentials=credentials,
            file_dimensions=file_dimensions,
        ).export(region_bbox=bbox, season=season, metres_per_polygon=50000)

        id = f"{model_name}_{sentinel_dataset}"
        data = {
            "bbox": bbox.url,
            "model_name": model_name,
            "dataset_name": sentinel_dataset,
            "complete_ee_tasks": 0,
            "total_ee_tasks": len(ids),
            "start_time": str(datetime.datetime.now()),
            "ee_files_exported": 0,
            "predictions_made": 0
        }
        db.collection('crop-mask-runs').document(id).set(data)

    except Exception as e:
        logger.exception(e)
        abort(500, description=str(e))

    return {
        "message": f"Started export of {sentinel_dataset}",
        "status": "https://us-central1-nasa-harvest.cloudfunctions.net/ee-status",
        "bbox": bbox.url,
    }


def get_status(request: Request):
    """
    Args:
         request (Request): Event payload.
    """
    logger.info(f"Called with: {request.args}")
    tasks = []
    try:
        credentials = get_ee_credentials()
        ee.Initialize(credentials)
        logger.info("Looping through ee task list:")
        tasks = [t.status() for t in ee.batch.Task.list() if t.state != "COMPLETED"]

    except Exception as e:
        logger.exception(e)
        abort(500, description=str(e))

    response = {"amount": len(tasks), "tasks": tasks}

    if request.args.get("pubsub"):
        logger.info("Publishing")
        publisher = pubsub_v1.PublisherClient()
        topic = "projects/nasa-harvest/topics/crop-maps"
        future = publisher.publish(topic, bytes(str(response), "UTF-8"))
        logger.info(future.result())

    return response
