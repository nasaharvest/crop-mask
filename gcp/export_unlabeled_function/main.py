import datetime
import logging
import json
import tempfile
import ee
import os

from datetime import date
from flask import abort, Request
from google.cloud import secretmanager
from google.cloud import firestore
from pathlib import Path

from src.ETL.ee_boundingbox import BoundingBox
from src.ETL.ee_exporter import RegionExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = firestore.Client()

BBOX_LIMIT = 25
SERVICE_ACCOUNT = "bsos-geog-harvest1@appspot.gserviceaccount.com"
PROJECT_ID = "1012768714927"
SECRET_NAME = "google_application_credentials_2"
SECRET_VERSION = "1"


def get_ee_credentials():
    logger.info("Fetching credentials")
    filename = tempfile.gettempdir() + "/creds.json"
    if not Path(filename).exists():
        client = secretmanager.SecretManagerServiceClient()
        name = client.secret_version_path(PROJECT_ID, SECRET_NAME, SECRET_VERSION)
        response = client.access_secret_version(name=name)
        payload = response.payload.data.decode("UTF-8")
        output_dict = json.loads(payload)
        with open(filename, "w") as outfile:
            json.dump(output_dict, outfile)

    logger.info(f"Setting ee credentials for {SERVICE_ACCOUNT}")
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, filename)
    logger.info("Credentials set")
    return credentials


def is_bbox_too_big(bbox: BoundingBox):
    lat_len = bbox.max_lat - bbox.min_lat
    lon_len = bbox.max_lon - bbox.min_lon
    return lat_len * lon_len > BBOX_LIMIT


def export_unlabeled(request: Request):
    """
    Args:
         request (Request): Event payload.
    """
    dest_bucket = os.environ.get("DEST_BUCKET")
    all_model_names = os.environ.get("MODELS").split(" ")

    request_json = request.get_json(silent=True)
    logger.info(request_json)

    bbox_keys = ["min_lon", "max_lon", "min_lat", "max_lat"]
    for key in ["model_name", "dataset_name", "year"] + bbox_keys:
        if key not in request_json:
            abort(400, description=f"{key} is missing from request_json: {request_json}")

    model_name = request_json["model_name"]
    if model_name not in all_model_names:
        abort(400, description=f"{model_name} not found in: {all_model_names}")

    sentinel_dataset = request_json["dataset_name"]

    start_year = request_json["year"]
    start_date = date(start_year, 4, 21)  # Made to match default end date from processor
    num_timesteps = 12
    if date(start_year + 1, 4, 21) > date.today():
        if "num_timesteps" not in request_json:
            abort(
                400,
                description=f"End date is in the future so num_timesteps must be "
                "set in request_json: {request_json}",
            )
        num_timesteps = request_json["num_timesteps"]

    file_dimensions = request_json.get("file_dimensions", 256)
    try:
        credentials = get_ee_credentials()
        bbox_args = {k: v for k, v in request_json.items() if k in bbox_keys}
        bbox = BoundingBox(**bbox_args)

        if is_bbox_too_big(bbox):
            abort(
                403,
                description="The specified bounding box is too large. "
                "Consider splitting it into several small bounding boxes",
            )

        ids = RegionExporter(
            sentinel_dataset=sentinel_dataset,
            credentials=credentials,
            file_dimensions=file_dimensions,
            num_timesteps=num_timesteps,
        ).export(
            dest_bucket=dest_bucket,
            model_name=model_name,
            region_bbox=bbox,
            start_date=start_date,
            metres_per_polygon=50000,
        )

        id = f"{model_name}_{sentinel_dataset}"
        data = {
            "bbox": bbox.url,
            "model_name": model_name,
            "dataset_name": sentinel_dataset,
            "start_year": start_year,
            "num_timesteps": num_timesteps,
            "complete_ee_tasks": 0,
            "total_ee_tasks": len(ids),
            "start_time": str(datetime.datetime.now()),
            "ee_status": "https://us-central1-bsos-geog-harvest1.cloudfunctions.net/ee-status",
            "ee_files_exported": 0,
            "predictions_made": 0,
        }
        db.collection("crop-mask-runs").document(id).set(data)
        return data

    except Exception as e:
        logger.exception(e)
        abort(500, description=str(e))


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

    return {"amount": len(tasks), "tasks": tasks}
