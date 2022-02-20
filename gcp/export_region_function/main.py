import logging
import json
import tempfile
import ee
import os

from cropharvest.countries import BBox
from cropharvest.eo import EarthEngineExporter
from datetime import datetime
from flask import abort, Request
from google.cloud import secretmanager
from google.cloud import firestore
from pathlib import Path

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


def is_bbox_too_big(bbox: BBox):
    lat_len = bbox.max_lat - bbox.min_lat
    lon_len = bbox.max_lon - bbox.min_lon
    return lat_len * lon_len > BBOX_LIMIT


def export_region(request: Request):
    """
    Args:
         request (Request): Event payload.
    """
    dest_bucket = os.environ.get("DEST_BUCKET")
    all_model_names = os.environ.get("MODELS").split(" ")

    request_json = request.get_json(silent=True)
    logger.info("Resquest json:")
    logger.info(request_json)

    bbox_keys = ["min_lon", "max_lon", "min_lat", "max_lat"]
    for key in ["model_name", "version", "start_date", "end_date"] + bbox_keys:
        if key not in request_json:
            abort(400, description=f"{key} is missing from request_json: {request_json}")

    model_name = request_json["model_name"]
    if model_name not in all_model_names:
        abort(400, description=f"{model_name} not found in: {all_model_names}")

    version = request_json["version"]
    start_date = datetime.strptime(request_json["start_date"], "%Y-%m-%d").date()
    end_date = datetime.strptime(request_json["end_date"], "%Y-%m-%d").date()

    file_dimensions = request_json.get("file_dimensions", 256)
    credentials = get_ee_credentials()
    bbox_args = {k: v for k, v in request_json.items() if k in bbox_keys}
    bbox = BBox(**bbox_args)

    if is_bbox_too_big(bbox):
        abort(
            403,
            description="The specified bounding box is too large. "
            "Consider splitting it into several small bounding boxes",
        )
    try:
        bbox_name = f"{model_name}/{version}"
        ids = EarthEngineExporter(
            credentials=credentials, dest_bucket=dest_bucket, check_ee=True, check_gcp=True
        ).export_for_bbox(
            bbox=bbox,
            bbox_name=bbox_name,
            start_date=start_date,
            end_date=end_date,
            metres_per_polygon=50000,
            file_dimensions=file_dimensions,
        )

        data = {
            "bbox": bbox.url,
            "model_name": model_name,
            "version": version,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "complete_ee_tasks": 0,
            "total_ee_tasks": len(ids),
            "run_start_time": str(datetime.now()),
            "ee_status": "https://us-central1-bsos-geog-harvest1.cloudfunctions.net/ee-status",
            "ee_files_exported": 0,
            "predictions_made": 0,
        }
        db.collection("crop-mask-runs").document(bbox_name).set(data)
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

    query_params = list(request.args.values())
    additional_statuses = query_params[0] if len(query_params) > 0 else ""

    include_completed = "COMPLETED" in additional_statuses
    include_failed = "FAILED" in additional_statuses

    def include_task(t):
        if t.state == "COMPLETED":
            return include_completed
        if t.state == "FAILED":
            return include_failed
        return True

    tasks = []
    try:
        credentials = get_ee_credentials()
        ee.Initialize(credentials)
        logger.info("Looping through ee task list:")
        tasks = [t.status() for t in ee.batch.Task.list() if include_task(t)]

    except Exception as e:
        logger.exception(e)
        abort(500, description=str(e))

    return {"amount": len(tasks), "tasks": tasks}
