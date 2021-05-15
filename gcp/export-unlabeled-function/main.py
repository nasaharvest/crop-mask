import logging
import ee
import json
import tempfile

from flask import abort
from google.cloud import secretmanager
from pathlib import Path

from src.ETL.ee_boundingbox import BoundingBox
from src.ETL.ee_exporter import Season, RegionExporter

logger = logging.getLogger(__name__)

def get_credentials():
    filename = tempfile.gettempdir() + "/creds.json"
    if not Path(filename).exists():
        client = secretmanager.SecretManagerServiceClient()
        name = client.secret_version_path("670160663426", "google_application_credentials", "2")
        response = client.access_secret_version(name=name)
        payload = response.payload.data.decode('UTF-8')
        output_dict = json.loads(payload)
        with open(filename, 'w') as outfile:
            json.dump(output_dict, outfile)
    return filename


def hello_gcs(request):
    """
    Args:
         request (Request): Event payload.
    """
    request_json = request.get_json(silent=True)
    logger.info(request_json)

    bbox_keys = ["min_lon", "max_lon", "min_lat", "max_lat"]
    for key in ["name", "season"] + bbox_keys:
        if key not in request_json:
            abort(400, description=f"{key} is missing from request_json: {request_json}")

    sentinel_dataset = request_json["name"]
    bbox_args = {k: v for k,v in request_json.items() if k in bbox_keys}
    try:
        season = Season(request_json["season"])
    except ValueError as e:
        logger.exception(e)
        abort(400, description=str(e))

    try:
        service_account = 'nasa-harvest@appspot.gserviceaccount.com'
        credentials = ee.ServiceAccountCredentials(service_account, get_credentials())
        logger.info(f"Obtained credentials for {service_account}")
        RegionExporter(sentinel_dataset=sentinel_dataset, credentials=credentials).export(
            region_bbox=BoundingBox(**bbox_args), season=season, metres_per_polygon=None)
    except Exception as e:
        logger.exception(e)
        abort(500, description=str(e))

    return f"Started export of {sentinel_dataset}"