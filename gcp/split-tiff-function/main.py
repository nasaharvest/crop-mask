"""
https://gis.stackexchange.com/questions/306861/split-geotiff-into-multiple-cells-with-
"""

from shapely import geometry
from pathlib import Path
import json
import math
import logging
import rasterio
import re
import tempfile
from rasterio.mask import mask
from rasterio.session import GSSession
from rasterio.io import MemoryFile
from google.cloud import secretmanager, storage
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.blob import Blob

logger = logging.getLogger(__name__)


# Takes a  dataset and splits it into squares of dimensions squareDim * squareDim
def splitImageIntoCells(src_path: str, dest_bucket: Bucket, squareDim: int = 1000):
    logger.debug(f"Splitting {src_path}")

    src_name = Path(src_path).stem
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", src_name)
    if len(dates) != 2:
        raise ValueError(f"Unable to extract dates from: {src_path}")

    start_date, end_date = dates
    name = src_name.split(start_date)[0]
    tile_identifier = src_name.split(end_date)[-1]

    img = rasterio.open(src_path)
    numberOfCellsWide = math.ceil(img.shape[1] / squareDim)
    numberOfCellsHigh = math.ceil(img.shape[0] / squareDim)
    count = 0
    for hc in range(numberOfCellsHigh):
        y = min(hc * squareDim, img.shape[0])
        for wc in range(numberOfCellsWide):
            x = min(wc * squareDim, img.shape[1])
            geom = getTileGeom(img.transform, x, y, squareDim)
            dest = f"{src_name}/{count}_{name}{tile_identifier}_{start_date}_{end_date}.tif"
            dest_blob = dest_bucket.blob(dest)
            writeImageAsGeoTIFF(img, geom, dest_blob)
            count = count + 1


# Generate a bounding box from the pixel-wise coordinates
# using the original datasets transform property
def getTileGeom(transform, x, y, squareDim):
    corner1 = (x, y) * transform
    corner2 = (x + squareDim, y + squareDim) * transform
    return geometry.box(corner1[0], corner1[1], corner2[0], corner2[1])


# Write the passed in dataset as a GeoTIFF
def writeImageAsGeoTIFF(img, geom, dest_blob: Blob):
    crop, cropTransform = mask(img, [geom], crop=True)
    img.meta.update({
        "driver": "GTiff",
        "height": crop.shape[1],
        "width": crop.shape[2],
        "transform": cropTransform,
        "crs": img.crs,
    })
    logger.info(f"Writing to {dest_blob}")

    with MemoryFile() as mem_file:
        with mem_file.open(**img.meta) as dataset:
            dataset.write(crop)

        dest_blob.upload_from_file(file_obj=mem_file, content_type="image/tiff")
        logger.info(f"Uploaded to {dest_blob}")


def get_credentials():
    filename = tempfile.gettempdir() + "/creds.json"
    if not Path(filename).exists():
        client = secretmanager.SecretManagerServiceClient()
        name = client.secret_version_path("670160663426", "google_application_credentials", "1")
        response = client.access_secret_version(name=name)
        payload = response.payload.data.decode('UTF-8')
        output_dict = json.loads(payload)
        with open(filename, 'w') as outfile:
            json.dump(output_dict, outfile)
    return filename


def hello_gcs(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    bucket_name = event['bucket']
    blob_name = event['name']
    src_path = f"gs://{bucket_name}/{blob_name}"

    creds_path = get_credentials()

    storage_client = storage.Client.from_service_account_json(creds_path)
    dest_bucket = storage_client.get_bucket('ee-data-for-inference')

    storage.blob._DEFAULT_CHUNKSIZE = 2097152 # 1024 * 1024 B * 2 = 2 MB
    storage.blob._MAX_MULTIPART_SIZE = 2097152 # 2 MB

    gs_session = GSSession(creds_path)
    with rasterio.Env(gs_session):
        splitImageIntoCells(src_path, dest_bucket)
