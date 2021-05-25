import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hello_gcs(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    logger.info(event)
    bucket_name = event['bucket']
    blob_name = event['name']
    src_path = f"gs://{bucket_name}/{blob_name}"
    logger.info(src_path)

