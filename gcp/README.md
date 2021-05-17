# Setup for Google Cloud Platform
### Creating buckets
```bash
gsutil mb --project=nasa-harvest gs://ee-data-to-be-split
gsutil mb --project=nasa-harvest gs://ee-data-for-inference
```

### Deploying export-unlabeled function

```bash
cp -R src/ETL gcp/export-unlabeled-function/src

gcloud functions deploy export-unlabeled \
    --source=gcp/export_unlabeled_function \
    --project=nasa-harvest \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python37 \
    --entry-point=export_unlabeled

gcloud functions deploy ee-status \
    --source=gcp/export_unlabeled_function \
    --project=nasa-harvest \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python39 \
    --entry-point=get_status

rm -rf gcp/export-unlabeled-function/src/*
```
### Deploying split-tiff function
```bash
gcloud functions deploy split-tiff \
    --source=gcp/split_tiff_function \
    --project=nasa-harvest \
    --trigger-bucket=ee-data-to-be-split \
    --runtime=python39 \
    --entry-point=hello_gcs \
    --memory=8GB \
    --timeout=540
```

**Note**: Uses secret google_app_credentials to authenticate EarthEngine

## Making a request
```bash
curl -X POST http://us-central1-nasa-harvest.cloudfunctions.net/export-unlabeled -H "Content-Type:application/json" \
    -d '{"season": "post_season", "name": "Test", "min_lon": 30.401, "max_lon": 30.402, "min_lat": -2.227, "max_lat": -2.226}'
```
**Note**: Uses secret google_app_credentials to allow rasterio to read from GCP bucket directly

## Seeing ee progress
```bash
curl https://us-central1-nasa-harvest.cloudfunctions.net/ee-status
```