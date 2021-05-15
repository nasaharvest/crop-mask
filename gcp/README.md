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
    --source=gcp/export-unlabeled-function \
    --project=nasa-harvest \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python37 \
    --entry-point=hello_gcs

rm -rf gcp/export-unlabeled-function/src/*
```

### Deploying split-tiff function
```bash
gcloud functions deploy split-tiff \
    --source=gcp/split-tiff-function \
    --project=nasa-harvest \
    --trigger-bucket=ee-data-to-be-split \
    --runtime=python39 \
    --entry-point=hello_gcs
```

**Note**: Uses secret google_app_credentials to authenticate EarthEngine

## Making a request
```bash
curl -X POST http://us-central1-nasa-harvest.cloudfunctions.net/export-unlabeled -H "Content-Type:application/json" \
    -d '{"season": "post_season", "name": "Test", "min_lon": 30.401, "max_lon": 30.402, "min_lat": -2.227, "max_lat": -2.226}'
```
**Note**: Uses secret google_app_credentials to allow rasterio to read from GCP bucket directly