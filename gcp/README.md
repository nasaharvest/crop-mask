# Setup for Google Cloud Platform

```bash
gcloud config set project nasa-harvest
```

### Creating buckets
```bash
gsutil mb --project=nasa-harvest gs://ee-data-for-inference
```

### Deploying export-unlabeled function

```bash
cp -R src/ETL gcp/export_unlabeled_function/src

gcloud functions deploy export-unlabeled \
    --source=gcp/export_unlabeled_function \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python37 \
    --entry-point=export_unlabeled \
    --set-env-vars DEST_BUCKET="ee-data-for-inference"

gcloud functions deploy ee-status \
    --source=gcp/export_unlabeled_function \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python39 \
    --entry-point=get_status

rm -rf gcp/export_unlabeled_function/src/*
```

**Note**: Uses secret google_app_credentials to authenticate EarthEngine

## Making a request
```bash
curl -X POST http://us-central1-nasa-harvest.cloudfunctions.net/export-unlabeled -H "Content-Type:application/json" \
    -d '{"season": "post_season", "name": "RwandaRukumberi768", "min_lon": 30.288, "max_lon": 30.535, "min_lat": -2.289, "max_lat": -2.035, "file_dimensions": 768}' \
    | python -mjson.tool
```

**Note**: Uses secret google_app_credentials to allow rasterio to read from GCP bucket directly

## Seeing ee progress
```bash
curl https://us-central1-nasa-harvest.cloudfunctions.net/ee-status?additional=FAILED,COMPLETED | python -mjson.tool
```

# Docker
**Building image:**
```bash
export TAG=us-central1-docker.pkg.dev/nasa-harvest/crop-mask/inference
docker build -f Dockerfile.inference . --build-arg models="Kenya Mali Rwanda Togo" -t $TAG
```
**Deploying to Google Cloud:**
```bash
docker push $TAG
gcloud run deploy inference --image ${TAG}:latest \
        --memory=4Gi \
        --platform=managed \
        --region=us-central1 \
        --allow-unauthenticated
```
After deployment the GOOGLE_APPLICATION_CREDENTIALS must be added using the Console

**Invoking deployed service:**
```bash
export URI="gs://ee-data-for-inference/RwandaSake/RwandaSake_2019-04-01_2020-04-01.tif"
export URL="https://inference-sgmgtky4sq-uc.a.run.app"
export MODEL_NAME="Mali"
curl -X POST -d "uri=$URI" $URL/$MODEL_NAME
```

**Running container locally:**
```bash
# Set credentials path to run image locally (this is an example path)
export GCLOUD_CREDENTIALS="$HOME/nasaharvest/secrets/google_application_credentials.json"
docker run -v $GCLOUD_CREDENTIALS:/root/.config/gcloud/creds.json -d -p 8080:8080 -p 8081:8081 $TAG

