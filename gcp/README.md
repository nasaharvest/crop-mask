# Setup for Google Cloud Platform
### Creating buckets
```bash
gsutil mb gs://ee-data-to-be-split
gsutil mb gs://ee-data-for-inference
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