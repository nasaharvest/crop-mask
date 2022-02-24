# exit when any command fails
set -e

# Ensure the models on DVC are being deployed
dvc pull data/models.dvc

export BUCKET=crop-mask-earthengine
export MODELS=$(
        python -c \
        "from pathlib import Path; \
        print(' '.join([p.stem for p in Path('data/models').glob('*.pt')]))"
)

gcloud functions deploy export-region \
    --source=gcp/export_region_function \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python37 \
    --entry-point=export_region \
    --timeout=300s \
    --memory 512MB \
    --set-env-vars DEST_BUCKET=$BUCKET \
    --set-env-vars MODELS="$MODELS"

gcloud functions deploy ee-status \
    --source=gcp/export_region_function \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python39 \
    --entry-point=get_status \
    --memory 512MB 
