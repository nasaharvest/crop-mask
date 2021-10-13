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

cp -R src/ETL gcp/export_unlabeled_function/src
cp -R src/bounding_boxes.py gcp/export_unlabeled_function/src

gcloud functions deploy export-unlabeled \
    --source=gcp/export_unlabeled_function \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python37 \
    --entry-point=export_unlabeled \
    --timeout=300s \
    --set-env-vars DEST_BUCKET=$BUCKET \
    --set-env-vars MODELS="$MODELS"

gcloud functions deploy ee-status \
    --source=gcp/export_unlabeled_function \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python39 \
    --entry-point=get_status

rm -rf gcp/export_unlabeled_function/src/*
