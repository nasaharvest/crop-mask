
# exit when any command fails
set -e

# Ensure the models on DVC are being deployed
dvc pull data/models.dvc

export TAG=us-central1-docker.pkg.dev/bsos-geog-harvest1/crop-mask/crop-mask
export BUCKET=crop-mask-earthengine
export URL="https://crop-mask-grxg7bzh2a-uc.a.run.app"
export MODELS=$(
        python -c \
        "from pathlib import Path; \
        print(' '.join([p.stem for p in Path('data/models').glob('*.pt')]))"
)

docker build -f Dockerfile.inference . --build-arg MODELS="$MODELS" -t $TAG
docker push $TAG
gcloud run deploy crop-mask --image ${TAG}:latest \
        --memory=4Gi \
        --platform=managed \
        --region=us-central1 \
        --allow-unauthenticated

gcloud functions deploy trigger-inference \
    --source=gcp/trigger_inference_function \
    --trigger-bucket=$BUCKET \
    --allow-unauthenticated \
    --runtime=python39 \
    --entry-point=hello_gcs \
    --set-env-vars MODELS="$MODELS",INFERENCE_HOST="$URL" \
    --timeout=300s
