export DOCKER_BUILDKIT=1

# Builds docker image with landcover-mapping conda environment to avoid reinstalling dependencies often
docker build -f Dockerfile.gpudeps -t cropmask/gpudependencies .

printf "Once model begins inference, you can view progress here:\n\
https://app.community.clear.ml/projects/15e0af16e2954760b5acf7d0117d4cdc \n"

export AWS_CREDENTIALS=$HOME/.aws/credentials
export RCLONE_CREDENTIALS=$HOME/.config/rclone/rclone.conf
export CLEARML_CREDENTIALS=$HOME/clearml.conf
export MODEL_NAME="kenya"
export GDRIVE_TIF_DIR="earth_engine_kenya_non_crop"

docker build -f Dockerfile.inference \
-t cropmask/inference-$MODEL_NAME \
--secret id=aws,src=$AWS_CREDENTIALS \
--secret id=rclone,src=$RCLONE_CREDENTIALS \
--secret id=clearml,src=$CLEARML_CREDENTIALS \
--build-arg MODEL_NAME=$MODEL_NAME \
--build-arg GDRIVE_TIF_DIR=$GDRIVE_TIF_DIR \
.
