export DOCKER_BUILDKIT=1

# Builds docker image with landcover-mapping conda environment to avoid reinstalling dependencies often
docker build -f Dockerfile.gpudeps -t cropmask/gpudependencies .

printf "Once model begins inference, you can view progress here:\n\
https://app.community.clear.ml/projects/15e0af16e2954760b5acf7d0117d4cdc \n"


export AWS_CREDENTIALS=<PATH TO AWS CREDENTIALS FILE> # ie.$HOME/.aws/credentials
export RCLONE_CREDENTIALS=<PATH TO RCLONE CREDENTIALS> # ie. $HOME/.config/rclone/rclone.conf
export CLEARML_CREDENTIALS=<PATH TO CLEARML CREDENTIALS FILE> # ie. $HOME/clearml.conf
export DATASETS=<DATASETS> #ie. "geowiki_landcover_2017,kenya_non_crop,one_acre_fund_kenya,plant_village_kenya"
export MODEL_NAME=<MODEL NAME> # ie. kenya
export GDRIVE_TIF_DIR=<GDRIVE_TIF_DIR> # ie. "earth_engine_rwanda"

docker build -f Dockerfile.inference \
-t cropmask/inference-$MODEL_NAME \
--secret id=aws,src=$AWS_CREDENTIALS \
--secret id=rclone,src=$RCLONE_CREDENTIALS \
--secret id=clearml,src=$CLEARML_CREDENTIALS \
--build-arg MODEL_NAME=$MODEL_NAME \
--build-arg GDRIVE_TIF_DIR=$GDRIVE_TIF_DIR \
--no-cache \
--progress plain \
.
