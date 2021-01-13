export DOCKER_BUILDKIT=1

# Builds docker image with landcover-mapping conda environment to avoid reinstalling dependencies often
docker build -f Dockerfile.deps -t cropmask/dependencies .

export AWS_CREDENTIALS=$HOME/.aws/credentials
export CLEARML_CREDENTIALS=$HOME/clearml.conf
export DATASETS="all"
export MODEL_NAME="kenya"

printf "Once model begins training, you can view progress here:\n\
https://app.community.clear.ml/projects/15e0af16e2954760b5acf7d0117d4cdc"

# Trains model
docker build -f Dockerfile.train \
-t cropmask/train-$MODEL_NAME \
--secret id=aws,src=$AWS_CREDENTIALS \
--secret id=clearml,src=$CLEARML_CREDENTIALS \
--build-arg DATASETS=$DATASETS \
--build-arg MODEL_NAME=$MODEL_NAME \
--output data \
.