export DOCKER_BUILDKIT=1

# Builds docker image with landcover-mapping conda environment to avoid reinstalling dependencies often
docker build -f Dockerfile.deps -t cropmask/dependencies .


export AWS_CREDENTIALS=$HOME/.aws/credentials
export CLEARML_CREDENTIALS=$HOME/clearml.conf
export DATASETS="all"
export MODEL_NAME="kenya"

# Trains model
docker build -f Dockerfile.train \
--secret id=aws,src=$AWS_CREDENTIALS \
--secret id=clearml,src=$CLEARML_CREDENTIALS \
--build-arg DATASETS=$DATASETS \
--build-arg MODEL_NAME=$MODEL_NAME \
.