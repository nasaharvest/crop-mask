printf "Once model begins inference, you can view progress here:\n\
https://app.community.clear.ml/projects/15e0af16e2954760b5acf7d0117d4cdc"

export AWS_CREDENTIALS=<PATH TO AWS CREDENTIALS FILE> # ie.$HOME/.aws/credentials
export CLEARML_CREDENTIALS=<PATH TO CLEARML CREDENTIALS FILE> # ie. $HOME/clearml.conf
export DATASETS=<DATASETS> #ie. "all"
export MODEL_NAME=<MODEL NAME> # ie. "kenya"
export PATH_TO_TIF_FILES=<PATH TO TIF FILES>

docker build -f Dockerfile.inference \
-t cropmask/inference-$MODEL_NAME \
--secret id=aws,src=$AWS_CREDENTIALS \
--secret id=clearml,src=$CLEARML_CREDENTIALS \
--build-arg MODEL_NAME=$MODEL_NAME \
--build-arg PATH_TO_TIF_FILES=$PATH_TO_TIF_FILES \
.