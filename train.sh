export DOCKER_BUILDKIT=1

# Builds docker image with landcover-mapping conda environment to avoid reinstalling dependencies often
docker build -f Dockerfile.deps -t cropmask/dependencies .

export AWS_CREDENTIALS=<PATH TO AWS CREDENTIALS FILE> # ie.$HOME/.aws/credentials
export CLEARML_CREDENTIALS=<PATH TO CLEARML CREDENTIALS FILE> # ie. $HOME/clearml.conf
export DATASETS=<DATASETS> #ie. "geowiki_landcover_2017,kenya_non_crop,one_acre_fund_kenya,plant_village_kenya"
export MODEL_NAME=<MODEL NAME> # ie. kenya

printf "Once model begins training, you can view progress here:\n\
https://app.community.clear.ml/projects/15e0af16e2954760b5acf7d0117d4cdc\n"

# Trains model 
docker build -f Dockerfile.train \
-t cropmask/train-$MODEL_NAME \
--secret id=aws,src=$AWS_CREDENTIALS \
--secret id=clearml,src=$CLEARML_CREDENTIALS \
--build-arg DATASETS=$DATASETS \
--build-arg MODEL_NAME=$MODEL_NAME \
--output data \
.