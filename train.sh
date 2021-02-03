export DOCKER_BUILDKIT=1

# Builds docker image with landcover-mapping conda environment to avoid reinstalling dependencies often
docker build -f Dockerfile.deps -t cropmask/dependencies .

printf "Once model begins training, you can view progress here:\n\
https://app.community.clear.ml/projects/15e0af16e2954760b5acf7d0117d4cdc\n"


export AWS_CREDENTIALS="" #$HOME/.aws/credentials
export CLEARML_CREDENTIALS=$HOME/clearml.conf
export DATASETS="geowiki_landcover_2017,kenya_non_crop,one_acre_fund_kenya,plant_village_kenya"
export MODEL_NAME=kenya

aws_build_secret () {
  if test -f "$1"
    then
      echo "--secret id=aws,src=$1"
    else
      echo ""
  fi
}

aws_build_secret_mount () {
  if test -f "$1"
    then
      echo "--mount=type=secret,id=aws,target=/root/.aws/credentials"
    else
      echo ""
  fi
}

# Trains model 
docker build -f Dockerfile.train \
-t cropmask/train-$MODEL_NAME \
--secret id=clearml,src=$CLEARML_CREDENTIALS $(aws_build_secret "$AWS_CREDENTIALS") \
--build-arg AWS_BUILD_SECRET_MOUNT=$(aws_build_secret_mount "$AWS_CREDENTIALS") \
--build-arg DATASETS=$DATASETS \
--build-arg MODEL_NAME=$MODEL_NAME \
--output data \
.

