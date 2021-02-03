export DOCKER_BUILDKIT=1

# Builds docker image with landcover-mapping conda environment to avoid reinstalling dependencies often
docker build -f Dockerfile.deps -t cropmask/dependencies .

export DATASETS="geowiki_landcover_2017,kenya_non_crop,one_acre_fund_kenya,plant_village_kenya"
export MODEL_NAME=kenya
export AWS_CREDENTIALS=$HOME/.aws/credentials
export CLEARML_CREDENTIALS=$HOME/clearml.conf

pull_credentials_from_secrets () {
  if test ! -f "$1"
    then
      aws secretsmanager get-secret-value --secret-id $2 --region us-east-1  --query SecretString --output text > $1
  fi
}

check_aws_credentials () {
  if test ! -f "$AWS_CREDENTIALS"
    then
      mkdir -p $( dirname "$AWS_CREDENTIALS") && touch "$AWS_CREDENTIALS"
  fi
}

pull_credentials_from_secrets "$CLEARML_CREDENTIALS" "ivan/clearml.conf"
check_aws_credentials

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

