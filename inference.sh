export DOCKER_BUILDKIT=1

printf "Once model begins inference, you can view progress here:\n\
https://app.community.clear.ml/projects/15e0af16e2954760b5acf7d0117d4cdc \n"

export DATASETS=<DATASETS> #ie. "geowiki_landcover_2017,kenya_non_crop,one_acre_fund_kenya,plant_village_kenya"
export MODEL_NAME=<MODEL NAME> # ie. kenya
export GDRIVE_TIF_DIR=<GDRIVE_TIF_DIR> # ie. "earth_engine_rwanda"

export AWS_CREDENTIALS=$HOME/.aws/credentials
export CLEARML_CREDENTIALS=$HOME/clearml.conf
export RCLONE_CREDENTIALS=$HOME/.config/rclone/rclone.conf

if test ! -f "$AWS_CREDENTIALS"
  then
    mkdir -p $( dirname "$AWS_CREDENTIALS") && touch "$AWS_CREDENTIALS" # Creates empty stub credential file
fi

pull_credentials_from_secrets () {
  if test ! -f "$1"
    then
      aws secretsmanager get-secret-value --secret-id $2 --region us-east-1  --query SecretString --output text > $1
  fi
}

pull_credentials_from_secrets "$CLEARML_CREDENTIALS" "ivan/clearml.conf"
pull_credentials_from_secrets "$RCLONE_CREDENTIALS" "ivan/rclone.conf"


docker build -f Dockerfile.inference \
-t cropmask/inference-$MODEL_NAME \
--secret id=aws,src=$AWS_CREDENTIALS \
--secret id=rclone,src=$RCLONE_CREDENTIALS \
--secret id=clearml,src=$CLEARML_CREDENTIALS \
--build-arg MODEL_NAME=$MODEL_NAME \
--build-arg GDRIVE_TIF_DIR=$GDRIVE_TIF_DIR \
.
