export DATASETS="geowiki_landcover_2017,kenya_non_crop,one_acre_fund_kenya,plant_village_kenya"
export MODEL_NAME=kenya
export AWS_CREDENTIALS=$HOME/.aws/credentials
export CLEARML_CREDENTIALS=$HOME/clearml.conf
export GDRIVE_TIF_DIR="nasaharvest/test"
export RCLONE_CREDENTIALS=$HOME/.config/rclone/rclone.conf

pull_credentials_from_secrets () {
  if test ! -f "$1"
    then
      mkdir -p $( dirname "$1")
      aws secretsmanager get-secret-value --secret-id $2 --region us-east-1  --query SecretString --output text > $1
  fi
}

pull_credentials_from_secrets "$CLEARML_CREDENTIALS" "ivan/clearml.conf"
pull_credentials_from_secrets "$RCLONE_CREDENTIALS" "ivan/rclone.conf"