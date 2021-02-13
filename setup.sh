export DOCKER_BUILDKIT=1

# Credential locations
export AWS_CREDENTIALS=$HOME/.aws/credentials
export CLEARML_CREDENTIALS=$HOME/clearml.conf
export RCLONE_CREDENTIALS=$HOME/.config/rclone/rclone.conf

# Check for AWS credentials
if test ! -f "$AWS_CREDENTIALS"
  then
    mkdir -p $( dirname "$AWS_CREDENTIALS") && touch "$AWS_CREDENTIALS" # Creates empty stub credential file
fi

# Function to pull credentials from AWS
pull_credentials_from_secrets () {
  if test ! -f "$1"
    then
      mkdir -p $( dirname "$1")
      aws secretsmanager get-secret-value --secret-id $2 --region us-east-1  --query SecretString --output text > $1
  fi
}

pull_credentials_from_secrets "$CLEARML_CREDENTIALS" "ivan/clearml.conf"
pull_credentials_from_secrets "$RCLONE_CREDENTIALS" "ivan/rclone.conf"