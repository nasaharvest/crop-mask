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

check_installed () {
  if ! command -v $1 &> /dev/null
  then
    echo "$1 must be installed."
    return 1
  else
    return 0
  fi
}

# Function to pull credentials from AWS
pull_credentials_from_secrets () {
  mkdir -p $( dirname "$1")
  secret=$(aws secretsmanager get-secret-value --secret-id $2 --region us-east-1  --query SecretString --output text)
  if test ! $secret
    then
      echo "$2 not found."
    else
      echo "Found credentials: $2" 
      echo "$secret" > $1
      echo "Downloaded to $1"
  fi
}

check_installed docker || return
check_installed aws || return

pull_credentials_from_secrets "$CLEARML_CREDENTIALS" "ivan/clearml.conf"
pull_credentials_from_secrets "$RCLONE_CREDENTIALS" "ivan/rclone.conf"