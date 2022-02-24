#!/bin/bash
if [[ -z "$1" ]]; then
  echo "ERROR: MODELS variable is undefined, please rebuild the image with --build-args MODELS <model name>"
  exit 1
fi

echo "$1"
torchserve --start --ncs --model-store model-store \
  --models $(echo $(for m in $(echo $1 | xargs); do echo $m=$m.mar ; done))