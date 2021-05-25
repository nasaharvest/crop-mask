#!/bin/sh
echo $1
torchserve --start --ncs --model-store model-store \
    --models $(echo $(for m in $(echo $1 | xargs); do echo $m=$m.mar ; done))