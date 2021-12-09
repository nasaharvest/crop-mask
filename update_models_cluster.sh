#!/bin/bash
set -e
set -x

source /apps/python/3.8/anaconda/etc/profile.d/conda.sh


conda activate landcover-mapping

dvc pull

python scripts/models_train_and_evaluate.py --offline --retrain_all

dvc commit -f
dvc push