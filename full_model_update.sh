#!/bin/bash
set -e
set -x

CONDA_ENV=/opt/anaconda3/envs/landcover-mapping/bin

$CONDA_ENV/dvc pull -f

$CONDA_ENV/python scripts/models_train_and_evaluate.py --offline --retrain_all

$CONDA_ENV/dvc commit -f
$CONDA_ENV/dvc push