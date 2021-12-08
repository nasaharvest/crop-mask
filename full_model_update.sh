# exit when any command fails
set -e

conda activate landcover-mapping

python scripts/models_train_and_evaluate.py --offline --retrain_all

dvc commit -f
dvc push