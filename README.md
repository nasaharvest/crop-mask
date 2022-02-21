# Crop Map Generation

[![Status](https://github.com/nasaharvest/crop-mask/actions/workflows/main.yml/badge.svg)](https://github.com/nasaharvest/crop-mask/actions)
[![codecov](https://codecov.io/gh/nasaharvest/crop-mask/branch/master/graph/badge.svg?token=MARPAEPZMS)](https://codecov.io/gh/nasaharvest/crop-mask)

This repository contains code and data to generate annual and in-season crop masks. Two models are trained - a multi-headed pixel wise classifier to classify pixels as containing crop or not, and a multi-spectral satellite image forecaster which forecasts a 12 month timeseries given a partial input:

<img src="diagrams/models.png" alt="models" height="200px"/>

These can be used to create annual and in season crop maps.

## Contents

-   [1. Setting up a local environment](#1-setting-up-a-local-environment)
-   [1.1. Setting up local environment for running shapefile notebook](#1.1-setting-up-local-environment-for-running-shapefile-notebook)
-   [2. Training a new model](#2-training-a-new-model)
-   [3. Running inference at scale (on GCP)](#3-running-inference-at-scale--on-gcp-)
-   [4. Tests](#4-tests)
-   [5. Previously generated crop maps](#5-previously-generated-crop-maps)
-   [6. Acknowledgments](#6-acknowledgments)
-   [7. Reference](#7-reference)

## 1. Setting up a local environment for development

1. Ensure you have [anaconda](https://www.anaconda.com/download/#macos) installed and run:
    ```bash
    conda config --set channel_priority true # Ensures conda will install environment
    conda env create -f environment-dev.yml   # Creates environment
    conda activate landcover-mapping      # Activates environment
    ```
2. [OPTIONAL] When adding new labeled data, Google Earth Engine is used to export Satellite data. To authenticate Earth Engine run:
    ```bash
    earthengine authenticate                # Authenticates Earth Engine
    python -c "import ee; ee.Initialize()"  # Will raise error if not setup
    ```
3. [OPTIONAL] To access existing data (ie. features, models), ensure you have [gcloud](https://cloud.google.com/sdk/docs/install) CLI installed and run:

    ```bash
    gcloud auth application-default login     # Authenticates gcloud
    dvc pull                                  # All data (will take long time)
    dvc pull data/features data/models        # For retraining or inference
    dvc pull data/processed                   # For labeled data analysis
    ```

4. [OPTIONAL] Weights and Biases is used for logging model training, to train and view logs run:
    ```bash
    wandb login
    ```

## 1.1 Setting up local environment for running shapefile notebook
1. Ensure you have [anaconda](https://www.anaconda.com/download/#macos) installed and run:
    ```bash
    conda env create -f environment-lite.yml   # Creates environment
    conda activate landcover-lite      # Activates environment
    ```
2. Start the jupyter server by running:
    ```bash
    jupyter notebook
    ```

## 2. Adding new labeled data

1. Ensure local environment is set up and all existing data is downloaded.
2. Add the shape file for new labels into [data/raw](data/raw)
3. In [dataset_labeled.py](src/datasets_labeled.py) add a new `LabeledDataset` object into the `labeled_datasets` list and specify the required parameters.
4. To create ML ready features run:

    ```bash
    gsutil -m cp -n -r gs://crop-mask-tifs2/tifs data/
    python scripts/create_features.py
    ```

5. Run `dvc commit` and `dvc push` to upload the new labeled data to remote storage.

<img src="diagrams/data_processing_chart.png" alt="models" height="200px"/>

## 2. Training a new model
```bash
python scripts/model_train.py \
    --min_lon 36.45 \
    --max_lon 40.00 \
    --min_lat 12.25 \
    -- max_lat 14.895 \
    --model_name Ethiopia_Tigray_2020 \
    --eval_datasets Ethiopia_Tigray_2020
```
After training is complete a new entry will be added to [data/models.json](data/models.json) with metrics and a link to all configuration parameters.

## 3. Running inference at scale (on GCP)

**Deploying**

1. Ensure you have [gcloud](https://cloud.google.com/sdk/docs/install) CLI installed and authenticated.
2. Ensure you have a secret in GCP titled `GOOGLE_APPLICATION_CREDENTIALS`; this will allow Google Earth Engine to be authenticated.
3. Run the following to deploy the project into Google Cloud:

```bash
gsutil mb gs://crop-mask-earthengine
gsutil mb gs://crop-mask-preds
sh deploy_ee_functions.sh
sh deploy_inference.sh
```

**Checking which models are available**
https://crop-mask-management-api-grxg7bzh2a-uc.a.run.app/models

**Actual inference at scale**

```bash

curl -X POST http://us-central1-bsos-geog-harvest1.cloudfunctions.net/export-region \
    -o - \
    -H "Content-Type:application/json" \
    -d @gcp/requests/<example>.json
```

**Tracking progress**

```bash
# Earth Engine progress
curl https://us-central1-bsos-geog-harvest1.cloudfunctions.net/ee-status?additional=FAILED,COMPLETED | python -mjson.tool

# Amount of files exported
gsutil du gs://crop-mask-earthengine/<model name>/<dataset> | wc -l

# Amount of files predicted
gsutil du gs://crop-mask-preds/<model name>/<dataset> | wc -l
```

**Addressing missed predictions (Not automated)**
When processing 100,000 tif files it is highly likely that crop-mask inference may fail on some files due to issues with not scaling up fast enough. Run the cells in [notebooks/fix-preds-on-gcloud.ipynb](notebooks/fix-preds-on-gcloud.ipynb) to address this problem.

**Putting it all together (Not automated)**
Once an inference run is complete the result is several small `.nc` files. These need to be merged into a single `.tif` file. Currently this operation is not automated and requires the user to:

```bash
export MODEL="Rwanda"
export DATASET="Rwanda_v2"
export START_DATE=2019-04-01
export END_DATE=2020-04-01

# Download appropriate folder
gsutil -m cp -n -r gs://crop-mask-preds/$MODEL/$DATASET/ .

# Run gdal merge script
cd crop-mask
python scripts/merge.py --p ../$DATASET

# [OPTIONAL] Upload COG tif output to Google Cloud Storage
gsutil cp ../$DATASET/tifs/final.tif gs://crop-mask-preds-merged/$DATASET

# [OPTIONAL] Upload COG to Google Earth Engine
earthengine upload image --asset_id users/izvonkov/$DATASET -ts $START_DATE -te $END_DATE gs://crop-mask-preds-merged/$DATASET
```

## 4. Tests

The following tests can be run against the pipeline:

```bash
flake8 --max-line-length 100 src data scripts test # code formatting
mypy src data scripts  # type checking
python -m unittest # unit tests

# Integration tests
cd test
python -m unittest integration_test_labeled.py
python -m unittest integration_test_predict.py
```

## 5. Previously generated crop maps

Google Earth Engine:

-   [Kenya (post season)](https://code.earthengine.google.com/ea3613a3a45badfd01ce2ec914dfe1ef)
-   [Busia (in season)](https://code.earthengine.google.com/f567cccc28dad7a25e088d56dabfbd4c)

Zenodo

-   [Kenya (post season) and Busia (in season)](https://doi.org/10.5281/zenodo.4271143).

## 6. Acknowledgments

This model requires data from [Plant Village](https://plantvillage.psu.edu/) and [One Acre Fund](https://oneacrefund.org/). We thank those organizations for making these datasets available to us - please contact them if you are interested in accessing the data.

## 7. Reference

If you find this code useful, please cite the following paper:

Gabriel Tseng, Hannah Kerner, Catherine Nakalembe and Inbal Becker-Reshef. 2020. Annual and in-season mapping of cropland at field scale with sparse labels. Tackling Climate Change with Machine Learning workshop at NeurIPS ’20: December 11th, 2020
