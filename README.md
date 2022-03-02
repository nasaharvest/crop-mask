# Crop Map Generation

[![Status](https://github.com/nasaharvest/crop-mask/actions/workflows/main.yml/badge.svg)](https://github.com/nasaharvest/crop-mask/actions)
[![codecov](https://codecov.io/gh/nasaharvest/crop-mask/branch/master/graph/badge.svg?token=MARPAEPZMS)](https://codecov.io/gh/nasaharvest/crop-mask)

This repository contains code and data to generate annual and in-season crop masks. 

To create a crop-mask using an already trained model click Open In Colab button below: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/crop-mask/blob/master/notebooks/crop_mask_inference.ipynb)



Two models are trained - a multi-headed pixel wise classifier to classify pixels as containing crop or not, and a multi-spectral satellite image forecaster which forecasts a 12 month timeseries given a partial input:

<img src="diagrams/models.png" alt="models" height="200px"/>

These can be used to create annual and in season crop maps.

## Contents

-   [1. Setting up a local environment](#1-setting-up-a-local-environment)
-   [2. Adding new labeled data](#2-adding-new-labeled-data)
-   [3. Training a new model](#3-training-a-new-model)
-   [4. Tests](#4-tests)
-   [5. Previously generated crop maps](#5-previously-generated-crop-maps)
-   [6. Acknowledgments](#6-acknowledgments)
-   [7. Reference](#7-reference)

## 1. Setting up a local environment

#### 1.1 For development 

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

#### 1.2 For shapefile notebook
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

## 3. Training a new model
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

Save the model to the repository by running:
```bash
dvc commit data/models.dvc
dvc push data/models
```
The model will be deployed when these files are merged into the main branch.

## 4. Tests

The following tests can be run against the pipeline:

```bash
flake8 . # code formatting
mypy .  # type checking
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
