# Crop Map Generation
[![Status](https://github.com/nasaharvest/crop-mask/actions/workflows/main.yml/badge.svg)](https://github.com/nasaharvest/crop-mask/actions)
[![codecov](https://codecov.io/gh/nasaharvest/crop-mask/branch/master/graph/badge.svg?token=MARPAEPZMS)](https://codecov.io/gh/nasaharvest/crop-mask)

This repository contains code and data to generate annual and in-season crop masks. Two models are trained - a multi-headed pixel wise classifier to classify pixels as containing crop or not, and a multi-spectral satellite image forecaster which forecasts a 12 month timeseries given a partial input:

<img src="diagrams/models.png" alt="models" height="200px"/>


These can be used to create annual and in season crop maps. 

## Contents
- [1. Setting up a local environment](#1-setting-up-a-local-environment)
- [2. Training a new model](#2-training-a-new-model)
- [3. Running inference locally](#3-running-inference-locally)
- [4. Running inference at scale (on GCP)](#4-running-inference-at-scale--on-gcp-)
- [5. Tests](#5-tests)
- [6. Previously generated crop maps](#6-previously-generated-crop-maps)
- [7. Acknowledgments](#7-acknowledgments)
- [8. Reference](#8-reference)

## 1. Setting up a local environment
1. Ensure you have [anaconda](https://www.anaconda.com/download/#macos) installed and run:
    ```bash
    conda env create -f environment-dev.yml   # Creates environment
    conda activate landcover-mapping      # Activates environment
    ```
2. [OPTIONAL] When adding new labeled data, Google Earth Engine is used to export Satellite data. To authenticate Earth Engine run:
    ```bash
    earthengine authenticate                # Authenticates Earth Engine              
    python -c "import ee; ee.Initialize()"  # Will raise error if not setup 
    ```
3. [OPTIONAL] To access existing data (ie. features, models), ensure you have [gcloud](https://cloud.google.com/sdk/docs/install) CLI installed and authenticated, and run:
    ```bash
    dvc pull                                  # All data (will take long time)
    dvc pull data/features data/models        # For retraining or inference
    dvc pull data/processed                   # For labeled data analysis
    ```
 

## 2. Training a new model
**Prerequisite:  Adding new labeled data:**
1. Ensure local environment is set up and all existing data is downloaded.
2. Add the shape file for new labels into [data/raw](data/raw)
3. In [dataset_labeled.py](src/datasets_labeled.py) add a new `LabeledDataset` object into the `labeled_datasets` list and specify the required parameters.
4. To process the labels into a standard format and begin exporting satellite data from Google Earth Engine run (from scripts directory):
    ```bash
    python export_for_labeled.py
    ``` 
5. Google Earth Engine will automatically export satellite images to Google Drive.
6. Once the satellite data has been exported, download it from Google Drive into [data/raw](data/raw).
7. To combine the labels and the satellite images into a machine learning suitable dataset run (from scripts directory):
    ```bash
    python create_features.py
    ```
8. Run `dvc commit` and `dvc push` to upload the new labeled data to remote storage.

![Add labeled data](diagrams/add_labeled_data.png)

**Actual Training**

You must have the specified datasets in `data/features`, then inside `scripts/` run:
```bash
python train_model.py --datasets "geowiki_landcover_2017,Kenya" --model_name "Kenya"
```

## 3. Running inference locally
**Prerequisite: Getting unlabeled data:**
1. Ensure local environment is set up.
3. In [dataset_unlabeled.py](src/datasets_unlabeled.py) add a new `UnlabeledDataset` object into the `unlabeled_datasets` list and specify the required parameters.
4. To begin exporting satellite data from Google Earth Engine, run (from scripts directory):
    ```bash
    python export_for_unlabeled.py --dataset_name <dataset name>
    ``` 
5. Google Earth Engine will automatically export satellite images to Google Drive.
6. Once the satellite data has been exported, download it from Google Drive into [data/raw](data/raw).
    
**Actual inference**
```bash
python predict.py --model_name "Kenya" --local_path_to_tif_files "../data/raw/<dataset name>
```
## 4. Running inference at scale (on GCP)
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
**Actual inference at scale**
```bash
curl -X POST http://us-central1-nasa-harvest.cloudfunctions.net/export-unlabeled \
    -H "Content-Type:application/json" \
    -d @gcp/<example>.json 
```
**Tracking progress**
```bash
# Earth Engine progress
curl https://us-central1-nasa-harvest.cloudfunctions.net/ee-status?additional=FAILED,COMPLETED | python -mjson.tool

# Amount of files exported
gsutil du gs://crop-mask-earthengine/<model name>/<dataset> | wc -l

# Amount of files predicted
gsutil du gs://crop-mask-unmerged-preds/<model name>/<dataset> | wc -l
```

**Putting it all together**
Once an inference run is complete the result is several small `.nc` files. These need to be merged into a single `.tif` file. Currently this operation is not automated and requires the user to:
1. Download the appropriate folder
    ```bash
    gsutil -m cp -r "gs://crop-mask-preds/<model>/<dataset>/"
    ```
2. Specify the folder location in [gcp/merger/main.py](gcp/merger/main.py) and run the script.


## 5. Tests
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

## 6. Previously generated crop maps
Google Earth Engine:
* [Kenya (post season)](https://code.earthengine.google.com/ea3613a3a45badfd01ce2ec914dfe1ef)
* [Busia (in season)](https://code.earthengine.google.com/f567cccc28dad7a25e088d56dabfbd4c)

Zenodo
- [Kenya (post season) and Busia (in season)](https://doi.org/10.5281/zenodo.4271143).

## 7. Acknowledgments
This model requires data from [Plant Village](https://plantvillage.psu.edu/) and [One Acre Fund](https://oneacrefund.org/). We thank those organizations for making these datasets available to us - please contact them if you are interested in accessing the data.

## 8. Reference

If you find this code useful, please cite the following paper:

Gabriel Tseng, Hannah Kerner, Catherine Nakalembe and Inbal Becker-Reshef. 2020. Annual and in-season mapping of cropland at field scale with sparse labels. Tackling Climate Change with Machine Learning workshop at NeurIPS ’20: December 11th, 2020
