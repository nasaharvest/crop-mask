# Crop Map Generation
[![Status](https://github.com/nasaharvest/crop-mask/actions/workflows/main.yml/badge.svg)](https://github.com/nasaharvest/crop-mask/actions)
[![codecov](https://codecov.io/gh/nasaharvest/crop-mask/branch/master/graph/badge.svg?token=MARPAEPZMS)](https://codecov.io/gh/nasaharvest/crop-mask)

This repository contains code and data to generate annual and in-season crop masks. Two models are trained - a multi-headed pixel wise classifier to classify pixels as containing crop or not, and a multi-spectral satellite image forecaster which forecasts a 12 month timeseries given a partial input:

<img src="diagrams/models.png" alt="models" height="200px"/>


These can be used to create annual and in season crop maps. 

## Contents
- [Adding new data](#adding-new-data)
  * [Setting up the environment](#setting-up-the-environment)
  * [Adding Unlabeled Data](#adding-unlabeled-data)
  * [Adding Labeled Data](#adding-labeled-data)
- [Generating a Crop Map](#generating-a-crop-map)
  * [Setup Environment Variables](#setup-environment-variables)
  * [Training a New Model to Generate a Crop Map](#training-a-new-model-to-generate-a-crop-map)
  * [Generating a Crop Map with an Existing Model](#generating-a-crop-map-with-an-existing-model)
  * [Monitoring Training and Inference](#monitoring-training-and-inference)
  * [Diagram of the Whole Process](#diagram-of-the-whole-process)
- [Development](#development)
  * [Building the docker image locally](#building-the-docker-image-locally)
  * [Tests](#tests)
- [Previously Generated Crop Maps](#previously-generated-crop-maps)
- [Acknowledgments](#acknowledgments)
- [Reference](#reference)
## Adding new data
Adding new labeled data is a prerequisite for training new machine learning models and adding new unlabeled data is a prerequisite for generating a crop map with an existing model. 

### Setting up the environment
1. Ensure you have [anaconda](https://www.anaconda.com/download/#macos) installed. Anaconda running python 3.6 is used as the package manager.
2. Setup a conda environment for this project
    ```bash
    conda env create -f environment.yml
    ```
    This will create an environment named `landcover-mapping` with all the necessary packages to run the code. 
3. Activate the environment by running
    ```bash
    conda activate landcover-mapping
    ```
4. Authenticate earthengine to allow for exporting satellite data from Google Earth Engine to Google Drive
    ```bash
    earthengine authenticate
    ```
    To verify the earthengine is correctly authenticated run
    ```bash
    python -c "import ee; ee.Initialize()"
    ```

### Adding Unlabeled Data
**Purpose:** 
Unlabeled data is a set of satellite images without a crop/non-crop label. Unlabeled data is used to make predictions.

**Steps to add unlabeled data:**
1. Open the [dataset_unlabeled.py](data/datasets_unlabeled.py) file and add a new `UnlabeledDataset` object into the `unlabeled_datasets` list and specify the required parameters (ie bounding box for region).
2. To begin exporting satellite data from Google Earth Engine to your Google Drive run (from scripts directory):
    ```
    python export_for_unlabeled.py --dataset_name <YOUR DATASET NAME>
    ```
![Add unlabeled data](diagrams/add_unlabeled_data.png)

Running exports can be viewed (and individually cancelled) in the `Tabs` bar on the [Earth Engine Code Editor](https://code.earthengine.google.com/).

### Adding Labeled Data
**Purpose:** 
Labeled data is a set of satellite images with a crop/non-crop label. Labeled data is used to train and evaluate the machine learning model.

Since the labeled data is directly tied to the machine learning model, it is kept track of using [dvc](https://dvc.org/doc) inside the [data](data) directory. The [data](data) directory contains *.dvc files which point to the version and location of the data in remote storage (in this case an AWS S3 Bucket).

**Accessing existing labeled data:**
1. Obtain valid NASAHarvest AWS credentials (access key, secret key)
2. Ensure you have the [AWS cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) on your machine
3. Setup a profile by running `aws configure` and entering your AWS credentials
4. Run `dvc pull` from the project root directory to pull in existing labeled data (dvc is installed as part of the conda environment)

**Steps to add new labaled data:**
1. Add the shape file for new labels into [data/raw](data/raw)
2. In [dataset_labeled.py](data/datasets_labeled.py) add a new `LabeledDataset` object into the `labeled_datasets` list and specify the required parameters.
3. To process the labels into a standard format and begin exporting satellite data from Google Earth Engine run (from scripts directory):
    ```bash
    python export_for_labeled.py
    ``` 
4. Google Earth Engine will automatically export satellite images to Google Drive.
5. Once the satellite data has been exported, download it from Google Drive into [data/raw](data/raw).
6. To combine the labels and the satellite images into a machine learning suitable dataset run (from scripts directory):
    ```bash
    python engineer.py
    ```
7. Run `dvc commit` and `dvc push` to upload the new labeled data to remote storage.

![Add labeled data](diagrams/add_labeled_data.png)
## Generating a Crop Map

You must have [docker](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html) and awscli installed on the machine. If doing inference and using the `--gpus all` flag the host machine must have accessible GPU drivers and [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) is setup.

### Setup Environment Variables
- If you don't have the crop-mask repo, and you aren't sure if you have all the credentials on your machine: simply copy and paste the contents of setup.sh into your shell.
- If you don't have the crop-mask repo, but have all the credential information locally, you can set the environment variables directly
  ```bash
  # Example
  export DOCKER_BUILDKIT=1
  export AWS_CREDENTIALS=$HOME/.aws/credentials
  export CLEARML_CREDENTIALS=$HOME/clearml.conf
  export RCLONE_CREDENTIALS=$HOME/.config/rclone/rclone.conf
  ```
- If you have the crop-mask repo available just run:
  ```bash
  source setup.sh
  ```

### Training a New Model to Generate a Crop Map
**Step 1:** Specify the following arguments:
- `DATASETS` - which datasets to use for trianing (labeled dataset generated above)
- `MODEL_NAME` - a unique identifier for the resulting model
- `MODELS_DVC_DIR` - a directory on the host machine where the models.dvc file will be exported

```bash
# Example
export DATASETS="geowiki_landcover_2017,kenya_non_crop,one_acre_fund_kenya,plant_village_kenya"
export MODEL_NAME="kenya"
export MODELS_DVC_DIR="$HOME/crop-mask/data"
```

**Step 2:** Begin training:
```bash
docker run \
  -v $AWS_CREDENTIALS:/root/.aws/credentials \
  -v $CLEARML_CREDENTIALS:/root/clearml.conf \
  --mount type=bind,source=$MODELS_DVC_DIR,target=/vol \
  -it ivanzvonkov/cropmask conda run -n landcover-mapping python model.py \
  --datasets $DATASETS \
  --model_name $MODEL_NAME
```
This command does the following:
1. Pulls in the specified labeled dataset to train a model 
2. Pushes trained model to remote storage and outputs the models.dvc file to `$MODELS_DVC_DIR`, this file needs to be git committed inorder to share the trained model with collaborators 

![train](diagrams/train.png)
### Generating a Crop Map with an Existing Model

**Step 1:** Specify the following arguments:
- `MODEL_NAME` - model used for inference
- `GDRIVE_DIR` - source of the input files on Google Drive (the unlabeled dataset generated above)
- `VOLUME` - a directory on the host with a lot of space for storing the inputs and predictions. If using an EC2 instance it is recommended to [mount an EBS volume](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html) and use its path for this variable

```bash
# Example
export MODEL_NAME="kenya"
export GDRIVE_DIR="remote2:earth_engine_region_rwanda"
export VOLUME="/data"
```

**Step 2:** Begin inference:
```bash
docker run --gpus all \
  -v $CLEARML_CREDENTIALS:/root/clearml.conf \
  -v $RCLONE_CREDENTIALS:/root/.config/rclone/rclone.conf \
  --mount type=bind,source=$VOLUME,target=/vol \
  -it ivanzvonkov/cropmask conda run -n landcover-mapping python predict.py \
  --gdrive_path_to_tif_files $GDRIVE_DIR \
  --local_path_to_tif_files /vol/input \
  --split_tif_files true \
  --model_name $MODEL_NAME \
  --predict_dir /vol/predict
```

This command does the following:
3. Pulls in the satellite images in Google Drive and splits them so they are ready for inference
4. Runs inference on each of the split files and outputs a crop map to remote storage.

**Note**: The ML model is packaged into the docker image at build time not during this command.

![inference](diagrams/inference.png)

### Monitoring Training and Inference
ClearML is used for monitoring training and inference during each docker run. You'll need a ClearML account and access to the ClearML workspace (contact izvonkov@umd.edu)

### Diagram of the Whole Process
![crop_map_generation](diagrams/crop_map_generation.png)
## Development
### Building the docker image locally
```bash
export DOCKER_BUILDKIT=1
export AWS_CREDENTIALS=$HOME/.aws/credentials
docker build -t ivanzvonkov/cropmask --secret id=aws,src=$AWS_CREDENTIALS .
```
### Tests

The following tests can be run against the pipeline:

```bash
black .  # code formatting
mypy src scripts  # type checking
python -m unittest
```

## Previously Generated Crop Maps
Google Earth Engine:
* [Kenya (post season)](https://code.earthengine.google.com/ea3613a3a45badfd01ce2ec914dfe1ef)
* [Busia (in season)](https://code.earthengine.google.com/f567cccc28dad7a25e088d56dabfbd4c)

Zenodo
- [Kenya (post season) and Busia (in season)](https://doi.org/10.5281/zenodo.4271143).

## Acknowledgments
This model requires data from [Plant Village](https://plantvillage.psu.edu/) and [One Acre Fund](https://oneacrefund.org/). We thank those organizations for making these datasets available to us - please contact them if you are interested in accessing the data.

## Reference

If you find this code useful, please cite the following paper:

Gabriel Tseng, Hannah Kerner, Catherine Nakalembe and Inbal Becker-Reshef. 2020. Annual and in-season mapping of cropland at field scale with sparse labels. Tackling Climate Change with Machine Learning workshop at NeurIPS ’20: December 11th, 2020
