# Crop Map Generation
[![Status](https://github.com/nasaharvest/crop-mask/actions/workflows/main.yml/badge.svg)](https://github.com/nasaharvest/crop-mask/actions)

## How it works
This repository contains code and data to generate annual and in-season crop masks. Two models are trained - a multi-headed pixel wise classifier to classify pixels as containing crop or not, and a multi-spectral satellite image forecaster which forecasts a 12 month timeseries given a partial input:

<img src="diagrams/models.png" alt="models" height="200px"/>


These can be used to create annual and in season crop maps. 

## Adding new data
Adding new data is a prerequisite for generating a crop map with an existing machine learning model and training new models. 

#### Setting up the environment
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
4. Authenticate earthengine to allow for exporting satellite data from Google Earth Engine to your Google Drive account.
    ```bash
    earthengine authenticate
    ```
    To verify the earthengine is correctly authenticated run
    ```bash
    python -c "import ee; ee.Initialize()"
    ```

#### Adding Unlabeled Data:
Unlabeled data is a set of satellite images without a crop/non-crop label. Unlabeled data is used to make predictions.

1. Open the [dataset_config.py](src/dataset_config.py) file and add a new `UnlabeledDataset` object into the `unlabeled_datasets` list. Specify the name of the dataset under `sentinel_dataset`, the bounding box to export (using EarthEngineExporter) and the season to export the satellite data for.
2. Navigate to the [export_for_unlabeled.py](scripts/export_unlabeled.py) file, specify the dataset name, and execute the script to begin exporting satellite data from Google Earth Engine to your Google Drive.

Running exports can be viewed (and individually cancelled) in the `Tabs` bar on the [Earth Engine Code Editor](https://code.earthengine.google.com/).

#### Adding Labeled Data
Labeled data is a set of satellite images with a crop/non-crop label. Labeled data is used to train and evaluate the machine learning model.

Since the labeled data is directly tied to the machine learning model, it is kept track of using [dvc](https://dvc.org/doc) inside the [data](data) directory. The [data](data) directory contains *.dvc files which point to the version and location of the data in remote storage (in this case an AWS S3 Bucket).

To have access to existing labeled data:
1. Obtain valid NASAHarvest AWS credentials (access key, secret key)
2. Ensure you have the [AWS cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) on your machine
3. Setup a profile by running `aws configure` and entering your AWS credentials
4. Run `dvc pull` from the project root directory (dvc is installed as part of the conda environment)

To add new labeled data:
1. Ensure the latest data exists locally by running `dvc pull`
2. Add the shape file for new labels into [data/raw](data/raw)
3. Open the [dataset_config.py](src/dataset_config.py) file and add a new `LabeledDataset` object into the `labeled_datasets` list and specify the required parameters.
4. Run [process.py](scripts/process.py) to process the labels into a standard format.
5. Run [export_for_labeled.py](scripts/export_for_labeled.py) to begin exporting satellite data from Google Earth Engine to your Google Drive.
6. Once the satellite data has been exported, download it from Google Drive into [data/raw](data/raw).
7. Run [engineer.py] to combine the labels and the satellite images into a machine learning suitabe dataset.
8. Run `dvc commit` and `dvc push` to upload the new labeled data to remote storage.


## Generating a Crop Map

You must have [docker](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html) and awscli installed on the machine. If doing inference and using the `--gpus all` the host machine must have accessible GPU drivers and [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) is setup.

#### Setup - Environment Variables (1 step)
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

#### Generating a Crop Map with an Existing Model (2 steps)

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
1. Gets latest docker image
2. Attaches host GPU to docker image
3. Downloads tif files from Google Drive
4. Splits tif files so they are ready for inference
5. Runs inference on each of the split files

#### Training a new model to generate crop maps (2 steps)
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
1. Gets latest docker image
2. Uses the datasets specified to train a model 
3. Logs model training to ClearML
4. Pushes trained model to dvc
5. Outputs the models.dvc file to `$MODELS_DVC_DIR`, this file needs to be git committed inorder to share the trained model with collaborators 

#### Monitoring Training and Inference
ClearML is used for monitoring training and inference during each docker build. You'll need a ClearML account and access to the ClearML workspace (contact ivan.zvonkov@gmail.com)

## General Development
#### Tests

The following tests can be run against the pipeline:

```bash
black .  # code formatting
mypy src scripts  # type checking
python -m unittest
```

#### Building the docker image locally
```
docker build -t ivanzvonkov/cropmask --secret id=aws,src=$AWS_CREDENTIALS .
```

## Already generated crop maps
Google Earth Engine:
* [Kenya (post season)](https://code.earthengine.google.com/ea3613a3a45badfd01ce2ec914dfe1ef)
* [Busia (in season)](https://code.earthengine.google.com/f567cccc28dad7a25e088d56dabfbd4c)

Zenodo
- [Kenya (post season) and Busia (in season)](https://doi.org/10.5281/zenodo.4271143).

## Acknowledgments
This model requires data from [Plant Village](https://plantvillage.psu.edu/) and [One Acre Fund](https://oneacrefund.org/). We thank those organizations for making these datasets available to us - please contact them if you are interested in accessing the data.

## Pipeline

The main entrypoints into the pipeline are the [scripts](scripts). Specifically:

* [scripts/export.py](scripts/export.py) exports data (locally, or to Google Drive - see below)
* [scripts/process.py](scripts/process.py) processes the raw data
* [scripts/engineer.py](scripts/engineer.py) combines the earth observation data with the labels to create (x, y) training data
* [scripts/models.py](scripts/model.py) trains the models
* [scripts/predict.py](scripts/predict.py) takes a trained model and runs it on an area

The [split_tiff.py](scripts/split_tiff.py) script is useful to break large exports from Google Earth Engine, which may
be too large to fit into memory.

## Reference

If you find this code useful, please cite the following paper:

Gabriel Tseng, Hannah Kerner, Catherine Nakalembe and Inbal Becker-Reshef. 2020. Annual and in-season mapping of cropland at field scale with sparse labels. Tackling Climate Change with Machine Learning workshop at NeurIPS ’20: December 11th, 2020

The hand labelled dataset used for training, and the crop maps, can be found at [https://doi.org/10.5281/zenodo.4271143](https://doi.org/10.5281/zenodo.4271143)
