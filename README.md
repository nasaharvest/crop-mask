# Crop Mask Repository
This repository contains code and data to generate annual and in-season crop masks. Two models are trained - a multi-headed pixel wise classifier to classify pixels as containing crop or not, and a multi-spectral satellite image forecaster which forecasts a 12 month timeseries given a partial input:

<img src="diagrams/models.png" alt="models" height="200px"/>

These can be used to create annual and in season crop maps. 

This repository currently contains the code to do this for Kenya, and Busia county in Kenya:

<img src="diagrams/kenya_busia_maps.png" alt="models" height="400px"/>

These maps are available on Google Earth Engine:

* [Kenya (post season)](https://code.earthengine.google.com/ea3613a3a45badfd01ce2ec914dfe1ef)
* [Busia (in season)](https://code.earthengine.google.com/f567cccc28dad7a25e088d56dabfbd4c)

In addition, they are available on [Zenodo](https://doi.org/10.5281/zenodo.4271143).

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

## Setup

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.6 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `landcover-mapping` with all the necessary packages to run the code. To
activate this environment, run

```bash
conda activate landcover-mapping
```

If you are using a GPU, `environment.gpu.yml` additionally installs `cudatoolkit` so that pytorch can use it too.

#### Earth Engine

Earth engine is used to export data. To use it, once the conda environment has been activated, run

```bash
earthengine authenticate
```

and follow the instructions. To test that everything has worked, run

```bash
python -c "import ee; ee.Initialize()"
```

Note that Earth Engine exports files to Google Drive by default (to the same google account used sign up to Earth Engine).

Running exports can be viewed (and individually cancelled) in the `Tabs` bar on the [Earth Engine Code Editor](https://code.earthengine.google.com/).

Exports from Google Drive should be saved in [`data/raw`](data/raw).

#### Accessing the data
This project uses an AWS S3 bucket to store the training data and `dvc` to manage and fetch the training data. 

To have access to training data:
1. Obtain valid NASAHarvest AWS credentials (access key, secret key)
2. Ensure you have the [AWS cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) on your machine
3. Setup a named profile by running `aws configure` and entering your AWS credentials

To pull the latest training:
1. Ensure you have [dvc](https://dvc.org/doc) installed
2. Run `dvc pull` from the project root directory

#### Tests

The following tests can be run against the pipeline:

```bash
black .  # code formatting
mypy src scripts  # type checking
python -m unittest
```

## Using with docker
#### Prerequisites
You must have docker and awscli installed on the machine.

Prior to building any docker image execute:
```
source setup.sh
```
This command sets up the necessary credentials.

#### Training a model
You must specify the build arguments: `DATASETS` and `MODEL_NAME`.  
The credential environment variables have been specified by the setup.sh script.
```
export DATASETS="geowiki_landcover_2017,kenya_non_crop,one_acre_fund_kenya,plant_village_kenya"
export MODEL_NAME="kenya"
docker build -f Dockerfile.train \
  -t cropmask/train \
  --build-arg DATASETS=$DATASETS \
  --build-arg MODEL_NAME=$MODEL_NAME \
  --secret id=aws,src=$AWS_CREDENTIALS \
  --secret id=clearml,src=$CLEARML_CREDENTIALS \
  --output data \
.
```
This command does the following:
- Sets up the environment for training
- Pulls in the necessary data
- Trains the model and logs to ClearML
- Uploads the model to remote storage
- Produces a models.dvc file which must be git committed to make the model accessible to collaborators

#### Inference
**Obtaining the inference image**
You can either pull the docker image from DockerHub
```

```
OR build it yourself with:
```
docker build -f Dockerfile.inference -t cropmask/inference --secret id=aws,src=$AWS_CREDENTIALS .
```

**Running inference:**
```
export MODEL_NAME="kenya"
export EBS_VOLUME="/data"
export GDRIVE_DIR="remote2:nasaharvest"
docker run \
  -v $RCLONE_CREDENTIALS:/root/.config/rclone/rclone.conf \
  -v $HOME/clearml.conf:/root/clearml.conf \
  --mount type=bind,source=$EBS_VOLUME,target=/vol \
  -it cropmask/inference \
  --model_name $MODEL_NAME \
  --gdrive_path_to_tif_files $GDRIVE_DIR
```


#### Monitoring Training and Inference
ClearML is used for monitoring training and inference during each docker build. You'll need a ClearML account and access to the ClearML workspace (contact ivan.zvonkov@gmail.com)

## Reference

If you find this code useful, please cite the following paper:

Gabriel Tseng, Hannah Kerner, Catherine Nakalembe and Inbal Becker-Reshef. 2020. Annual and in-season mapping of cropland at field scale with sparse labels. Tackling Climate Change with Machine Learning workshop at NeurIPS ’20: December 11th, 2020

The hand labelled dataset used for training, and the crop maps, can be found at [https://doi.org/10.5281/zenodo.4271143](https://doi.org/10.5281/zenodo.4271143)
