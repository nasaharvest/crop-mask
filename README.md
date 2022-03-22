# Crop Map Generation

[![Status](https://github.com/nasaharvest/crop-mask/actions/workflows/test.yml/badge.svg)](https://github.com/nasaharvest/crop-mask/actions/workflows/test.yml) [![Status](https://github.com/nasaharvest/crop-mask/actions/workflows/deploy.yml/badge.svg)](https://github.com/nasaharvest/crop-mask/actions/workflows/deploy.yml)

End-to-end workflow for generating high resolution cropland maps.

![Cropland gif](assets/cropmask.gif)

## Contents
-   [Training a new model](#training-a-new-model)
-   [Creating a crop map](#creating-a-crop-map)
-   [Setting up a local environment](#setting-up-a-local-environment)
-   [Adding new labeled data](#adding-new-labeled-data)
-   [Tests](#tests)
-   [Previously generated crop maps](#previously-generated-crop-maps)
-   [Acknowledgments](#acknowledgments)
-   [Reference](#reference)

## Training a new model
To train a new model run the following colab notebook (or use it as a guide):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/crop-mask/blob/master/notebooks/inference.ipynb)

Two models are trained - a multi-headed pixel wise classifier to classify pixels as containing crop or not, and a multi-spectral satellite image forecaster which forecasts a 12 month timeseries given a partial input:

<img src="assets/models.png" alt="models" height="200px"/>

## Creating a crop map
To create a crop map run the following colab notebook (or use it as a guide): 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/crop-mask/blob/master/notebooks/inference.ipynb)

## Setting up a local environment
Ensure you have [anaconda](https://www.anaconda.com/download/#macos) installed.
#### 1. For development 
Ensure you have [gcloud](https://cloud.google.com/sdk/docs/install) installed.
```bash
conda install mamba -n base -c conda-forge  # Install mamba
mamba env create -f environment-dev.yml     # Create environment with mamba (faster)
conda activate landcover-mapping            # Activate environment
gcloud auth application-default login       # Authenticates with Google Cloud
```

#### 2. For shapefile notebook
```bash
conda env create -f environment-lite.yml    # Create environment
conda activate landcover-lite               # Activate environment
jupyter notebook
```

## Adding new labeled data
Ensure the local environment is setup.
1. Add the csv or shape file for new labels into [data/raw](data/raw)
2. In [dataset_labeled.py](src/datasets_labeled.py) add a new `LabeledDataset` object into the `labeled_datasets` list and specify the required parameters.
```bash
# Activate environment setup in: Setting up a local environment
conda activate landcover-mapping 

dvc pull                                    # Get latest data from dvc
earthengine authenticate                    # Authenticates Earth Engine 
python -c "import ee; ee.Initialize()"      # Will raise error if not setup

# Pull in latest EarthEngine tifs (you may need to rerun this command)
gsutil -m cp -n -r gs://crop-mask-tifs2/tifs data/

# Create features (you may need to rerun this command)
python scripts/create_features.py

dvc commit                                  # Save new features to repository
dvc push                                    # Push features to remote storage

# Push changes to github
git checkout -b'new-Ethiopia-Tigray-data'
git add .
git commit -m 'Added mew Ethiopia Tigray data for 2020'
git push
```

## Tests

The following tests can be run against the pipeline:

```bash
flake8 . # code formatting
mypy .  # type checking
python -m unittest # unit tests

# Integration tests
python -m unittest test/integration_test_labeled.py
python -m unittest test/integration_test_model_bbox.py
python -m unittest test/integration_test_model_evaluation.py
```

## Previously generated crop maps

Google Earth Engine:

-   [Kenya (post season)](https://code.earthengine.google.com/ea3613a3a45badfd01ce2ec914dfe1ef)
-   [Busia (in season)](https://code.earthengine.google.com/f567cccc28dad7a25e088d56dabfbd4c)

Zenodo

-   [Kenya (post season) and Busia (in season)](https://doi.org/10.5281/zenodo.4271143).

## Reference

If you find this code useful, please cite the following paper:

Gabriel Tseng, Hannah Kerner, Catherine Nakalembe and Inbal Becker-Reshef. 2020. Annual and in-season mapping of cropland at field scale with sparse labels. Tackling Climate Change with Machine Learning workshop at NeurIPS ’20: December 11th, 2020
