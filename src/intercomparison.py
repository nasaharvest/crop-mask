from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.ETL.constants import CROP_PROB, SUBSET
import geopandas as gpd
import pandas as pd
import rasterio as rio
from pyproj import Proj
from pyproj import transform
import numpy as np
import json
import os

def run_comparison(validation_path: str, cropmap_path: str, validation_projection: str = None, cropmap_projection: str = None, subset: str = None) -> dict:
  """ 
  Returns a dictionary of intercomparison metrics. 

  Arguments:
  validation_path - filepath for validation csv
  cropmap_path - filepath for cropmap tiff file
  validaiton_projeciton -  espg projection of csv
  crop_map_projection - espg projection of tiff
  subset - subset of csv used for validation

  """
    
  # Assume datatypes to be csv and tiff for validation and cropmap respectively
  report = {}
  cropmap = rio.open(cropmap_path)
  validation = gpd.read_file(validation_path)

  # Resets csv to specified subset
  if subset != None:
    validation = validation[validation[SUBSET] == subset]

  validation[CROP_PROB] = validation[CROP_PROB].astype(float)

  # Map values to binary values
  if np.unique(validation[CROP_PROB]).size != 2:
    validation.loc[validation[CROP_PROB].astype(int) >= 0.5, CROP_PROB] = 1
    validation.loc[validation[CROP_PROB].astype(int) < 0.5, CROP_PROB] = 0
  
  pIn = Proj(init='epsg:4326')
  out = Proj(init=cropmap.crs.to_dict()['init'])

  # Reproject csv to tif projection
  validation = gpd.GeoDataFrame(data = validation[['lat', 'lon', CROP_PROB]])
  newLat, newLon = transform(pIn, out, np.array(validation['lon']), np.array(validation['lat']))

  # Sample points from tif files
  cropmap_sampled = rio.sample.sample_gen(cropmap, zip(newLat, newLon))
  cropmap_sampled = np.array([x for x in cropmap_sampled]).astype(int)

  # Adds sampled data to dataframe
  combined_maps = pd.DataFrame()
  combined_maps['validation'] = validation[CROP_PROB]
  combined_maps['cropmap'] = cropmap_sampled

  if np.unique(combined_maps['cropmap']).size != 2:
    combined_maps = combined_maps[(combined_maps.cropmap == 1) | (combined_maps.cropmap == 0)]

  # Calculate metrics
  target_names = ['non_crop', 'crop']
  class_report = classification_report(combined_maps['validation'], combined_maps['cropmap'], target_names=target_names, output_dict=True)
  accuracy = accuracy_score(combined_maps['validation'], combined_maps['cropmap'])
  f1 = f1_score(combined_maps['validation'], combined_maps['cropmap'])

  report = {
      'dataset' : cropmap_path
    .split('/')[-1],
      'f1' : f1,
      'crop precision' : class_report['crop']['precision'], 
      'crop recall' : class_report['crop']['recall'],
      'non-crop precision' : class_report['non_crop']['precision'], 
      'non-crop recall' : class_report['non_crop']['recall']
  }
    
  return report
