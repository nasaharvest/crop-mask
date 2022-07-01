import sklearn.metrics
import geopandas as gpd
import pandas as pd
import rasterio as rio
from pyproj import Proj
from pyproj import transform
import numpy as np
import json
import os

def run_comparison(validationPath: str, cropmapPath: str, validation_projection = None, cropmap_projection: str = None, asJSON = False, subset: str = None):
    
  # Assume datatypes to be csv and tiff for validation and cropmap respectively
  report = []
  cropmap = rio.open(cropmapPath)
  validation = gpd.read_file(validationPath)

  # Resets csv to specified subset
  if subset != None:
    validation = validation[validation['subset'] == subset]

  validation['crop_probability'] = validation['crop_probability'].astype(float)

  # Map values to binary values
  if np.unique(validation['crop_probability']).size != 2:
    validation.loc[validation['crop_probability'].astype(int) >= 0.5, 'crop_probability'] = 1
    validation.loc[validation['crop_probability'].astype(int) < 0.5, 'crop_probability'] = 0
  

  pIn = Proj(init='epsg:4326')
  out = Proj(init=cropmap.crs.to_dict()['init'])

  # Reproject csv to tif projection
  validation = gpd.GeoDataFrame(data = validation[['lat', 'lon', 'crop_probability']])
  newLat, newLon = transform(pIn, out, np.array(validation['lon']), np.array(validation['lat']))

  # Sample points from tif files
  cropmap_sampled = rio.sample.sample_gen(cropmap, zip(newLat, newLon))
  cropmap_sampled = np.array([x for x in cropmap_sampled]).astype(int)

  # Adds sampled data to dataframe
  combined_maps = pd.DataFrame()
  combined_maps['validation'] = validation['crop_probability']
  combined_maps['cropmap'] = cropmap_sampled

  if np.unique(combined_maps['cropmap']).size != 2:
    combined_maps = combined_maps[(combined_maps.cropmap == 1) | (combined_maps.cropmap == 0)]

  # Calculate metrics
  target_names = ['non_crop', 'crop']
  class_report = sklearn.metrics.classification_report(combined_maps['validation'], combined_maps['cropmap'], target_names=target_names, output_dict=True)
  accuracy = sklearn.metrics.accuracy_score(combined_maps['validation'], combined_maps['cropmap'])
  f1 = sklearn.metrics.f1_score(combined_maps['validation'], combined_maps['cropmap'])

  if asJSON:
    report = {
        'dataset' : cropmapPath.split('/')[-1],
        'f1' : f1,
        'crop precision' : class_report['crop']['precision'], 
        'crop recall' : class_report['crop']['recall'],
        'non-crop precision' : class_report['non_crop']['precision'], 
        'non-crop recall' : class_report['non_crop']['recall']
    }
  else: 
    report = [f1, accuracy, class_report['crop']['precision'], class_report['non_crop']['precision'], class_report['non_crop']['recall'], class_report['crop']['recall']]

  return report
