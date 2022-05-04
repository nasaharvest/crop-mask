from typing import List
import sklearn.metrics
import geopandas as gpd
import rasterio as ras
from pyproj import Proj
from pyproj import transform
import numpy as np


import fiona
fiona.supported_drivers
fiona.drvsupport.supported_drivers['csv'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['CSV'] = 'rw' # enable KML support which is disabled by default

def run_comparison(validation: str, cropmap: str, validation_projection = None, cropmap_projection: str = None):
    #assume datatypes to be csv and tiff for validation and cropmap respectively
    report = []
    cropmap = ras.open(cropmap)
    validation = gpd.read_file(validation)

    validation = validation[validation['subset'] == 'testing']
    validation['crop_probability'] = validation['crop_probability'].astype(float)
    validation.loc[validation['crop_probability'].astype(float) >= 0.5, 'crop_probability'] = 1
    validation.loc[validation['crop_probability'].astype(float) < 0.5, 'crop_probability'] = 0
    validation = validation[(validation.crop_probability == 1) | (validation.crop_probability == 0)]

    pIn = Proj(init="epsg:4326")
    out = Proj(init=cropmap.crs.to_dict()['init'])

    validation = gpd.GeoDataFrame(data = validation[['lat', 'lon', 'crop_probability']])

    newLat, newLon = transform(pIn, out, np.array(validation['lon']), np.array(validation['lat']))

    cropmap_sampled = ras.sample.sample_gen(cropmap, zip(newLat, newLon))
    cropmap_sampled = np.array([x for x in cropmap_sampled])

    print(cropmap_sampled)

    target_names = ['non_crop', 'crop']
    class_report = sklearn.metrics.classification_report(np.array(validation['crop_probability']), cropmap_sampled, target_names=target_names, output_dict=True)
    accuracy = sklearn.metrics.accuracy_score(np.array(validation['crop_probability']), cropmap_sampled)

    report = [accuracy, class_report['crop']['precision'], class_report['non_crop']['precision'], class_report['non_crop']['recall'], class_report['crop']['recall']]

    return report

