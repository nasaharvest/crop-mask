from typing import List
import geopandas as gpd
import numpy as np
import sklearn.metrics
import rasterio as ras
from pyproj import Proj, transform

import fiona
fiona.supported_drivers
fiona.drvsupport.supported_drivers['csv'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['CSV'] = 'rw' # enable KML support which is disabled by default

def run_comparison(validation: str, cropmap: str, validation_projection = None, cropmap_projection: str = None):
    #assume datatypes to be csv and tiff for validation and cropmap respectively
    report = []
    cropmap = ras.open(cropmap)
    validation = gpd.read_file(validation)

    if (validation_projection != None):
        validation.crs = validation_projection

    if (cropmap_projection != None):
        cropmap.crs = Proj(cropmap_projection)

    validation = gpd.GeoDataFrame(data = validation[['lat', 'lon', 'crop_probability']], crs = validation.crs)

    if (Proj(cropmap.crs) != Proj(validation.crs)):
        newLat, newLon = reproject(validation['lat'].values, validation['lon'].values, Proj(validation.crs), Proj(cropmap.crs))

    newLat, newLon = reproject(validation['lat'].values, validation['lon'].values, Proj('+init=epsg:4326'), Proj(cropmap.crs))

    cropmap_sampled = ras.sample.sample_gen(cropmap, ((float(x),float(y)) for x,y in zip(validation['lat'], validation['lon'])))

    target_names = ['non_crop', 'crop']
    class_report = sklearn.metrics.classification_report(validation['crop_probability'], cropmap_sampled['cropmask'], target_names = target_names, output_dict=True)
    accuracy = sklearn.metrics.accuracy_score(validation['crop_probability'], cropmap_sampled)

    report = [accuracy, class_report['crop']['precision'], class_report['non_crop']['precision'], class_report['non_crop']['recall'], class_report['crop']['recall']]

    return report

def reproject(latIn, lonIn, ProjIn, ProjOut) -> np.array:
    if latIn.size != latIn.size:
        raise Exception("Input arrays are not of equal size.")

    latOut = []
    lonOut = []

    for coordinate in np.arange(latIn.size):
        x1,y1 = latIn[coordinate],lonIn[coordinate]
        newLat, newLon = transform(ProjIn,ProjOut,x1,y1)
        latOut.append(newLat), lonOut.append(newLon)
    
    return np.array(latOut), np.array(lonOut)
