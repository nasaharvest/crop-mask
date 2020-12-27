import ee
from datetime import date
import math

from .constants import (
    cloudFreeKeepThresh,
    cloudHeights,
    cloudThresh,
    ndviThresh,
    irSumThresh,
    erodePixels,
    dilationPixels,
    image_collection,
)
from .utils import date_to_string, rescale


def get_single_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:

    dates = ee.DateRange(date_to_string(start_date), date_to_string(end_date),)

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()
    imgC = ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region)

    imgC = (
        imgC.map(lambda x: x.clip(region))
        .map(lambda x: x.set("ROI", region))
        .map(computeS2CloudScore)
        .map(calcCloudStats)
        .map(projectShadows)
        .map(computeQualityScore)
        .sort("CLOUDY_PERCENTAGE")
    )

    cloudFree = mergeCollection(imgC)

    return cloudFree


def calcCloudStats(img):
    imgPoly = ee.Algorithms.GeometryConstructors.Polygon(
        ee.Geometry(img.get("system:footprint")).coordinates()
    )

    roi = ee.Geometry(img.get("ROI"))

    intersection = roi.intersection(imgPoly, ee.ErrorMargin(0.5))
    cloudMask = img.select(["cloudScore"]).gt(cloudThresh).clip(roi).rename("cloudMask")

    cloudAreaImg = cloudMask.multiply(ee.Image.pixelArea())

    stats = cloudAreaImg.reduceRegion(
        **{"reducer": ee.Reducer.sum(), "geometry": roi, "scale": 10, "maxPixels": 1e12}
    )

    cloudPercent = ee.Number(stats.get("cloudMask")).divide(imgPoly.area()).multiply(100)
    coveragePercent = ee.Number(intersection.area()).divide(roi.area()).multiply(100)
    cloudPercentROI = ee.Number(stats.get("cloudMask")).divide(roi.area()).multiply(100)

    img = img.set("CLOUDY_PERCENTAGE", cloudPercent)
    img = img.set("ROI_COVERAGE_PERCENT", coveragePercent)
    img = img.set("CLOUDY_PERCENTAGE_ROI", cloudPercentROI)

    return img


def computeQualityScore(img):
    score = img.select(["cloudScore"]).max(img.select(["shadowScore"]))

    score = score.reproject("EPSG:4326", None, 20).reduceNeighborhood(
        **{"reducer": ee.Reducer.mean(), "kernel": ee.Kernel.square(5)}
    )

    score = score.multiply(-1)

    return img.addBands(score.rename("cloudShadowScore"))


def computeS2CloudScore(img):
    toa = img.select(
        ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12",]
    ).divide(10000)

    toa = toa.addBands(img.select(["QA60"]))

    # ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A',
    #  'B9',          'B10', 'B11','B12']
    # ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2',
    #  'waterVapor', 'cirrus','swir1', 'swir2']);

    # Compute several indicators of cloudyness and take the minimum of them.
    score = ee.Image(1)

    # Clouds are reasonably bright in the blue and cirrus bands.
    score = score.min(rescale(toa, "img.B2", [0.1, 0.5]))
    score = score.min(rescale(toa, "img.B1", [0.1, 0.3]))
    score = score.min(rescale(toa, "img.B1 + img.B10", [0.15, 0.2]))

    # Clouds are reasonably bright in all visible bands.
    score = score.min(rescale(toa, "img.B4 + img.B3 + img.B2", [0.2, 0.8]))

    # Clouds are moist
    ndmi = img.normalizedDifference(["B8", "B11"])
    score = score.min(rescale(ndmi, "img", [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = img.normalizedDifference(["B3", "B11"])
    score = score.min(rescale(ndsi, "img", [0.8, 0.6]))

    # Clip the lower end of the score
    score = score.max(ee.Image(0.001))

    # score = score.multiply(dilated)
    score = score.reduceNeighborhood(
        **{"reducer": ee.Reducer.mean(), "kernel": ee.Kernel.square(5)}
    )

    return img.addBands(score.rename("cloudScore"))


def projectShadows(image):
    meanAzimuth = image.get("MEAN_SOLAR_AZIMUTH_ANGLE")
    meanZenith = image.get("MEAN_SOLAR_ZENITH_ANGLE")

    cloudMask = image.select(["cloudScore"]).gt(cloudThresh)

    # Find dark pixels
    darkPixelsImg = image.select(["B8", "B11", "B12"]).divide(10000).reduce(ee.Reducer.sum())

    ndvi = image.normalizedDifference(["B8", "B4"])
    waterMask = ndvi.lt(ndviThresh)

    darkPixels = darkPixelsImg.lt(irSumThresh)

    # Get the mask of pixels which might be shadows excluding water
    darkPixelMask = darkPixels.And(waterMask.Not())
    darkPixelMask = darkPixelMask.And(cloudMask.Not())

    # Find where cloud shadows should be based on solar geometry
    # Convert to radians
    azR = ee.Number(meanAzimuth).add(180).multiply(math.pi).divide(180.0)
    zenR = ee.Number(meanZenith).multiply(math.pi).divide(180.0)

    # Find the shadows
    def getShadows(cloudHeight):
        cloudHeight = ee.Number(cloudHeight)

        shadowCastedDistance = zenR.tan().multiply(cloudHeight)  # Distance shadow is cast
        x = azR.sin().multiply(shadowCastedDistance).multiply(-1)  # /X distance of shadow
        y = azR.cos().multiply(shadowCastedDistance).multiply(-1)  # Y distance of shadow
        return image.select(["cloudScore"]).displace(
            ee.Image.constant(x).addBands(ee.Image.constant(y))
        )

    shadows = ee.List(cloudHeights).map(getShadows)
    shadowMasks = ee.ImageCollection.fromImages(shadows)
    shadowMask = shadowMasks.mean()

    # Create shadow mask
    shadowMask = dilatedErossion(shadowMask.multiply(darkPixelMask))

    shadowScore = shadowMask.reduceNeighborhood(
        **{"reducer": ee.Reducer.max(), "kernel": ee.Kernel.square(1)}
    )

    image = image.addBands(shadowScore.rename(["shadowScore"]))

    return image


def dilatedErossion(score):
    # Perform opening on the cloud scores
    score = (
        score.reproject("EPSG:4326", None, 20)
        .focal_min(**{"radius": erodePixels, "kernelType": "circle", "iterations": 3})
        .focal_max(**{"radius": dilationPixels, "kernelType": "circle", "iterations": 3})
        .reproject("EPSG:4326", None, 20)
    )

    return score


def mergeCollection(imgC):
    # Select the best images, which are below the cloud free threshold, sort them in reverse order
    # (worst on top) for mosaicing
    best = imgC.filterMetadata("CLOUDY_PERCENTAGE", "less_than", cloudFreeKeepThresh).sort(
        "CLOUDY_PERCENTAGE", False
    )
    filtered = imgC.qualityMosaic("cloudShadowScore")

    # Add the quality mosaic to fill in any missing areas of the ROI which aren't covered by good
    # images
    newC = ee.ImageCollection.fromImages([filtered, best.mosaic()])

    return ee.Image(newC.mosaic())
