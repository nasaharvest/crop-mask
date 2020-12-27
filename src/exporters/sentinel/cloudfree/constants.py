# These are algorithm settings for the cloud filtering algorithm
image_collection = "COPERNICUS/S2"

# Ranges from 0-1.Lower value will mask more pixels out.
# Generally 0.1-0.3 works well with 0.2 being used most commonly
cloudThresh = 0.2
# Height of clouds to use to project cloud shadows
cloudHeights = [200, 10000, 250]
# Sum of IR bands to include as shadows within TDOM and the
# shadow shift method (lower number masks out less)
irSumThresh = 0.3
ndviThresh = -0.1
# Pixels to reduce cloud mask and dark shadows by to reduce inclusion
# of single-pixel comission errors
erodePixels = 1.5
dilationPixels = 3

# images with less than this many cloud pixels will be used with normal
# mosaicing (most recent on top)
cloudFreeKeepThresh = 3

BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
