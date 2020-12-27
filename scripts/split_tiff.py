"""
https://gis.stackexchange.com/questions/306861/split-geotiff-into-multiple-cells-with-
"""

from shapely import geometry
from pathlib import Path
import math
import rasterio
from rasterio.mask import mask


# Takes a  dataset and splits it into squares of dimensions squareDim * squareDim
def splitImageIntoCells(img_path: Path, filename: str, squareDim: int, output_folder: Path):

    img = rasterio.open(img_path)
    numberOfCellsWide = math.ceil(img.shape[1] / squareDim)
    numberOfCellsHigh = math.ceil(img.shape[0] / squareDim)
    count = 0
    for hc in range(numberOfCellsHigh):
        y = min(hc * squareDim, img.shape[0])
        for wc in range(numberOfCellsWide):
            x = min(wc * squareDim, img.shape[1])
            geom = getTileGeom(img.transform, x, y, squareDim)
            getCellFromGeom(img, geom, filename, count, output_folder)
            count = count + 1


# Generate a bounding box from the pixel-wise coordinates using the original datasets transform property
def getTileGeom(transform, x, y, squareDim):
    corner1 = (x, y) * transform
    corner2 = (x + squareDim, y + squareDim) * transform
    return geometry.box(corner1[0], corner1[1], corner2[0], corner2[1])


# Crop the dataset using the generated box and write it out as a GeoTIFF
def getCellFromGeom(img, geom, filename, count, output_folder):
    crop, cropTransform = mask(img, [geom], crop=True)
    writeImageAsGeoTIFF(
        crop, cropTransform, img.meta, img.crs, f"{count}-{filename}", output_folder
    )


# Write the passed in dataset as a GeoTIFF
def writeImageAsGeoTIFF(img, transform, metadata, crs, filename, output_folder):
    metadata.update(
        {
            "driver": "GTiff",
            "height": img.shape[1],
            "width": img.shape[2],
            "transform": transform,
            "crs": crs,
        }
    )
    with rasterio.open(output_folder / f"{filename}.tif", "w", **metadata) as dest:
        dest.write(img)


if __name__ == "__main__":

    images = Path("PATH_TO_TIF_FILES").glob("*.tif")
    output_folder = Path("PATH_TO_SAVE_FOLDER")
    for idx, image in enumerate(images):

        print(f"Splitting {image}")

        name, start_date, end_bit = image.name.split("_")
        end_date = end_bit[:10]
        tile_identifier = end_bit[10:-4]

        new_filename = f"{idx}-{name}{tile_identifier}_{start_date}_{end_date}"

        splitImageIntoCells(image, new_filename, 1000, output_folder)

        print(f"Finished {image}. Removing the original file")
        image.unlink()
