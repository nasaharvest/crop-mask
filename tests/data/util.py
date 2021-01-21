import gdal
import osr

def generate_empty_tif():
    driver = gdal.GetDriverByName('GTiff')

    spatref = osr.SpatialReference()
    spatref.ImportFromEPSG(27700)
    wkt = spatref.ExportToWkt()

    outfn = 'tif_files/empty.tif'
    num_bands = 13
    nodata = 255
    xres = 5
    yres = -5

    xmin = 0
    xmax = 5
    ymin = 0
    ymax = 5
    dtype = gdal.GDT_Int16

    xsize = abs(int((xmax - xmin) / xres))
    ysize = abs(int((ymax - ymin) / yres))

    ds = driver.Create(outfn, xsize, ysize, num_bands, dtype, options=['COMPRESS=LZW', 'TILED=YES'])
    ds.SetProjection(wkt)
    ds.SetGeoTransform([xmin, xres, 0, ymax, 0, yres])
    ds.GetRasterBand(1).Fill(0)
    ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds.FlushCache()


if __name__ == '__main__':
    generate_empty_tif()
