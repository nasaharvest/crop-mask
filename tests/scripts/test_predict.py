from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
import xarray as xr
import gdal
import osr
import tempfile
import shutil
from typing import Tuple
from scripts.predict import make_prediction, gdal_merge


def generate_gtiff(file_path: Path, bbox: Tuple[int, int, int, int] = (0,10, 0,10), resolution: int = 10) -> Path:
    """
    Utility function for generating tif files for testing purposes
    """
    driver = gdal.GetDriverByName('GTiff')
    spatref = osr.SpatialReference()
    spatref.ImportFromEPSG(27700)
    wkt = spatref.ExportToWkt()
    num_bands = 13
    xres, yres = resolution, -resolution
    xmin, xmax, ymin, ymax = bbox
    dtype = gdal.GDT_Int16

    xsize = abs(int((xmax - xmin) / xres))
    ysize = abs(int((ymax - ymin) / yres))

    ds = driver.Create(
        str(file_path),
        xsize,
        ysize,
        num_bands,
        dtype,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    ds.SetProjection(wkt)
    ds.SetGeoTransform([xmin, xres, 0, ymax, 0, yres])
    ds.GetRasterBand(1).Fill(0)
    ds.GetRasterBand(1).SetNoDataValue(255)
    ds.FlushCache()
    return file_path


class TestPredict(TestCase):
    """Tests the predict script"""

    temp_dir = None

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.empty_tif = generate_gtiff(cls.temp_dir / 'empty.tif')
        cls.forecasted_path = cls.predicted_file_path(cls.empty_tif.name, with_forecaster=True)
        cls.normal_path = cls.predicted_file_path(cls.empty_tif.name, with_forecaster=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    @classmethod
    def predicted_file_path(cls, origin_file: str, with_forecaster=False):
        if with_forecaster:
            prefix = 'forecasted'
        else:
            prefix = 'normal'
        return cls.temp_dir / f'preds_{prefix}_{origin_file}.nc'

    @patch('src.models.Model')
    @patch('scripts.predict.plot_results')
    def test_make_prediction_normal(self, mock_plot_results, mock_model):
        mock_model.predict.return_value = xr.Dataset(None)
        output_path = make_prediction(model=mock_model,
                                      test_path=self.empty_tif,
                                      save_dir=self.temp_dir)
        mock_model.predict.assert_called()
        mock_plot_results.assert_not_called()
        self.assertEqual(output_path, self.normal_path)
        self.assertTrue(output_path.exists())
        output_path.unlink()

    @patch('src.models.Model')
    @patch('scripts.predict.plot_results')
    def test_make_prediction_forecasted(self, mock_plot_results, mock_model):
        mock_model.predict.return_value = xr.Dataset(None)
        output_path = make_prediction(model=mock_model,
                                      test_path=self.empty_tif,
                                      save_dir=self.temp_dir,
                                      with_forecaster=True,
                                      plot_results_enabled=True)
        mock_model.predict.assert_called()
        mock_plot_results.assert_called()
        self.assertEqual(output_path, self.forecasted_path)
        self.assertTrue(output_path.exists())
        self.forecasted_path.unlink()

    @patch('src.models.Model')
    @patch('scripts.predict.plot_results')
    def test_make_prediction_already_exists(self, mock_plot_results, mock_model):
        self.normal_path.touch()
        output_path = make_prediction(model=mock_model,
                                      test_path=self.empty_tif,
                                      save_dir=self.temp_dir)
        mock_model.predict.assert_not_called()
        mock_plot_results.assert_not_called()
        self.assertEqual(output_path, None)
        self.normal_path.unlink()

    def test_gdal_merge(self):
        for with_forecaster in [False, True]:
            # Setup
            path_1, path_2 = (self.predicted_file_path(name, with_forecaster) for name in ['1.tif', '2.tif'])
            generate_gtiff(path_1, bbox=(0,10,0,10))
            self.assertTrue(path_1.exists())
            generate_gtiff(path_2, bbox=(10,20,0,10))
            self.assertTrue(path_2.exists())

            # Use gdal_merge
            merged_path = gdal_merge(save_dir=self.temp_dir, with_forecaster=with_forecaster)

            # Verify merged gtiff
            prefix = {"forecasted" if with_forecaster else "normal"}
            self.assertEqual(merged_path, Path(self.temp_dir / f'merged_{prefix}.tif'))
            self.assertTrue(merged_path.exists())
            merged_file = gdal.Open(str(merged_path))
            ulx, xres, _, uly, _, yres  = merged_file.GetGeoTransform()
            lrx = ulx + (merged_file.RasterXSize * xres)
            lry = uly + (merged_file.RasterYSize * yres)
            self.assertEqual((ulx, uly, lrx, lry), (0,10,20,0))

            # Clean up
            for p in [path_1, path_2, merged_path]:
                p.unlink()

    def test_main(self):
        
