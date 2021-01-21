from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
import xarray as xr
import gdal
import osr
import tempfile
import shutil
from scripts.predict import make_prediction


def generate_empty_tif(file_path: Path) -> Path:
    driver = gdal.GetDriverByName('GTiff')
    spatref = osr.SpatialReference()
    spatref.ImportFromEPSG(27700)
    wkt = spatref.ExportToWkt()
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

    ds = driver.Create(str(file_path), xsize, ysize, num_bands, dtype, options=['COMPRESS=LZW', 'TILED=YES'])
    ds.SetProjection(wkt)
    ds.SetGeoTransform([xmin, xres, 0, ymax, 0, yres])
    ds.GetRasterBand(1).Fill(0)
    ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds.FlushCache()
    return file_path


class TestPredict(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.empty_tif = generate_empty_tif(cls.temp_dir / 'empty.tif')
        cls.forecasted_path = cls.temp_dir / f'preds_forecasted_{cls.empty_tif.name}.nc'
        cls.normal_path = cls.temp_dir / f'preds_normal_{cls.empty_tif.name}.nc'

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

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
        # Figuring out if gdal_merge should have an argument for forecasted vs normal
        print('hello')
