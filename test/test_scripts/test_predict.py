from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
import xarray as xr
import gdal
import osr
import tempfile
import shutil
from typing import Tuple
from scripts.predict import make_prediction, gdal_merge, run_inference


class TestPredict(TestCase):
    """Tests the predict script"""

    temp_dir: Path = Path("")

    @staticmethod
    def generate_gtiff(
        file_path: Path, bbox: Tuple[int, int, int, int] = (0, 10, 0, 10), resolution: int = 10
    ) -> Path:
        """
        Utility function for generating tif files for testing purposes
        """
        driver = gdal.GetDriverByName("GTiff")
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
            str(file_path), xsize, ysize, num_bands, dtype, options=["COMPRESS=LZW", "TILED=YES"]
        )
        ds.SetProjection(wkt)
        ds.SetGeoTransform([xmin, xres, 0, ymax, 0, yres])
        ds.GetRasterBand(1).Fill(0)
        ds.GetRasterBand(1).SetNoDataValue(255)
        ds.FlushCache()
        return file_path

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.input_dir = cls.temp_dir / "input"
        cls.input_dir.mkdir(exist_ok=True)
        cls.empty_tif = cls.generate_gtiff(cls.input_dir / "empty.tif")
        cls.forecasted_path = cls.predicted_file_path(cls.empty_tif.name, "forecasted")
        cls.normal_path = cls.predicted_file_path(cls.empty_tif.name, "normal")
        cls.forecasted_path.parent.mkdir(exist_ok=True)
        cls.normal_path.parent.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    @classmethod
    def tearDown(cls):
        paths = cls.temp_dir.glob("**/*")
        for p in paths:
            if p.is_file() and p != cls.empty_tif:
                p.unlink()

    @classmethod
    def predicted_file_path(cls, origin_file: str, prefix: str) -> Path:
        return cls.temp_dir / prefix / f"preds_{prefix}_{origin_file}.nc"

    @patch("src.models.Model")
    @patch("scripts.predict.plot_results")
    def test_make_prediction_normal(self, mock_plot_results, mock_model):
        mock_model.predict.return_value = xr.Dataset(None)
        output_path = make_prediction(
            model=mock_model, test_path=self.empty_tif, save_dir=self.temp_dir / "normal"
        )
        mock_model.predict.assert_called()
        mock_plot_results.assert_not_called()
        self.assertEqual(output_path, self.normal_path)
        self.assertTrue(output_path.exists())

    @patch("src.models.Model")
    @patch("scripts.predict.plot_results")
    def test_make_prediction_forecasted(self, mock_plot_results, mock_model):
        mock_model.predict.return_value = xr.Dataset(None)
        output_path = make_prediction(
            model=mock_model,
            test_path=self.empty_tif,
            save_dir=self.temp_dir / "forecasted",
            with_forecaster=True,
            plot_results_enabled=True,
        )
        mock_model.predict.assert_called()
        mock_plot_results.assert_called()
        self.assertEqual(output_path, self.forecasted_path)
        self.assertTrue(output_path.exists())

    @patch("src.models.Model")
    @patch("scripts.predict.plot_results")
    def test_make_prediction_already_exists(self, mock_plot_results, mock_model):
        self.normal_path.touch()
        output_path = make_prediction(
            model=mock_model, test_path=self.empty_tif, save_dir=self.temp_dir / "normal"
        )
        mock_model.predict.assert_not_called()
        mock_plot_results.assert_not_called()
        self.assertEqual(output_path, None)

    def test_gdal_merge(self):
        for prefix in ["forecasted", "normal"]:
            # Setup
            path_1, path_2 = (self.predicted_file_path(name, prefix) for name in ["1.tif", "2.tif"])
            self.generate_gtiff(path_1, bbox=(0, 10, 0, 10))
            self.assertTrue(path_1.exists())
            self.generate_gtiff(path_2, bbox=(10, 20, 0, 10))
            self.assertTrue(path_2.exists())

            # Use gdal_merge
            merged_path = gdal_merge(
                unmerged_tifs_folder=self.temp_dir / prefix,
                output_file=self.temp_dir / f"merged_{prefix}.tif",
            )

            # Verify merged gtiff
            self.assertEqual(merged_path, Path(self.temp_dir / f"merged_{prefix}.tif"))
            self.assertTrue(merged_path.exists())
            merged_file = gdal.Open(str(merged_path))
            ulx, xres, _, uly, _, yres = merged_file.GetGeoTransform()
            lrx = ulx + (merged_file.RasterXSize * xres)
            lry = uly + (merged_file.RasterYSize * yres)
            self.assertEqual((ulx, uly, lrx, lry), (0, 10, 20, 0))

    def test_run_inference_error(self):
        self.assertRaises(
            ValueError,
            run_inference,
            local_path_to_tif_files=str(self.temp_dir),
            model_name="mock_model",
            data_dir=str(self.temp_dir),
            predict_without_forecaster=False,
            predict_with_forecaster=False,
        )

    @patch("src.models.Model.load_from_checkpoint")
    @patch("scripts.predict.make_prediction")
    def test_run_inference_no_merge(self, mock_make_prediction, mock_load_from_checkpoint):
        run_inference(
            local_path_to_tif_files=str(self.temp_dir / "input"),
            model_name="mock_model",
            data_dir=str(self.temp_dir),
            predict_dir=str(self.temp_dir),
        )
        mock_load_from_checkpoint.assert_called()
        self.assertEqual(mock_make_prediction.call_count, 2)

    @patch("src.models.Model.load_from_checkpoint")
    @patch("scripts.predict.make_prediction")
    @patch("scripts.predict.gdal_merge")
    def test_run_inference_with_merge(
        self, mock_gdal_merge, mock_make_prediction, mock_load_from_checkpoint
    ):
        run_inference(
            local_path_to_tif_files=str(self.temp_dir / "input"),
            model_name="mock_model",
            data_dir=str(self.temp_dir),
            predict_dir=str(self.temp_dir),
            merge_predictions=True,
        )
        mock_load_from_checkpoint.assert_called()
        self.assertEqual(mock_make_prediction.call_count, 2)
        self.assertEqual(mock_gdal_merge.call_count, 2)

    @patch("src.models.Model.load_from_checkpoint")
    @patch("scripts.predict.make_prediction")
    @patch("scripts.predict.gdal_merge")
    def test_run_inference_with_merge_and_upload(
        self, mock_gdal_merge, mock_make_prediction, mock_load_from_checkpoint
    ):
        mock_gdal_merge.return_value = Path("mock_path_1")
        run_inference(
            local_path_to_tif_files=str(self.temp_dir / "input"),
            model_name="mock_model",
            data_dir=str(self.temp_dir),
            predict_dir=str(self.temp_dir),
            merge_predictions=True,
        )
        mock_load_from_checkpoint.assert_called()
        self.assertEqual(mock_make_prediction.call_count, 2)
        self.assertEqual(mock_gdal_merge.call_count, 2)
