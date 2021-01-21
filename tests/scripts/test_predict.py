from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
from scripts.predict import make_prediction
import xarray as xr

class TestPredict(TestCase):

    @patch('src.models.Model')
    @patch('scripts.predict.plot_results')
    def test_make_prediction(self, mock_model, mock_plot_results):
        # Setup
        mock_model.predict.return_value = xr.Dataset(None)
        mock_plot_results.return_value = None
        save_dir = Path('../data/predictions')

        tif_file_name = 'empty.tif'
        test_path = Path(f'../data/tif_files/{tif_file_name}')
        expected_forecasted_path = save_dir / f'preds_forecasted_{tif_file_name}.nc'
        expected_normal_path = save_dir / f'preds_normal_{tif_file_name}.nc'

        for p in [expected_forecasted_path, expected_normal_path]:
            if p.exists():
                p.unlink()

        # Test making forecasted prediction
        output_file_path = make_prediction(model=mock_model,
                                      test_path=test_path,
                                      save_dir=save_dir,
                                      with_forecaster=True)
        self.assertEqual(output_file_path, expected_forecasted_path)
        self.assertTrue(output_file_path.exists())

        # Test that prediction is not made if file already exists
        output_file_path = make_prediction(model=mock_model,
                                           test_path=test_path,
                                           save_dir=save_dir,
                                           with_forecaster=True)
        self.assertEqual(output_file_path, None)

        # Test making normal prediction
        output_file_path = make_prediction(model=mock_model,
                                           test_path=test_path,
                                           save_dir=save_dir,
                                           with_forecaster=False)
        self.assertEqual(output_file_path, expected_normal_path)
        self.assertTrue(output_file_path.exists())


    def test_gdal_merge(self):
        print('hello')
