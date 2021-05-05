from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import shutil
import tempfile

from src.ETL.split_tiff import run_split_tiff


class SplitTiffTest(TestCase):

    temp_data_dir: Path

    @classmethod
    def setUp(cls):
        cls.temp_data_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDown(cls):
        shutil.rmtree(cls.temp_data_dir)

    @patch("src.ETL.split_tiff.splitImageIntoCells")
    def test_run_split_tiff_1(self, mock_splitImageIntoCells):
        input_file = "Mali_USAID_ZOIS_lower_2020-04-01_2021-04-01-0000023040-0000038400.tif"
        output_file = "0-Mali_USAID_ZOIS_lower_-0000023040-0000038400_2020-04-01_2021-04-01"

        (self.temp_data_dir / input_file).touch()

        run_split_tiff(self.temp_data_dir)

        mock_splitImageIntoCells.assert_called_with(
            self.temp_data_dir / input_file, output_file, 1000, self.temp_data_dir
        )

    @patch("src.ETL.split_tiff.splitImageIntoCells")
    def test_run_split_tiff_2(self, mock_splitImageIntoCells):
        input_file = "Rwanda_2020-04-01_2021-04-01-0000023040-0000038400.tif"
        output_file = "0-Rwanda_-0000023040-0000038400_2020-04-01_2021-04-01"

        (self.temp_data_dir / input_file).touch()

        run_split_tiff(self.temp_data_dir)

        mock_splitImageIntoCells.assert_called_with(
            self.temp_data_dir / input_file, output_file, 1000, self.temp_data_dir
        )

