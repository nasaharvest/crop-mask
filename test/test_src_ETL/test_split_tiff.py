from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import shutil
import tempfile

from src.ETL.split_tiff import run_split_tiff


class SplitTiffTest(TestCase):

    temp_data_dir: Path

    @classmethod
    def setUpClass(cls):
        cls.temp_data_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_data_dir)

    @patch("src.ETL.split_tiff.splitImageIntoCells")
    def test_run_split_tiff(self, mock_splitImageIntoCells):
        file_names = [
            (
                "Mali_USAID_ZOIS_lower_2020-04-01_2021-04-01-0000023040-0000038400.tif",
                "1-Mali_USAID_ZOIS_lower_-0000023040-0000038400_2020-04-01_2021-04-01"
            ),
            (
                "Rwanda_2020-04-01_2021-04-01-0000023040-0000038400.tif",
                "0-Rwanda_-0000023040-0000038400_2020-04-01_2021-04-01"
            )
        ]
        for input_image, _ in file_names:
            (self.temp_data_dir / input_image).touch()

        run_split_tiff(self.temp_data_dir)

        self.assertEqual(mock_splitImageIntoCells.call_count, 2)

        mock_splitImageIntoCells.assert_called_with(
            self.temp_data_dir / file_names[0][0],
            file_names[0][1],
            1000,
            self.temp_data_dir
        )
