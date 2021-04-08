from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
from src.ETL.label_downloader import RawLabels


class TestRawLabels(TestCase):
    @patch("src.ETL.label_downloader.urllib.request.urlretrieve")
    @patch("src.ETL.label_downloader.zipfile.ZipFile")
    def test_download_file(self, MockZipFile, mock_urlretrieve):
        mock_url = "https://some-url.zip"
        output_folder = Path("mock_path")
        output_path = output_folder / "some-url.zip"
        RawLabels(url=mock_url).download_file(output_folder, remove_zip=False)
        mock_urlretrieve.assert_called_once_with(mock_url, output_path)
        MockZipFile.assert_called_once_with(output_path, "r")
