from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
from datetime import date
from typing import Dict
import pandas as pd
import tempfile
import shutil

from src.ETL.ee_exporter import (
    Season,
    BoundingBox,
    LabelExporter,
    RegionExporter,
    EarthEngineExporter,
)
from src.ETL.constants import DEST_TIF, LAT, LON, START, END

module = "src.ETL.ee_exporter"


class TestEEExporters(TestCase):

    temp_data_dir: Path
    exporters: Dict[str, EarthEngineExporter]

    @classmethod
    def setUpClass(cls):
        cls.temp_data_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_data_dir)

    @patch(f"{module}.ee")
    @patch(f"{module}.ee.Geometry.Polygon")
    @patch(f"{module}.pd.read_csv")
    @patch(f"{module}.EarthEngineExporter._export_for_polygon")
    def test_export_for_labels(self, mock_export_for_polygon, pd_read_csv, mock_Polygon, ee):
        pd_read_csv.return_value = pd.DataFrame(
            {
                LAT: [1],
                LON: [1],
                END: ["2020-04-16"],
                START: ["2019-04-22"],
                DEST_TIF: ["tmp/0_2019-04-22_2020-04-16.tif"],
            }
        )
        mock_poly_return = "mock_poly"
        mock_Polygon.return_value = mock_poly_return
        ee_dataset = "mock_dataset_name"
        LabelExporter(sentinel_dataset=ee_dataset).export(
            labels_path=Path("a/fake/path"),
            num_labelled_points=None,
            output_folder=self.temp_data_dir / "raw",
        )

        mock_export_for_polygon.assert_called_with(
            file_name_prefix="mock_dataset_name/tmp/0_2019-04-22_2020-04-16.tif",
            polygon=mock_poly_return,
            start_date=date(2019, 4, 22),
            end_date=date(2020, 4, 16),
        )
        mock_export_for_polygon.reset_mock()

    @patch(f"{module}.get_user_input")
    @patch(f"{module}.date")
    def test_start_end_dates_using_in_season(self, mock_date, get_user_input):
        mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

        mock_date.today.return_value = date(2021, 1, 23)

        start, end = RegionExporter._start_end_dates_using_season(Season.in_season)
        self.assertEqual(start, date(2020, 4, 1))
        self.assertEqual(end, date(2021, 1, 23))

        mock_date.today.return_value = date(2021, 4, 1)
        start, end = RegionExporter._start_end_dates_using_season(Season.in_season)
        self.assertEqual(start, date(2020, 4, 1))
        self.assertEqual(end, date(2021, 4, 1))

        # 1 month span should produce a warning and proceed if user input is yes
        mock_date.today.return_value = date(2021, 5, 1)
        get_user_input.return_value = "yes"
        start, end = RegionExporter._start_end_dates_using_season(Season.in_season)
        self.assertEqual(start, date(2021, 4, 1))
        self.assertEqual(end, date(2021, 5, 1))

        # 1 month span should produce a warning and exit if user input is no
        get_user_input.return_value = "no"
        self.assertRaises(
            SystemExit, RegionExporter._start_end_dates_using_season, Season.in_season
        )

        # 6 month span should produce a warning and exit if user input is no
        mock_date.today.return_value = date(2021, 10, 1)
        get_user_input.return_value = "no"
        self.assertRaises(
            SystemExit, RegionExporter._start_end_dates_using_season, Season.in_season
        )
        self.assertEqual(
            get_user_input.call_count, 3, "get_user_input should have been called thrice."
        )

    @patch(f"{module}.date")
    def test_start_end_dates_using_post_season(self, mock_date):
        mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

        mock_date.today.return_value = date(2021, 1, 23)
        start, end = RegionExporter._start_end_dates_using_season(Season.post_season)
        self.assertEqual(start, date(2019, 4, 1))
        self.assertEqual(end, date(2020, 4, 1))

        mock_date.today.return_value = date(2021, 4, 1)
        start, end = RegionExporter._start_end_dates_using_season(Season.post_season)
        self.assertEqual(start, date(2019, 4, 1))
        self.assertEqual(end, date(2020, 4, 1))

        mock_date.today.return_value = date(2021, 5, 1)
        start, end = RegionExporter._start_end_dates_using_season(Season.post_season)
        self.assertEqual(start, date(2020, 4, 1))
        self.assertEqual(end, date(2021, 4, 1))

    @patch(f"{module}.ee")
    @patch(f"{module}.ee.Geometry.Polygon")
    @patch("src.ETL.cloudfree.fast.ee")
    @patch("src.ETL.cloudfree.utils.ee.batch.Export.image.toDrive")
    def test_export_for_region_metres_per_polygon_none(
        self, mock_export_image, mock_cloudfree_ee, mock_ee_polygon, mock_base_ee
    ):
        RegionExporter(sentinel_dataset="Togo").export(
            season=Season.post_season,
            metres_per_polygon=None,
        )
        mock_base_ee.Initialize.assert_called()

        expected_polygon_count = 1
        expected_image_colls_to_export = 12 * expected_polygon_count
        self.assertEqual(mock_ee_polygon.call_count, expected_polygon_count)
        self.assertEqual(mock_cloudfree_ee.DateRange.call_count, expected_image_colls_to_export * 3)
        self.assertEqual(
            mock_cloudfree_ee.ImageCollection.call_count, expected_image_colls_to_export
        )
        self.assertEqual(mock_export_image.call_count, expected_polygon_count)

    @patch(f"{module}.ee")
    @patch(f"{module}.ee.Geometry.Polygon")
    @patch("src.ETL.cloudfree.fast.ee")
    @patch("src.ETL.cloudfree.utils.ee.batch.Export.image.toDrive")
    def test_export_for_region_metres_per_polygon_set(
        self, mock_export_image, mock_cloudfree_ee, mock_ee_polygon, mock_base_ee
    ):
        RegionExporter(sentinel_dataset="Togo").export(
            season=Season.post_season,
            metres_per_polygon=10000,
        )
        mock_base_ee.Initialize.assert_called()
        expected_polygon_count = 1155
        expected_image_colls_to_export = 12 * expected_polygon_count
        self.assertEqual(mock_ee_polygon.call_count, expected_polygon_count)
        self.assertEqual(mock_cloudfree_ee.DateRange.call_count, expected_image_colls_to_export * 3)
        self.assertEqual(
            mock_cloudfree_ee.ImageCollection.call_count, expected_image_colls_to_export
        )
        self.assertEqual(mock_export_image.call_count, expected_polygon_count)
