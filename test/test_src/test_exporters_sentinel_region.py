from unittest import TestCase
from unittest.mock import patch
from datetime import date
from src.exporters.sentinel.region import RegionalExporter, Season
from src.data_classes import BoundingBox
from pathlib import Path
import tempfile
import shutil


class TestRegionalExporter(TestCase):
    """Tests for the RegionalExporter"""

    temp_dir: Path = Path("")

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    @patch("src.exporters.sentinel.base.ee.Initialize")
    def test_init(self, mock_ee_initialize):
        exporter = RegionalExporter(self.temp_dir)
        mock_ee_initialize.assert_called()
        self.assertTrue(exporter.labels.empty)

    @patch("src.exporters.sentinel.region.get_user_input")
    @patch("src.exporters.sentinel.region.date")
    def test_determine_start_end_dates_in_season(self, mock_date, get_user_input):
        mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

        mock_date.today.return_value = date(2021, 1, 23)

        start, end = RegionalExporter.determine_start_end_dates(Season.in_season)
        self.assertEqual(start, date(2020, 4, 1))
        self.assertEqual(end, date(2021, 1, 23))

        mock_date.today.return_value = date(2021, 4, 1)
        start, end = RegionalExporter.determine_start_end_dates(Season.in_season)
        self.assertEqual(start, date(2020, 4, 1))
        self.assertEqual(end, date(2021, 4, 1))

        # 1 month span should produce a warning and proceed if user input is yes
        mock_date.today.return_value = date(2021, 5, 1)
        get_user_input.return_value = "yes"
        start, end = RegionalExporter.determine_start_end_dates(Season.in_season)
        self.assertEqual(start, date(2021, 4, 1))
        self.assertEqual(end, date(2021, 5, 1))

        # 1 month span should produce a warning and exit if user input is no
        get_user_input.return_value = "no"
        self.assertRaises(SystemExit, RegionalExporter.determine_start_end_dates, Season.in_season)

        # 6 month span should produce a warning and exit if user input is no
        mock_date.today.return_value = date(2021, 10, 1)
        get_user_input.return_value = "no"
        self.assertRaises(SystemExit, RegionalExporter.determine_start_end_dates, Season.in_season)
        self.assertEqual(
            get_user_input.call_count, 3, "get_user_input should have been called thrice."
        )

    @patch("src.exporters.sentinel.region.date")
    def test_determine_start_end_dates_post_season(self, mock_date):
        mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

        mock_date.today.return_value = date(2021, 1, 23)
        start, end = RegionalExporter.determine_start_end_dates(Season.post_season)
        self.assertEqual(start, date(2019, 4, 1))
        self.assertEqual(end, date(2020, 4, 1))

        mock_date.today.return_value = date(2021, 4, 1)
        start, end = RegionalExporter.determine_start_end_dates(Season.post_season)
        self.assertEqual(start, date(2019, 4, 1))
        self.assertEqual(end, date(2020, 4, 1))

        mock_date.today.return_value = date(2021, 5, 1)
        start, end = RegionalExporter.determine_start_end_dates(Season.post_season)
        self.assertEqual(start, date(2020, 4, 1))
        self.assertEqual(end, date(2021, 4, 1))

    @patch("src.exporters.sentinel.base.ee")
    @patch("src.exporters.sentinel.utils.ee.Geometry.Polygon")
    @patch("src.exporters.sentinel.cloudfree.fast.ee")
    @patch("src.exporters.sentinel.cloudfree.utils.ee.batch.Export.image")
    def test_export_for_region_metres_per_polygon_none(
        self, mock_export_image, mock_cloudfree_ee, mock_ee_polygon, mock_base_ee
    ):
        func_to_test = RegionalExporter(self.temp_dir).export_for_region
        mock_base_ee.Initialize.assert_called()
        region_name = "test_region_name"
        region_bbox = BoundingBox(0, 1, 0, 1)
        metres_per_polygon = None

        self.assertRaises(
            ValueError, func_to_test, region_name=region_name, region_bbox=region_bbox
        )

        func_to_test(
            region_name=region_name,
            region_bbox=region_bbox,
            season=Season.post_season,
            metres_per_polygon=metres_per_polygon,
        )

        expected_polygon_count = 1
        expected_image_colls_to_export = 12 * expected_polygon_count
        self.assertEqual(mock_ee_polygon.call_count, expected_polygon_count)
        self.assertEqual(mock_cloudfree_ee.DateRange.call_count, expected_image_colls_to_export * 3)
        self.assertEqual(
            mock_cloudfree_ee.ImageCollection.call_count, expected_image_colls_to_export
        )
        self.assertEqual(mock_export_image.call_count, expected_polygon_count)

    @patch("src.exporters.sentinel.base.ee")
    @patch("src.exporters.sentinel.utils.ee.Geometry.Polygon")
    @patch("src.exporters.sentinel.cloudfree.fast.ee")
    @patch("src.exporters.sentinel.cloudfree.utils.ee.batch.Export.image")
    def test_export_for_region_metres_per_polygon_set(
        self, mock_export_image, mock_cloudfree_ee, mock_ee_polygon, mock_base_ee
    ):
        func_to_test = RegionalExporter(self.temp_dir).export_for_region
        mock_base_ee.Initialize.assert_called()
        region_name = "test_region_name"
        region_bbox = BoundingBox(0, 1, 0, 1)
        metres_per_polygon = 10000

        self.assertRaises(
            ValueError, func_to_test, region_name=region_name, region_bbox=region_bbox
        )

        func_to_test(
            region_name=region_name,
            region_bbox=region_bbox,
            season=Season.post_season,
            metres_per_polygon=metres_per_polygon,
        )

        expected_polygon_count = 121
        expected_image_colls_to_export = 12 * expected_polygon_count
        self.assertEqual(mock_ee_polygon.call_count, expected_polygon_count)
        self.assertEqual(mock_cloudfree_ee.DateRange.call_count, expected_image_colls_to_export * 3)
        self.assertEqual(
            mock_cloudfree_ee.ImageCollection.call_count, expected_image_colls_to_export
        )
        self.assertEqual(mock_export_image.call_count, expected_polygon_count)
