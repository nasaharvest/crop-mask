from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
from datetime import timedelta, date
import pandas as pd
import tempfile
import shutil
import xarray as xr

from src.ETL.ee_exporter import EarthEngineExporter, Season, BoundingBox, EEBoundingBox

module = "src.ETL.ee_exporter"


class TestEEExporters(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_data_dir = Path(tempfile.mkdtemp())
        cls.exporters = {
            "earth_engine_kenya_non_crop": EarthEngineExporter(end_date=date(2020, 4, 16)),
            "earth_engine_one_acre_fund_kenya": EarthEngineExporter(
                end_date=date(2020, 4, 16),
            ),
            "earth_engine_geowiki": EarthEngineExporter(
                start_date=date(2017, 3, 28),
                end_date=date(2018, 3, 28),
            ),
            "earth_engine_plant_village_kenya": EarthEngineExporter(
                additional_cols=["index", "planting_d", "harvest_da"],
                end_month_day=(4, 16),
            ),
        }

        dataset_obj = {
            "lat": [1],
            "lon": [1],
            "index": [0],
            "planting_d": ["2020-01-01 00:00:00"],
            "harvest_da": ["2020-02-01 00:00:00"],
        }
        cls.labels_df = pd.DataFrame(dataset_obj)
        cls.labels_xr = xr.Dataset(dataset_obj)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_data_dir)

    def test_end_date_using_max_overlap(self):
        kwargs = {
            "planting_date": date(2019, 11, 1),
            "harvest_date": date(2020, 3, 1),
            "end_month_day": (4, 1),
            "total_days": timedelta(30 * 12),
        }
        end_date = EarthEngineExporter._end_date_using_max_overlap(**kwargs)
        self.assertEqual(2020, end_date.year)

        # Test when range misses end date by a bit
        kwargs["planting_date"] = date(2019, 11, 1)
        kwargs["harvest_date"] = date(2020, 5, 1)
        end_date = EarthEngineExporter._end_date_using_max_overlap(**kwargs)
        self.assertEqual(2020, end_date.year)

        # Test when range misses end date by a lot
        kwargs["planting_date"] = date(2019, 11, 1)
        kwargs["harvest_date"] = date(2020, 11, 1)
        end_date = EarthEngineExporter._end_date_using_max_overlap(**kwargs)
        self.assertEqual(2021, end_date.year)

    def test_labels_to_bounding_boxes(self):
        for exporter in self.exporters.values():
            output = exporter._labels_to_bounding_boxes(
                labels=self.labels_df, num_labelled_points=None, surrounding_metres=80
            )

            self.assertEqual(1, len(output))
            label = output[0]
            self.assertEqual(0, label[0], "index is not equal")
            self.assertEqual(EEBoundingBox, type(label[1]), "Should be type EEBoundingBox")

            if exporter.end_date:
                expected_end_date = exporter.end_date
            elif exporter.end_month_day:
                expected_end_date = date(2020, *exporter.end_month_day)
            else:
                raise ValueError("Expected end date could not be set.")

            if exporter.start_date:
                expected_start_date = exporter.start_date
            else:
                expected_start_date = expected_end_date - timedelta(360)

            self.assertEqual(expected_end_date, label[3], "end date is not equal")
            self.assertEqual(expected_start_date, label[2], "start date is not equal")

    @patch(f"{module}.geopandas.read_file")
    @patch("src.ETL.ee_exporter.xr.open_dataset")
    def test_load_labels(self, mock_xr_open_dataset, mock_geopandas_read_file):
        mock_geopandas_read_file.return_value = self.labels_df
        mock_xr_open_dataset.return_value = self.labels_xr
        for i, exporter in enumerate(self.exporters.values()):
            labels_path = (
                self.temp_data_dir / "processed" / ("data.nc" if i == 2 else "data.geojson")
            )
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            with labels_path.open("w", encoding="utf-8") as f:
                f.write("")
            actual_df = exporter._load_labels(labels_path)
            cols = ["lat", "lon"] + exporter.additional_cols
            self.assertTrue(actual_df[cols].equals(self.labels_df[cols]))

    @patch(f"{module}.ee")
    @patch(f"{module}.ee.Geometry.Polygon")
    @patch(f"{module}.EarthEngineExporter._load_labels")
    @patch(f"{module}.EarthEngineExporter._export_for_polygon")
    def test_export_for_labels(
        self, mock_export_for_polygon, mock_load_labels, mock_Polygon, mock_ee
    ):
        mock_load_labels.return_value = self.labels_df
        mock_poly_return = "mock_poly"
        mock_Polygon.return_value = mock_poly_return
        for ee_dataset, exporter in self.exporters.items():
            output_folder = self.temp_data_dir / "raw" / ee_dataset
            exporter.export_for_labels(
                labels_path=Path("a/fake/path"),
                sentinel_dataset=ee_dataset,
                output_folder=output_folder,
                num_labelled_points=None,
                monitor=False,
                checkpoint=True,
            )
            if ee_dataset == "earth_engine_geowiki":
                start_date = date(2017, 3, 28)
                end_date = date(2018, 3, 28)
            else:
                start_date = date(2019, 4, 22)
                end_date = date(2020, 4, 16)

            mock_export_for_polygon.assert_called_with(
                output_folder=output_folder,
                sentinel_dataset=ee_dataset,
                polygon=mock_poly_return,
                polygon_identifier=0,
                start_date=start_date,
                end_date=end_date,
                checkpoint=True,
                monitor=False,
                fast=True,
            )
            mock_export_for_polygon.reset_mock()

    @patch(f"{module}.get_user_input")
    @patch(f"{module}.date")
    def test_start_end_dates_using_in_season(self, mock_date, get_user_input):
        mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

        mock_date.today.return_value = date(2021, 1, 23)

        start, end = EarthEngineExporter._start_end_dates_using_season(Season.in_season)
        self.assertEqual(start, date(2020, 4, 1))
        self.assertEqual(end, date(2021, 1, 23))

        mock_date.today.return_value = date(2021, 4, 1)
        start, end = EarthEngineExporter._start_end_dates_using_season(Season.in_season)
        self.assertEqual(start, date(2020, 4, 1))
        self.assertEqual(end, date(2021, 4, 1))

        # 1 month span should produce a warning and proceed if user input is yes
        mock_date.today.return_value = date(2021, 5, 1)
        get_user_input.return_value = "yes"
        start, end = EarthEngineExporter._start_end_dates_using_season(Season.in_season)
        self.assertEqual(start, date(2021, 4, 1))
        self.assertEqual(end, date(2021, 5, 1))

        # 1 month span should produce a warning and exit if user input is no
        get_user_input.return_value = "no"
        self.assertRaises(
            SystemExit, EarthEngineExporter._start_end_dates_using_season, Season.in_season
        )

        # 6 month span should produce a warning and exit if user input is no
        mock_date.today.return_value = date(2021, 10, 1)
        get_user_input.return_value = "no"
        self.assertRaises(
            SystemExit, EarthEngineExporter._start_end_dates_using_season, Season.in_season
        )
        self.assertEqual(
            get_user_input.call_count, 3, "get_user_input should have been called thrice."
        )

    @patch(f"{module}.date")
    def test_start_end_dates_using_post_season(self, mock_date):
        mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

        mock_date.today.return_value = date(2021, 1, 23)
        start, end = EarthEngineExporter._start_end_dates_using_season(Season.post_season)
        self.assertEqual(start, date(2019, 4, 1))
        self.assertEqual(end, date(2020, 4, 1))

        mock_date.today.return_value = date(2021, 4, 1)
        start, end = EarthEngineExporter._start_end_dates_using_season(Season.post_season)
        self.assertEqual(start, date(2019, 4, 1))
        self.assertEqual(end, date(2020, 4, 1))

        mock_date.today.return_value = date(2021, 5, 1)
        start, end = EarthEngineExporter._start_end_dates_using_season(Season.post_season)
        self.assertEqual(start, date(2020, 4, 1))
        self.assertEqual(end, date(2021, 4, 1))

    @patch(f"{module}.ee")
    @patch(f"{module}.ee.Geometry.Polygon")
    @patch("src.ETL.cloudfree.fast.ee")
    @patch("src.ETL.cloudfree.utils.ee.batch.Export.image")
    def test_export_for_region_metres_per_polygon_none(
        self, mock_export_image, mock_cloudfree_ee, mock_ee_polygon, mock_base_ee
    ):
        region_name = "test_region_name"
        region_bbox = BoundingBox(0, 1, 0, 1)
        func_to_test = EarthEngineExporter(
            region_bbox=region_bbox, season=Season.post_season
        ).export_for_region
        metres_per_polygon = None

        self.assertRaises(
            ValueError,
            EarthEngineExporter,
            region_bbox=region_bbox,
        )

        func_to_test(
            sentinel_dataset=region_name,
            output_folder=self.temp_data_dir / "raw" / region_name,
            metres_per_polygon=metres_per_polygon,
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
    @patch("src.ETL.cloudfree.utils.ee.batch.Export.image")
    def test_export_for_region_metres_per_polygon_set(
        self, mock_export_image, mock_cloudfree_ee, mock_ee_polygon, mock_base_ee
    ):
        region_name = "test_region_name"
        region_bbox = BoundingBox(0, 1, 0, 1)
        func_to_test = EarthEngineExporter(
            season=Season.post_season, region_bbox=region_bbox
        ).export_for_region

        metres_per_polygon = 10000

        self.assertRaises(ValueError, EarthEngineExporter, region_bbox=region_bbox)

        func_to_test(
            sentinel_dataset=region_name,
            output_folder=self.temp_data_dir / "raw" / region_name,
            metres_per_polygon=metres_per_polygon,
        )
        mock_base_ee.Initialize.assert_called()
        expected_polygon_count = 121
        expected_image_colls_to_export = 12 * expected_polygon_count
        self.assertEqual(mock_ee_polygon.call_count, expected_polygon_count)
        self.assertEqual(mock_cloudfree_ee.DateRange.call_count, expected_image_colls_to_export * 3)
        self.assertEqual(
            mock_cloudfree_ee.ImageCollection.call_count, expected_image_colls_to_export
        )
        self.assertEqual(mock_export_image.call_count, expected_polygon_count)
