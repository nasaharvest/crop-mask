from unittest import TestCase
from unittest.mock import patch
from scripts.export import export_from_labeled, export_from_bbox, STR2BB
from src.boundingbox import BoundingBox


class TestExport(TestCase):
    """Tests the export script"""

    def test_verify_STR2BB(self):
        for region_key, region_bbox in STR2BB.items():
            self.assertIsInstance(region_key, str)
            self.assertIsInstance(region_bbox, BoundingBox)

    @patch("scripts.export.GeoWikiExporter")
    @patch("scripts.export.GeoWikiSentinelExporter")
    @patch("scripts.export.KenyaPVSentinelExporter")
    @patch("scripts.export.KenyaNonCropSentinelExporter")
    @patch("scripts.export.KenyaOAFSentinelExporter")
    def test_export_from_labeled(self, *mock_exporters):
        export_from_labeled()
        for mock_exporter in mock_exporters:
            mock_exporter.assert_called()

    @patch("scripts.export.RegionalExporter")
    def test_export_from_bbox(self, mock_regional_exporter):
        export_from_bbox("Busia")
        mock_regional_exporter.assert_called()
        self.assertRaises(ValueError, export_from_bbox, "garbage")
