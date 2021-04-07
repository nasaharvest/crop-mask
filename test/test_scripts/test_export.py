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

    @patch("src.ETL.dataset.Dataset.download_raw_labels")
    @patch("scripts.export.GeoWikiSentinelExporter")
    @patch("scripts.export.KenyaPVSentinelExporter")
    @patch("scripts.export.KenyaNonCropSentinelExporter")
    @patch("scripts.export.KenyaOAFSentinelExporter")
    def test_export_from_labeled(
        self,
        KOAFExporter,
        KNonCropExporter,
        KPVExporter,
        GeoWikiSentinelExporter,
        mock_download_raw_labels,
    ):
        export_from_labeled()
        for mock_exporter in [KOAFExporter, KNonCropExporter, KPVExporter, GeoWikiSentinelExporter]:
            mock_exporter.assert_called()
        mock_download_raw_labels.assert_called()

    @patch("scripts.export.RegionalExporter")
    def test_export_from_bbox(self, mock_regional_exporter):
        export_from_bbox("Busia")
        mock_regional_exporter.assert_called()
        self.assertRaises(ValueError, export_from_bbox, "garbage")
