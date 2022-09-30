from unittest import TestCase
from pathlib import Path

from src.area_utils import (
    load_ne,
    load_raster,
    binarize,
    cal_map_area_class,
    estimate_num_sample_per_class,
    generate_ref_samples,
    reference_sample_agree,
    compute_confusion_matrix,
    compute_area_estimate,
)

home_dir = Path(__file__).parent.parent

map_path = home_dir / "data/test_area/test_map.tif"

ceo_path_1 = (
    home_dir
    / "data/test_area/ceo-2019-Rwanda-Cropland-(RCMRD-Set-1)-sample-data-2022-08-29_subset.csv"
)

ceo_path_2 = (
    home_dir
    / "data/test_area/ceo-2019-Rwanda-Cropland-(RCMRD-Set-2)-sample-data-2022-08-29_subset.csv"
)


class IntegrationTestArea(TestCase):

    sample_input = {
        "country_iso3": "RWA",
        "regions_in_country": ["Kigali City", "Northern"],
        "crop_user_acc": 0.8,
        "non_crop_user_acc": 0.7,
        "est_standard_error": 0.02,
    }

    def test_area_utils(self):
        self.assertTrue(map_path.exists())
        self.assertTrue(ceo_path_1.exists())
        self.assertTrue(ceo_path_2.exists())

        roi = load_ne(self.sample_input["country_iso3"], self.sample_input["regions_in_country"])
        self.assertEqual(roi.shape, (2, 121))

        map_array, map_meta = load_raster(map_path, roi)
        self.assertEqual(map_array.shape, (6513, 9232))
        self.assertEqual(map_meta["crs"], "EPSG:32735")

        binary_map = binarize(map_array, map_meta)
        self.assertEqual(binary_map.dtype, "uint8")

        crop_fraction, non_crop_fraction = cal_map_area_class(binary_map, unit="fraction")
        crop_pixel, non_crop_pixel = cal_map_area_class(binary_map)
        self.assertAlmostEqual(crop_fraction, 0.4825700)
        self.assertAlmostEqual(non_crop_fraction, 0.5174299946966485)

        crop_num_sample, non_crop_num_sample = estimate_num_sample_per_class(
            crop_fraction,
            non_crop_fraction,
            self.sample_input["crop_user_acc"],
            self.sample_input["non_crop_user_acc"],
            self.sample_input["est_standard_error"],
        )
        self.assertEqual(crop_num_sample, 187)
        self.assertEqual(non_crop_num_sample, 187)

        generate_ref_samples(binary_map, map_meta, crop_num_sample, non_crop_num_sample)
        self.assertTrue(Path(Path.cwd() / "ceo_reference_sample.shp").exists())

        ceo_geom = reference_sample_agree(binary_map, map_meta, ceo_path_1, ceo_path_2)
        self.assertEqual(ceo_geom.shape, (63, 15))

        cm = compute_confusion_matrix(ceo_geom)
        self.assertEqual(cm[0], 20)
        self.assertEqual(cm[1], 9)

        summary = compute_area_estimate(crop_pixel, non_crop_pixel, cm, map_meta)
        self.assertEqual(summary.shape, (8, 2))
        self.assertAlmostEqual(summary.loc["Estimated area [ha]"][0], 331112.02102444763)
        self.assertAlmostEqual(summary.loc["Estimated area [ha]"][1], 267357.48621850886)

        # TODO delete files after test
