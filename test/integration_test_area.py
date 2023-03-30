import os
from pathlib import Path
from unittest import TestCase
import numpy as np

from src.area_utils import (
    binarize,
    cal_map_area_class,
    compute_area_estimate,
    compute_confusion_matrix,
    estimate_num_sample_per_class,
    generate_ref_samples,
    load_ne,
    load_raster,
    reference_sample_agree,
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

def get_test_parameter():
    return {
        "country_iso3": "RWA",
        "regions_in_country": ["Kigali City","Northern"],
        "crop_user_acc": 0.8,
        "non_crop_user_acc": 0.7,
        "est_standard_error": 0.02,
    }

class TestAreaUtils(TestCase):
    def setUp(self):
        self.sample_input = get_test_parameter()
        self.roi = load_ne(self.sample_input["country_iso3"], self.sample_input["regions_in_country"])

    def test_area_path(self):
        self.assertTrue(map_path.exists(), f"{map_path} not found. Try dvc pull.")
        self.assertTrue(ceo_path_1.exists(), f"{ceo_path_1} not found. Try dvc pull.")
        self.assertTrue(ceo_path_2.exists(), f"{ceo_path_2} not found. Try dvc pull.")

    def test_region_of_interests(self):
        self.assertEqual(self.roi.shape, (1, 121), f"region of interest shape is {self.roi.shape}")

    def test_read_map_with_map_roi(self):
        map_array, map_meta = load_raster(map_path, self.roi)
        binary_map = binarize(map_array, map_meta)
        self.assertEqual(binary_map.dtype, "uint8", f"map dtype is {binary_map.dtype}")
        self.assertEqual(np.unique(binary_map).shape, (3,), f"map unique values are {np.unique(binary_map)}")

    def test_area_util(self):
        map_array, map_meta = load_raster(map_path)
        binary_map = binarize(map_array, map_meta)
        self.assertEqual(binary_map.dtype, "uint8", f"map dtype is {binary_map.dtype}")
        self.assertEqual(np.unique(binary_map).shape, (2,), f"map unique values are {np.unique(binary_map)}")

        crop_fraction, non_crop_fraction = cal_map_area_class(binary_map, unit="fraction")
        crop_pixel, non_crop_pixel = cal_map_area_class(binary_map)
        self.assertEqual(crop_pixel, 2642505, f"crop pixel is {crop_pixel}")
        self.assertEqual(non_crop_pixel, 57485511, f"non-crop pixel is {non_crop_pixel}")

        crop_num_sample, non_crop_num_sample = estimate_num_sample_per_class(
            crop_fraction,
            non_crop_fraction,
            self.sample_input["crop_user_acc"],
            self.sample_input["non_crop_user_acc"],
            self.sample_input["est_standard_error"],
        )
        self.assertEqual(crop_num_sample, 176, f"crop sample number is {crop_num_sample}")
        self.assertEqual(non_crop_num_sample, 176, f"non-crop sample number is {non_crop_num_sample}")

        generate_ref_samples(
            binary_map,
            map_meta,
            crop_num_sample,
            non_crop_num_sample
            )

        self.assertTrue(Path(Path.cwd() / "ceo_reference_sample.shp").exists(), "reference sample file not generated")

        ceo_geom = reference_sample_agree(binary_map, 
        map_meta, ceo_path_1, ceo_path_2)
        self.assertEqual(ceo_geom.shape, (63,15), f"ceo_geom shape is {ceo_geom.shape}")

        cm = compute_confusion_matrix(ceo_geom)
        self.assertEqual(cm[0], 28, f"cm[0] is {cm[0]}")
        self.assertEqual(cm[1], 1, f"cm[1] is {cm[1]}")

        summary = compute_area_estimate(crop_pixel, non_crop_pixel, cm, map_meta)
        self.assertAlmostEqual(summary.loc["Estimated area [ha]"][0], 322690.607527, places=2)
        self.assertAlmostEqual(summary.loc["Estimated area [ha]"][1], 275778.899716, places=2)

        for file in os.listdir(Path.cwd()):
            if file.startswith("ceo_reference_sample"):
                os.remove(file)

        print("\u2714 clean up")
