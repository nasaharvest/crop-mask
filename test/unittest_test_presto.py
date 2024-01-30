import os
import sys
import unittest

import torch
from einops import repeat

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.single_file_presto_v2 import BANDS_GROUPS_IDX, Presto


class PrestoTest(unittest.TestCase):
    def test_band_groups_mean(self):
        # hidden size = 1
        num_timesteps = 12
        x = torch.arange(-1, len(BANDS_GROUPS_IDX)).unsqueeze(-1).float()
        x = torch.stack((x, x))
        cur_index, kept_indices = 0, []
        for band, _ in BANDS_GROUPS_IDX.items():
            kept_indices.append(cur_index)
            if band == "SRTM":
                cur_index += 1
            else:
                cur_index += num_timesteps
        kept_indices_t = torch.tensor(kept_indices)
        kept_indices_t = torch.stack((kept_indices_t, kept_indices_t))
        model = Presto.construct()
        out = model.encoder.band_groups_mean(x, kept_indices_t, num_timesteps)
        expected_out = torch.arange(0, len(BANDS_GROUPS_IDX))
        expected_out = torch.stack((expected_out, expected_out))
        self.assertTrue(torch.equal(expected_out, out))

    def test_band_groups_mean_d_128(self):
        num_timesteps = 12
        x = torch.tensor([-1, 0, 0, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8]).float()
        x = repeat(x, "t -> b t d", b=2, d=128)
        kept_indices = torch.tensor(
            [
                [0, 1, 12, 24, 25, 26, 36, 48, 60, 72, 84, 85],
                [0, 8, 12, 24, 25, 28, 36, 48, 60, 72, 84, 85],
            ]
        )
        model = Presto.construct()
        out = model.encoder.band_groups_mean(x, kept_indices, num_timesteps)
        expected_out = torch.arange(0, len(BANDS_GROUPS_IDX))
        expected_out = torch.repeat_interleave(expected_out, 128)
        expected_out = torch.stack((expected_out, expected_out))
        self.assertTrue(torch.equal(expected_out, out))
