import os
import sys
import unittest

import numpy as np

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.area_utils_refac import (
    compute_acc,
    compute_area_error_matrix,
    compute_p_i,
    compute_std_p_i,
    compute_u_j,
    compute_var_acc,
    compute_var_p_i,
    compute_var_u_j,
    compute_area_estimate
)


class ChangeAreaTest(unittest.TestCase):
    """Multi-year change area estimation testcase.

    Comparison with 'toy example' seen in Olofsson et al., 2014 "Good Practices...".

    """

    def setUp(self):
        self.cm = np.array([[66, 0, 1, 2], [0, 55, 0, 1], [5, 8, 153, 9], [4, 12, 11, 313]])

        self.am = np.array(
            [
                [0.0176, 0.0000, 0.0019, 0.0040],
                [0.0000, 0.0110, 0.0000, 0.0020],
                [0.0013, 0.0016, 0.2967, 0.0179],
                [0.0011, 0.0024, 0.0213, 0.6212],
            ]
        )

        self.a_j = np.array([200_000, 150_000, 3_200_000, 6_450_000], dtype=np.int64)
        self.w_j = self.a_j / self.a_j.sum()

        self.u_j = np.array([0.8800, 0.7333, 0.9274, 0.9631])
        self.err_u_j = np.array([0.0740, 0.1008, 0.0398, 0.0205])

        self.p_i = np.array([0.7487, 0.8472, 0.9345, 0.9616])
        self.err_p_i = np.array([0.2133, 0.2544, 0.0343, 0.0184])

        self.acc = 0.9464
        self.err_acc = 0.0185

    def test_area_error_matrix(self):
        am = compute_area_error_matrix(self.cm, self.w_j)
        np.testing.assert_array_almost_equal(
            x=am,
            y=self.am,
            decimal=3,
            err_msg="[Change Area] - Incorrect error matrix expressed as area proportion.",
            verbose=True,
        )

    def test_user_accuracy(self):
        u_j = compute_u_j(self.am)
        np.testing.assert_array_almost_equal(
            x=u_j,
            y=self.u_j,
            decimal=2,
            err_msg="[Change Area] - Incorrect user's accuracy.",
            verbose=True,
        )

    def test_variance_user_accuracy(self):
        var_u_j = compute_var_u_j(self.u_j, self.cm)
        np.testing.assert_array_almost_equal(
            x=1.96 * np.sqrt(var_u_j),
            y=self.err_u_j,
            decimal=4,
            err_msg="[Change Area] - Incorrect variance of user's accuracy.",
            verbose=True,
        )

    def test_producer_accuracy(self):
        p_i = compute_p_i(self.am)
        np.testing.assert_array_almost_equal(
            x=p_i,
            y=self.p_i,
            decimal=2,
            err_msg="[Change Area] - Incorrect producer's accuracy.",
            verbose=True,
        )

    def test_variance_producer_accuracy(self):
        var_p_i = compute_var_p_i(self.p_i, self.u_j, self.a_j, self.cm)
        np.testing.assert_array_almost_equal(
            x=1.96 * np.sqrt(var_p_i),
            y=self.err_p_i,
            decimal=4,
            err_msg="[Change Area] - Incorrect variance of producer's accuracy.",
            verbose=True,
        )

    def test_accuracy(self):
        acc = compute_acc(self.am)
        np.testing.assert_almost_equal(
            actual=acc,
            desired=self.acc,
            decimal=3,
            err_msg="[Change Area] - Incorrect accuracy.",
            verbose=True,
        )

    def test_variance_of_accuracy(self):
        var_acc = compute_var_acc(self.w_j, self.u_j, self.cm)
        np.testing.assert_almost_equal(
            actual=1.96 * np.sqrt(var_acc),
            desired=self.err_acc,
            decimal=3,
            err_msg="[Change Area] - Incorrect variance of accuracy.",
            verbose=True,
        )

    def test_area_estimation(self):
        am = compute_area_error_matrix(self.cm, self.w_j)
        a_i = am.sum(axis=1)
        a_px = a_i * self.a_j.sum()
        a_ha = a_px * 0.09
        np.testing.assert_almost_equal(
            actual=a_ha,
            desired=np.array([21_158, 11_686, 285_770, 581_386]),
            decimal=0,
            err_msg="[Change Area] - Incorrect area estimation.",
            verbose=True,
        )

    def test_stddev_of_area_estimate(self):
        std_pi = compute_std_p_i(self.w_j, self.am, self.cm)
        np.testing.assert_almost_equal(
            actual=std_pi,
            desired=np.array([0.0035, 0.0021, 0.0088, 0.0092]),
            decimal=4,
            err_msg="[Change Area] - Incorrect standard deviation of area estimation.",
            verbose=True,
        )

    def test_compute_area_estimate(self):
        estimates = compute_area_estimate(self.cm, self.a_j, px_size=30)
        u_j, err_u_j = estimates["user"]
        p_i, err_p_i = estimates["producer"]
        acc, err_acc = estimates["accuracy"]
        a_ha, err_ha = estimates["area"]["ha"]

        # users
        np.testing.assert_almost_equal(
            actual=np.stack([u_j, err_u_j]),
            desired=np.stack([self.u_j, self.err_u_j]),
            decimal=4,
            err_msg="[Change Area] - ",
            verbose=True
        )

        # producers
        np.testing.assert_almost_equal(
            actual=np.stack([p_i, err_p_i]),
            desired=np.stack([self.p_i, self.err_p_i]),
            decimal=4,
            err_msg="[Change Area] - ",
            verbose=True
        )

        # accuracy
        np.testing.assert_almost_equal(
            actual=np.hstack([acc, err_acc]),
            desired=np.hstack([self.acc, self.err_acc]),
            decimal=4,
            err_msg="[Change Area] - ",
            verbose=True
        )

        # ha
        np.testing.assert_almost_equal(
            actual=np.stack([a_ha, err_ha]),
            desired=np.stack([
                np.array([21_158, 11_686, 285_770, 581_386]),
                np.array([6_158, 3_756, 15_510, 16_282])
            ]),
            decimal=0,
            err_msg="[Change Area] - ",
            verbose=True
        )

class CropAreaTest(unittest.TestCase):
    """Single year crop area estimation testcase.

    Comparison with prior estimates computed for Liaoning, China 2017.

    """

    def setUp(self):
        self.cm = np.array([[248, 15], [26, 179]])

        self.am = np.array([[0.3436694530, 0.0479613932], [0.0360298620, 0.5723392919]])

        self.a_j = np.array([556_725_045, 909_501_053], dtype=np.int64)
        self.w_j = self.a_j / self.a_j.sum()

        self.u_j = np.array([0.91, 0.92])
        self.p_i = np.array([0.88, 0.94])

    def test_area_error_matrix(self):
        am = compute_area_error_matrix(self.cm, self.w_j)
        np.testing.assert_array_almost_equal(
            x=am,
            y=self.am,
            decimal=3,
            err_msg="[Crop Area] - Incorrect error matrix expressed as area proportion.",
            verbose=True,
        )

    def test_user_accuracy(self):
        u_j = compute_u_j(self.am)
        np.testing.assert_array_almost_equal(
            x=u_j,
            y=self.u_j,
            decimal=2,
            err_msg="[Crop Area] - Incorrect user's accuracy.",
            verbose=True,
        )

    def test_variance_user_accuracy(self):
        var_u_j = compute_var_u_j(self.u_j, self.cm)
        np.testing.assert_array_almost_equal(
            x=1.96 * np.sqrt(var_u_j),
            y=np.array([0.035, 0.038]),
            decimal=3,
            err_msg="[Crop Area] - Incorrect variance of user's accuracy.",
            verbose=True,
        )

    def test_producer_accuracy(self):
        p_i = compute_p_i(self.am)
        np.testing.assert_array_almost_equal(
            x=p_i,
            y=self.p_i,
            decimal=2,
            err_msg="[Crop Area] - Incorrect producer's accuracy.",
            verbose=True,
        )

    def test_variance_producer_accuracy(self):
        var_p_i = compute_var_p_i(self.p_i, self.u_j, self.a_j, self.cm)
        np.testing.assert_array_almost_equal(
            x=1.96 * np.sqrt(var_p_i),
            y=np.array([0.053, 0.021]),
            decimal=3,
            err_msg="[Crop Area] - Incorrect variance of producer's accuracy.",
            verbose=True,
        )

    def test_accuracy(self):
        acc = compute_acc(self.am)
        np.testing.assert_almost_equal(
            actual=acc,
            desired=0.916,
            decimal=3,
            err_msg="[Crop Area] - Incorrect accuracy.",
            verbose=True,
        )

    def test_variance_of_accuracy(self):
        var_acc = compute_var_acc(self.w_j, self.u_j, self.cm)
        np.testing.assert_almost_equal(
            actual=1.96 * np.sqrt(var_acc),
            desired=0.027,
            decimal=3,
            err_msg="[Crop Area] - Incorrect variance of accuracy.",
            verbose=True,
        )

    def test_area_estimation(self):
        am = compute_area_error_matrix(self.cm, self.w_j)
        a_i = am.sum(axis=1)
        a_px = a_i * self.a_j.sum()
        a_ha = a_px * (10**2) / (100**2)
        np.testing.assert_almost_equal(
            actual=a_ha,
            desired=np.array([5_742_194, 8_920_067]),
            decimal=0,
            err_msg="[Crop Area] - Incorrect area estimation.",
            verbose=True,
        )

    def test_stddev_of_area_estimate(self):
        std_pi = compute_std_p_i(self.w_j, self.am, self.cm)
        np.testing.assert_almost_equal(
            actual=std_pi,
            desired=np.array([0.0137, 0.0137]),
            decimal=4,
            err_msg="[Crop Area] - Incorrect standard deviation of area estimation.",
            verbose=True,
        )
