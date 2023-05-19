import os, sys
import numpy as np
import unittest

from src.area_utils_refac import (
    compute_area_error_matrix,
    compute_u_j,
    compute_var_u_j,
    compute_p_i,
    compute_var_p_i,
    compute_acc,
    compute_var_acc,
    compute_std_p_i
)

class ChangeAreaTest(unittest.TestCase):
    """ Multi-year change area estimation testcase.

    Comparison with 'toy example' seen in Olofsson et al., 2014 "Good Practices...". 
    
    """

    def setUp(self):
        self.cm = np.array([
            [66,0,1,2],
            [0,55,0,1],
            [5,8,153,9],
            [4,12,11,313]
        ])

        self.am = np.array([
            [0.0176, 0.0000, 0.0019, 0.0040],
            [0.0000, 0.0110, 0.0000, 0.0020],
            [0.0013, 0.0016, 0.2967, 0.0179],
            [0.0011, 0.0024, 0.0213, 0.6212]
        ])

        self.a_j = np.array([200_000, 150_000, 3_200_000, 6_450_000], dtype = np.int64)
        self.w_j = np.array([0.020, 0.015, 0.320, 0.645], dtype = np.float64)

        self.u_j = np.array([0.88, 0.73, 0.93, 0.96])
        self.p_i = np.array([0.75, 0.85, 0.93, 0.96])

    def test_area_error_matrix(self):
        am = compute_area_error_matrix(self.cm, self.w_j)
        np.testing.assert_array_almost_equal(
            x = am,
            y = self.am,
            decimal = 3,
            err_msg = "[Change Area] - Incorrect error matrix expressed as area proportion.",
            verbose = True
        )

    def test_user_accuracy(self):
        u_j = compute_u_j(self.am)
        np.testing.assert_array_almost_equal(
            x = u_j,
            y = self.u_j,
            decimal = 2, 
            err_msg = "[Change Area] - Incorrect user's accuracy.",
            verbose = True
        )

    def test_variance_user_accuracy(self):
        var_u_j = compute_var_u_j(self.u_j, self.cm)
        np.testing.assert_array_almost_equal(
            x = 1.96 * np.sqrt(var_u_j),
            y = np.array([0.074, 0.101, 0.040, 0.021]),
            decimal = 3,
            err_msg = "[Change Area] - Incorrect variance of user's accuracy.",
            verbose = True
        )
    
    def test_producer_accuracy(self):
        p_i = compute_p_i(self.am)
        np.testing.assert_array_almost_equal(
            x = p_i,
            y = self.p_i,
            decimal = 2,
            err_msg = "[Change Area] - Incorrect producer's accuracy.",
            verbose = True
        )

    def test_variance_producer_accuracy(self):
        var_p_i = compute_var_p_i(self.p_i, self.u_j, self.a_j, self.cm)
        np.testing.assert_array_almost_equal(
            x = 1.96 * np.sqrt(var_p_i),
            y = np.array([0.213, 0.254, 0.034, 0.018]),
            decimal = 3,
            err_msg = "[Change Area] - Incorrect variance of producer's accuracy.",
            verbose = True
        )

    def test_accuracy(self):
        acc = compute_acc(self.am)
        np.testing.assert_almost_equal(
            actual = acc,
            desired = 0.947,
            decimal = 3,
            err_msg = "[Change Area] - Incorrect accuracy.",
            verbose = True
        )
    
    def test_variance_of_accuracy(self):
        var_acc = compute_var_acc(self.w_j, self.u_j, self.cm)
        np.testing.assert_almost_equal(
            actual = 1.96 * np.sqrt(var_acc),
            desired = 0.018,
            decimal = 3,
            err_msg = "[Change Area] - Incorrect variance of accuracy.",
            verbose = True
        )

    def test_area_estimation(self):
        am = compute_area_error_matrix(self.cm, self.w_j)
        a_i = am.sum(axis = 1)
        a_px = a_i * self.a_j.sum()
        a_ha = a_px *  0.09
        np.testing.assert_almost_equal(
            actual = a_ha,
            desired = np.array([21_158, 11_686, 285_770, 581_386]),
            decimal = 0,
            err_msg = "[Change Area] - Incorrect area estimation.",
            verbose = True
        )

    def test_stddev_of_area_estimate(self):
        std_pi = compute_std_p_i(self.w_j, self.am, self.cm)
        np.testing.assert_almost_equal(
            actual = std_pi,
            desired = np.array([0.0035, 0.0021, 0.0088, 0.0092]),
            decimal = 4,
            err_msg = "[Change Area] - Incorrect standard deviation of area estimation.",
            verbose = True
        )

class CropAreaTest(unittest.TestCase):
    """ Single year crop area estimation testcase.
    
    Comparison with prior estimates computed for Liaoning, China 2017.

    """

    def setUp(self):
        self.cm = np.array([
            [248, 15],
            [26, 179]
        ])

        self.am = np.array([
            [0.3437, 0.0480],
            [0.0360, 0.5723]
        ])

        self.a_j = np.array([556_725_045, 909_501_053], dtype = np.int64)
        self.w_j = np.array([0.380, 0.620], dtype = np.float64)

        self.u_j = np.array([0.91, 0.92])
        self.p_i = np.array([0.88, 0.94])

    def test_area_error_matrix(self):
        am = compute_area_error_matrix(self.cm, self.w_j)
        np.testing.assert_array_almost_equal(
            x = am,
            y = self.am,
            decimal = 3,
            err_msg = "[Crop Area] - Incorrect error matrix expressed as area proportion.",
            verbose = True
        )

    def test_user_accuracy(self):
        u_j = compute_u_j(self.am)
        np.testing.assert_array_almost_equal(
            x = u_j,
            y = self.u_j,
            decimal = 2, 
            err_msg = "[Crop Area] - Incorrect user's accuracy.",
            verbose = True
        )

    def test_variance_user_accuracy(self):
        var_u_j = compute_var_u_j(self.u_j, self.cm)
        np.testing.assert_array_almost_equal(
            x = 1.96 * np.sqrt(var_u_j),
            y = np.array([0.035, 0.038]),
            decimal = 3,
            err_msg = "[Crop Area] - Incorrect variance of user's accuracy.",
            verbose = True
        )
    
    def test_producer_accuracy(self):
        p_i = compute_p_i(self.am)
        np.testing.assert_array_almost_equal(
            x = p_i,
            y = self.p_i,
            decimal = 2,
            err_msg = "[Crop Area] - Incorrect producer's accuracy.",
            verbose = True
        )

    def test_variance_producer_accuracy(self):
        var_p_i = compute_var_p_i(self.p_i, self.u_j, self.a_j, self.cm)
        np.testing.assert_array_almost_equal(
            x = 1.96 * np.sqrt(var_p_i),
            y = np.array([0.053, 0.021]),
            decimal = 3,
            err_msg = "[Crop Area] - Incorrect variance of producer's accuracy.",
            verbose = True
        )

    def test_accuracy(self):
        acc = compute_acc(self.am)
        np.testing.assert_almost_equal(
            actual = acc,
            desired = 0.916,
            decimal = 3,
            err_msg = "[Crop Area] - Incorrect accuracy.",
            verbose = True
        )
    
    def test_variance_of_accuracy(self):
        var_acc = compute_var_acc(self.w_j, self.u_j, self.cm)
        np.testing.assert_almost_equal(
            actual = 1.96 * np.sqrt(var_acc),
            desired = 0.027,
            decimal = 3,
            err_msg = "[Crop Area] - Incorrect variance of accuracy.",
            verbose = True
        )

    @unittest.skip("Skip until original pixel size for this project becomes known.")
    def test_area_estimation(self):
        am = compute_area_error_matrix(self.cm, self.w_j)
        a_i = am.sum(axis = 1)
        a_px = a_i * self.a_j.sum()
        a_ha = a_px *  (9.999995 ** 2) / (100 ** 2) # Aproximation b/c unknown
        np.testing.assert_almost_equal(
            actual = a_ha,
            desired = np.array([5_742_194, 8_920_067]),
            decimal = 0,
            err_msg = "[Crop Area] - Incorrect area estimation.",
            verbose = True
        )

    def test_stddev_of_area_estimate(self):
        std_pi = compute_std_p_i(self.w_j, self.am, self.cm)
        np.testing.assert_almost_equal(
            actual = std_pi,
            desired = np.array([0.0137, 0.0137]),
            decimal = 4,
            err_msg = "[Crop Area] - Incorrect standard deviation of area estimation.",
            verbose = True
        )