import numpy as np
from unittest import TestCase
import numpy.testing as npt

import distancematrix.math_tricks as math_tricks


def brute_sliding_mean(data, m):
    return np.array([np.mean(data[i:i + m]) for i in range(len(data) - m + 1)])


def brute_sliding_var(data, m):
    return np.array([np.var(data[i:i + m]) for i in range(len(data) - m + 1)])


def brute_sliding_std(data, m):
    return np.array([np.std(data[i:i + m]) for i in range(len(data) - m + 1)])

MEAN_STABILITY_DATA = np.array([
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 43.33, 69.39, 76.01, 76.03, 75.19, 82.21, 91.37, 86.44, 88.09, 88.56, 98.88, 91.62, 93.97, 90.81, 88.25,
    95.3, 100., 95.96, 98.13, 97.57, 94.02, 95.24, 92.59, 98.98, 100., 100., 100., 97.88, 96.33, 98.07, 95.18,
    93.52, 79.99, 37.08, 13.9, 17.43, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 58.58, 70.16, 83.06, 82.79, 85.38, 100., 100., 100.,
    100., 100., 85.97, 56.18, 0., 0., 18.69, 0., 0., 13.9, 13.94, 25.69, 34.33, 65.06, 80.1, 85.65,
    84.57, 83.74, 94.75, 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
    100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
    100., 100., 100., 100., 100., 100., 100., 100., 90.1, 79.01, 65.47, 54.24, 25.05, 15.01, 0., 0.])

# For a subsequence length of 24, this data array provided a lot of approximation errors for various techniques
# that were tested to calculate sliding variance/std.
STD_VAR_STABILITY_DATA = np.array([
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 43.33, 69.39, 76.01, 76.03, 75.19, 82.21, 91.37, 86.44, 88.09,
    88.56, 98.88, 91.62, 93.97, 90.81, 88.25, 95.3, 100., 95.96, 98.13, 97.57, 94.02, 95.24, 92.59,
    98.98, 100., 100., 100., 97.88, 96.33, 98.07, 95.18, 93.52, 79.99, 37.08, 13.9, 17.43, 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 58.58, 70.16, 83.06, 82.79, 85.38, 100., 100., 100.,
    100., 100., 85.97, 56.18, 12., 12., 18.69, 12., 12., 13.9, 13.94, 25.69, 34.33, 65.06,
    80.1, 85.65, 84.57, 83.74, 94.75, 100., 100., 100., 100., 100., 100., 100., 100., 100.,
    100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
    100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
    90.1, 79.01, 65.47, 54.24, 25.05, 15.01, 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 15.94, 42.61, 71.12, 100., 100., 100., 100., 100., 100., 100., 100., 100.,
    100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
    100., 100., 100., 120., 120., 120., 120., 120., 120., 120., 14.69, 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 15.19, 14.81, 22.67, 31.61, 32.21, 39.68, 47.36, 52.63, 61.79, 62.49, 67.66, 120.,
    120., 120., 120., 109.44, 87.13, 51.72, 55.24, 57.78, 62.97, 66.43, 120., 120., 120., 120.,
    110.46, 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 31.04, 52.73, 49.78, 57.56, 66.5, 66.92, 75.89, 88.17,
    97.98, 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
    100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
    100., 100., 100., 49.6, 45.2, 13.15, 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
    12., 12., 12., 12., 12., 12., 12., 12., 12., 12.])


class TestSlidingMeanStdVar(TestCase):
    def test_sliding_mean_std(self):
        random_gen = np.random.RandomState(0)

        data_array = [
            np.array([5.15, 2.15, 1.05, -9.2, 0.01, 7.14, 4.18, 10.2, 3.25, 14.1, -9.85, 5.12, 0.11, 0.14, 0.98]),
            np.array([0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -50, -50, -50, -50, -50, -50]),
            np.array([1e8, 1.6e9, 0.9e8, 6e4, 5.6e2, 9.9e6, 9e7, 6.48e4, 9.2e4, 1e8, 3.14e7]),
            random_gen.rand(1000)
        ]
        m = 5

        for data in data_array:
            correct_mean = [np.mean(data[i:i + m]) for i in range(len(data) - m + 1)]
            correct_std = [np.std(data[i:i + m]) for i in range(len(data) - m + 1)]

            mean, std = math_tricks.sliding_mean_std(data, m)

            npt.assert_allclose(mean, correct_mean)
            npt.assert_allclose(std, correct_std)

    def test_sliding_mean_numerical_stability(self):
        npt.assert_allclose(
            math_tricks.sliding_mean_std(MEAN_STABILITY_DATA, 24)[0],
            brute_sliding_mean(MEAN_STABILITY_DATA, 24), )

    def test_sliding_std_numerical_stability(self):
        npt.assert_allclose(
            math_tricks.sliding_mean_std(STD_VAR_STABILITY_DATA, 24)[1],
            brute_sliding_std(STD_VAR_STABILITY_DATA, 24))

    def test_sliding_var_numerical_stability(self):
        npt.assert_allclose(
            math_tricks.sliding_mean_var(STD_VAR_STABILITY_DATA, 24)[1],
            brute_sliding_var(STD_VAR_STABILITY_DATA, 24))


class TestStreamingStatistics(TestCase):
    def test_different_m(self):
        data = np.array([
            5.15, 2.15, 1.05, -9.2, 0.01, 7.14, 4.18, 10.2, 3.25, 14.1,
            -9.85, 5.12, 0.11, 0.14, 0.98, 0., 0., 0., 0., 0.,
            0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., -50, -50, -50, -50, -50, -50, 1e8, 1.6e9, 0.9e8,
            6e4, 5.6e2, 9.9e6, 9e7, 6.48e4, 9.2e4, 1e8, 3.14e7, 42., 1.
        ])

        self._test_for_params(data, 10, 5)
        self._test_for_params(data, 10, 4)
        self._test_for_params(data, 10, 3)
        self._test_for_params(data, 10, 2)
        self._test_for_params(data, 10, 1)
        self._test_for_params(data, 5, 5)
        self._test_for_params(data, 5, 4)
        self._test_for_params(data, 5, 3)
        self._test_for_params(data, 5, 2)
        self._test_for_params(data, 5, 1)

    def test_different_stepsize(self):
        data = np.array([
            5.15, 2.15, 1.05, -9.2, 0.01, 7.14, 4.18, 10.2, 3.25, 14.1,
            -9.85, 5.12, 0.11, 0.14, 0.98, 0., 0., 0., 0., 0.,
            0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., -50, -50, -50, -50, -50, -50, 1e8, 1.6e9, 0.9e8,
            6e4, 5.6e2, 9.9e6, 9e7, 6.48e4, 9.2e4, 1e8, 3.14e7, 42., 1.
        ])

        self._test_for_params(data, 10, 5, 1)
        self._test_for_params(data, 10, 5, 2)
        self._test_for_params(data, 10, 5, 3)
        self._test_for_params(data, 10, 5, 4)
        self._test_for_params(data, 10, 5, 5)
        self._test_for_params(data, 10, 5, 6)
        self._test_for_params(data, 10, 5, 7)
        self._test_for_params(data, 10, 5, 9)
        self._test_for_params(data, 10, 5, 10)
        self._test_for_params(data, 10, 5, 11)
        self._test_for_params(data, 10, 5, 12)

    def _test_for_params(self, data, data_len, m, stepsize=1):
        start = 0
        ss = math_tricks.StreamingStats(data[:data_len], m)

        npt.assert_equal(ss.data, data[start: start + data_len])
        npt.assert_allclose(ss.mean, [np.mean(data[start + i: start + i + m]) for i in range(data_len - m + 1)])
        npt.assert_allclose(ss.var, [np.var(data[start + i: start + i + m]) for i in range(data_len - m + 1)])

        while start + data_len + stepsize < len(data):
            ss.append(data[start + data_len: start + data_len + stepsize])
            start += stepsize
            npt.assert_equal(ss.data, data[start: start + data_len])
            npt.assert_allclose(
                ss.mean, [np.mean(data[start + i: start + i + m]) for i in range(data_len - m + 1)],
                atol=2e-15, err_msg="Different for window starting at " + str(start))

            expected_var = [np.var(data[start + i: start + i + m]) for i in range(data_len - m + 1)]
            npt.assert_allclose(
                ss.var, expected_var,
                atol=2e-15,
                err_msg="Different for window starting at " + str(start) + ": " + str(ss.var - expected_var))

    def test_stability(self):
        self._test_for_params(STD_VAR_STABILITY_DATA, 50, 24)
