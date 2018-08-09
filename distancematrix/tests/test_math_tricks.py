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


class TestSlidingMeanStd(TestCase):
    def test_sliding_mean_std(self):
        random_gen = np.random.RandomState(0)

        data_array = [
            np.array([5.15, 2.15, 1.05, -9.2, 0.01, 7.14, 4.18, 10.2, 3.25, 14.1, -9.85, 5.12, 0.11, 0.14, 0.98]),
            np.array([0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -50, -50, -50, -50, -50, -50]),
            np.array([1e14, 1.6e19, 0.9e18, 6e14, 5.6e10, 9.9e16, 9e17, 6.48e14, 9.2e14, 1e18, 3.14e13]),
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
        data = np.array([1e18, 3.14e13, 3.14e13, 42., 1])
        npt.assert_allclose(
            math_tricks.sliding_mean_std(data, 2)[0],
            brute_sliding_mean(data, 2))

    def test_sliding_std_numerical_stability(self):
        # For some reason, only 9.2e14 caused loss of precision in the old implementation.
        data = np.array([1e19, 1e18, 1e15, 1e10, 1e16, 1e17, 1e15, 9.2e14])
        npt.assert_allclose(
            math_tricks.sliding_mean_std(data, 2)[1],
            brute_sliding_std(data, 2))


class TestStreamingStatistics(TestCase):
    def test_different_m(self):
        data = np.array([
            5.15, 2.15, 1.05, -9.2, 0.01, 7.14, 4.18, 10.2, 3.25, 14.1,
            -9.85, 5.12, 0.11, 0.14, 0.98, 0., 0., 0., 0., 0.,
            0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., -50, -50, -50, -50, -50, -50, 1e14, 1.6e19, 0.9e18,
            6e14, 5.6e10, 9.9e16, 9e17, 6.48e14, 9.2e14, 1e18, 3.14e13, 42, 1
        ])

        self._test_for_params(data, 10, 5)
        self._test_for_params(data, 10, 4)
        self._test_for_params(data, 10, 3)
        #self._test_for_params(data, 10, 2) # Temporarily disabled, error margin is a bit too high
        self._test_for_params(data, 10, 1)
        self._test_for_params(data, 5, 5)
        self._test_for_params(data, 5, 4)
        self._test_for_params(data, 5, 3)
        #self._test_for_params(data, 5, 2) # Temporarily disabled, error margin is a bit too high
        self._test_for_params(data, 5, 1)

    def test_different_stepsize(self):
        data = np.array([
            5.15, 2.15, 1.05, -9.2, 0.01, 7.14, 4.18, 10.2, 3.25, 14.1,
            -9.85, 5.12, 0.11, 0.14, 0.98, 0., 0., 0., 0., 0.,
            0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., -50, -50, -50, -50, -50, -50, 1e14, 1.6e19, 0.9e18,
            6e14, 5.6e10, 9.9e16, 9e17, 6.48e14, 9.2e14, 1e18, 3.14e13, 42, 1
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
                atol=2e-15, err_msg="Different for window starting at " + str(start) + ": " + str(ss.var - expected_var))
