import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.util import diag_indices
from distancematrix.generator.znorm_euclidean import ZNormEuclidean


class TestZnormEuclidean(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = np.array([2, 2, 2, 2, 2, 2, 0.02841,
                               0.5578463, 0.4555404, 0.18124978, 0.252396, 0.60623881,
                               0.5546021, 0.13714127, 0.903246, 0.03695094, 0.23420792,
                               0.27482897, 0.57765821, 0.23571178, 0.65772705, 0.00292154,
                               0.87258653, 0.29869269, 0.91492178, 0.69096235, 0.6786107,
                               0.85413687, 0.19725933, 0.39460891, 0.32650366, 0.35188833,
                               0.92658149, 0.07503563, 0.37864432, 0.9415974, 0.62313779])

        cls.query = np.array([6., 6., 6., 6., 6.,
                              0.03737861, 0.53931239, 0.06194507, 0.0938707, 0.95875364,
                              0.09495936, 0.12392364, 0.81358582, 0.56507776, 0.61620183,
                              0.24720462, 0.83886639, 0.38130506, 0.13693176, 0.90555723,
                              0.23274948, 0.31526678, 0.28504739, 0.45200344, 0.9867946])

        cls.m = 5

        num_cols = len(cls.series) - cls.m + 1
        num_rows = len(cls.query) - cls.m + 1
        cls.distance_matrix = np.zeros((num_rows, num_cols))

        for row in range(num_rows):
            for col in range(num_cols):
                cls.distance_matrix[row, col] = cls.euclidean_znorm_distance(
                    cls.query[row: row + cls.m],
                    cls.series[col: col + cls.m])

    def setUp(self):
        self.zeuclid = ZNormEuclidean(noise_std=0)
        self.zeuclid.prepare(TestZnormEuclidean.series, TestZnormEuclidean.query, TestZnormEuclidean.m)

    @staticmethod
    def euclidean_znorm_distance(s1, s2):
        return np.sqrt(np.sum(np.square(TestZnormEuclidean.znorm(s1) - TestZnormEuclidean.znorm(s2))))

    @staticmethod
    def znorm(a):
        std = np.std(a)
        if std == 0:
            std = 1
        return (a - np.mean(a)) / std

    def test_calc_diagonal(self):
        h = len(TestZnormEuclidean.query) - TestZnormEuclidean.m + 1
        w = len(TestZnormEuclidean.series) - TestZnormEuclidean.m + 1

        for i in range(-h + 1, w):
            result = self.zeuclid.calc_diagonal(i)
            expected = TestZnormEuclidean.distance_matrix[diag_indices(h, w, i)]
            npt.assert_allclose(result, expected, atol=1e-10)

    def test_calc_column_no_cache(self):
        w = len(TestZnormEuclidean.series) - TestZnormEuclidean.m + 1

        for i in range(w - 1, -1, -1):
            result = self.zeuclid.calc_column(i)
            expected = TestZnormEuclidean.distance_matrix[:, i]
            npt.assert_allclose(result, expected, atol=1e-10)

    def test_calc_column_cache(self):
        w = len(TestZnormEuclidean.series) - TestZnormEuclidean.m + 1

        for i in range(w):
            result = self.zeuclid.calc_column(i)
            expected = TestZnormEuclidean.distance_matrix[:, i]
            npt.assert_allclose(result, expected, atol=1e-10)


class TestZnormEuclideanNoiseElimination(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = np.array([2, 2, 2, 2, 2, 2, 0.02841,
                               0.5578463, 0.4555404, 0.18124978, 0.252396, 0.60623881,
                               0.5546021, 0.13714127, 0.903246, 0.03695094, 0.23420792,
                               0.27482897, 0.57765821, 0.23571178, 0.65772705, 0.00292154,
                               0.87258653, 0.29869269, 0.91492178, 0.69096235, 0.6786107,
                               0.85413687, 0.19725933, 0.39460891, 0.32650366, 0.35188833,
                               0.92658149, 0.07503563, 0.37864432, 0.9415974, 0.62313779])

        cls.query = np.array([6., 6., 6., 6., 6.,
                              0.03737861, 0.53931239, 0.06194507, 0.0938707, 0.95875364,
                              0.09495936, 0.12392364, 0.81358582, 0.56507776, 0.61620183,
                              0.24720462, 0.83886639, 0.38130506, 0.13693176, 0.90555723,
                              0.23274948, 0.31526678, 0.28504739, 0.45200344, 0.9867946])

        cls.m = 5
        cls.noise_std = 0.2

        num_cols = len(cls.series) - cls.m + 1
        num_rows = len(cls.query) - cls.m + 1
        cls.distance_matrix = np.zeros((num_rows, num_cols))

        for row in range(num_rows):
            for col in range(num_cols):
                cls.distance_matrix[row, col] = cls.euclidean_znorm_distance(
                    cls.query[row: row + cls.m],
                    cls.series[col: col + cls.m])

    def setUp(self):
        self.zeuclid = ZNormEuclidean(noise_std=TestZnormEuclideanNoiseElimination.noise_std)
        self.zeuclid.prepare(
            TestZnormEuclideanNoiseElimination.series,
            TestZnormEuclideanNoiseElimination.query,
            TestZnormEuclideanNoiseElimination.m)

    @staticmethod
    def euclidean_znorm_distance(s1, s2):
        m = TestZnormEuclideanNoiseElimination.m
        sq_dist = np.sum(
            np.square(TestZnormEuclideanNoiseElimination.znorm(s1) - TestZnormEuclideanNoiseElimination.znorm(s2)))

        max_std = np.maximum(np.std(s1), np.std(s2))
        if max_std != 0:
            sq_dist -= (2 * (m + 1) * np.square(TestZnormEuclideanNoiseElimination.noise_std) /
                        np.square(max_std))
            sq_dist = np.maximum(sq_dist, 0)
        return np.sqrt(sq_dist)

    @staticmethod
    def znorm(a):
        std = np.std(a)
        if std == 0:
            std = 1
        return (a - np.mean(a)) / std

    def test_calc_diagonal(self):
        h = len(TestZnormEuclideanNoiseElimination.query) - TestZnormEuclideanNoiseElimination.m + 1
        w = len(TestZnormEuclideanNoiseElimination.series) - TestZnormEuclideanNoiseElimination.m + 1

        for i in range(-h + 1, w):
            result = self.zeuclid.calc_diagonal(i)
            expected = TestZnormEuclideanNoiseElimination.distance_matrix[diag_indices(h, w, i)]
            npt.assert_allclose(result, expected, atol=1e-10)

    def test_calc_column_no_cache(self):
        w = len(TestZnormEuclideanNoiseElimination.series) - TestZnormEuclideanNoiseElimination.m + 1

        for i in range(w - 1, -1, -1):
            result = self.zeuclid.calc_column(i)
            expected = TestZnormEuclideanNoiseElimination.distance_matrix[:, i]
            npt.assert_allclose(result, expected, atol=1e-10)

    def test_calc_column_cache(self):
        w = len(TestZnormEuclideanNoiseElimination.series) - TestZnormEuclideanNoiseElimination.m + 1

        for i in range(w):
            result = self.zeuclid.calc_column(i)
            expected = TestZnormEuclideanNoiseElimination.distance_matrix[:, i]
            npt.assert_allclose(result, expected, atol=1e-10)
