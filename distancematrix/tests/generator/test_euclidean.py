import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.util import diag_indices
from distancematrix.generator.euclidean import Euclidean


class TestEuclidean(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = np.array([0.5578463, 0.4555404, 0.18124978, 0.252396, 0.60623881,
                           0.5546021, 0.13714127, 0.903246, 0.03695094, 0.23420792,
                           0.27482897, 0.57765821, 0.23571178, 0.65772705, 0.00292154,
                           0.87258653, 0.29869269, 0.91492178, 0.69096235, 0.6786107,
                           0.85413687, 0.19725933, 0.39460891, 0.32650366, 0.35188833,
                           0.92658149, 0.07503563, 0.37864432, 0.9415974, 0.62313779])

        cls.query = np.array([0.03737861, 0.53931239, 0.06194507, 0.0938707, 0.95875364,
                          0.09495936, 0.12392364, 0.81358582, 0.56507776, 0.61620183,
                          0.24720462, 0.83886639, 0.38130506, 0.13693176, 0.90555723,
                          0.23274948, 0.31526678, 0.28504739, 0.45200344, 0.9867946])

        cls.m = 5

        cls.m = 1
        cls.series = np.array([1, 2, 3])
        cls.query = np.array([5, 10])

        num_cols = len(cls.series) - cls.m + 1
        num_rows = len(cls.query) - cls.m + 1
        cls.distance_matrix = np.zeros((num_rows, num_cols))

        for row in range(num_rows):
            for col in range(num_cols):
                cls.distance_matrix[row, col] = cls.euclidean_distance(
                    cls.query[row : row+cls.m],
                    cls.series[col : col+cls.m])

    def setUp(self):
        self.euclid = Euclidean()
        self.euclid.prepare(TestEuclidean.series, TestEuclidean.query, TestEuclidean.m)

    @staticmethod
    def euclidean_distance(s1, s2):
        return np.sqrt(np.sum(np.square(s1 - s2)))

    def test_calc_diagonal(self):
        h = len(TestEuclidean.query) - TestEuclidean.m + 1
        w = len(TestEuclidean.series) - TestEuclidean.m + 1

        for i in range(-h+1, w):
            result = self.euclid.calc_diagonal(i)
            expected = TestEuclidean.distance_matrix[diag_indices(h, w, i)]
            npt.assert_allclose(result, expected)

    def test_calc_column_no_cache(self):
        w = len(TestEuclidean.series) - TestEuclidean.m + 1

        for i in range(w-1, -1, -1):
            result = self.euclid.calc_column(i)
            expected = TestEuclidean.distance_matrix[:, i]
            npt.assert_allclose(result, expected)

    def test_calc_column_cache(self):
        w = len(TestEuclidean.series) - TestEuclidean.m + 1

        for i in range(w):
            result = self.euclid.calc_column(i)
            expected = TestEuclidean.distance_matrix[:, i]
            npt.assert_allclose(result, expected)

