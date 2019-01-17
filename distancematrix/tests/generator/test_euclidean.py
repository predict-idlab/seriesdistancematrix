import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.util import diag_indices
from distancematrix.generator.euclidean import Euclidean


class TestEuclidean(TestCase):
    def setUp(self):
        self.series = np.array(
            [0.5578463, 0.4555404, 0.18124978, 0.252396, 0.60623881,
             0.5546021, 0.13714127, 0.903246, 0.03695094, 0.23420792,
             0.27482897, 0.57765821, 0.23571178, 0.65772705, 0.00292154,
             0.87258653, 0.29869269, 0.91492178, 0.69096235, 0.6786107,
             0.85413687, 0.19725933, 0.39460891, 0.32650366, 0.35188833,
             0.92658149, 0.07503563, 0.37864432, 0.9415974, 0.62313779])

        self.query = np.array(
            [0.03737861, 0.53931239, 0.06194507, 0.0938707, 0.95875364,
             0.09495936, 0.12392364, 0.81358582, 0.56507776, 0.61620183,
             0.24720462, 0.83886639, 0.38130506, 0.13693176, 0.90555723,
             0.23274948, 0.31526678, 0.28504739, 0.45200344, 0.9867946])

    def test_calc_diagonal(self):
        m = 5
        euclid = Euclidean().prepare(m, self.series, self.query)
        _verify_diagonals_correct(self.series, self.query, m, euclid)

    def test_calc_column_no_cache(self):
        m = 5
        euclid = Euclidean().prepare(m, self.series, self.query)
        _verify_columns_correct(self.series, self.query, m, euclid, True)

    def test_calc_column_cache(self):
        m = 5
        euclid = Euclidean().prepare(m, self.series, self.query)
        _verify_columns_correct(self.series, self.query, m, euclid, False)


class TestEuclideanSelfJoin(TestCase):
    def setUp(self):
        self.series = np.array(
            [0.5578463, 0.4555404, 0.18124978, 0.252396, 0.60623881,
             0.5546021, 0.13714127, 0.903246, 0.03695094, 0.23420792,
             0.27482897, 0.57765821, 0.23571178, 0.65772705, 0.00292154,
             0.87258653, 0.29869269, 0.91492178, 0.69096235, 0.6786107,
             0.85413687, 0.19725933, 0.39460891, 0.32650366, 0.35188833,
             0.92658149, 0.07503563, 0.37864432, 0.9415974, 0.62313779])

    def test_calc_diagonal(self):
        m = 5
        euclid = Euclidean().prepare(m, self.series)
        _verify_diagonals_correct(self.series, self.series, m, euclid)

    def test_calc_column_no_cache(self):
        m = 5
        euclid = Euclidean().prepare(m, self.series)
        _verify_columns_correct(self.series, self.series, m, euclid, True)

    def test_calc_column_cache(self):
        m = 5
        euclid = Euclidean().prepare(m, self.series)
        _verify_columns_correct(self.series, self.series, m, euclid, False)


class TestStreamingEuclidean(TestCase):
    def setUp(self):
        self.series = np.array(
            [0.5578463, 0.4555404, 0.18124978, 0.252396, 0.60623881,
             0.5546021, 0.13714127, 0.903246, 0.03695094, 0.23420792,
             0.27482897, 0.57765821, 0.23571178, 0.65772705, 0.00292154,
             0.87258653, 0.29869269, 0.91492178, 0.69096235, 0.6786107,
             0.85413687, 0.19725933, 0.39460891, 0.32650366, 0.35188833,
             0.92658149, 0.07503563, 0.37864432, 0.9415974, 0.62313779])

        self.query = np.array(
            [0.03737861, 0.53931239, 0.06194507, 0.0938707, 0.95875364,
             0.09495936, 0.12392364, 0.81358582, 0.56507776, 0.61620183,
             0.24720462, 0.83886639, 0.38130506, 0.13693176, 0.90555723,
             0.23274948, 0.31526678, 0.28504739, 0.45200344, 0.9867946])

    def test_calc_diagonal(self):
        m = 5
        euclid = Euclidean().prepare_streaming(m, 20, 15)

        euclid.append_series(self.series[:10])
        euclid.append_query(self.query[:5])
        _verify_diagonals_correct(self.series[:10], self.query[:5], m, euclid)

        euclid.append_series(self.series[10: 15])
        euclid.append_query(self.query[5: 10])
        _verify_diagonals_correct(self.series[:15], self.query[:10], m, euclid)

        euclid.append_series(self.series[15: 25])
        euclid.append_query(self.query[10: 20])
        _verify_diagonals_correct(self.series[5: 25], self.query[5: 20], m, euclid)

        euclid.append_series(self.series[25:30])
        _verify_diagonals_correct(self.series[10: 30], self.query[5: 20], m, euclid)

    def test_calc_column_no_cache(self):
        m = 5
        euclid = Euclidean().prepare_streaming(m, 20, 15)

        euclid.append_series(self.series[:10])
        euclid.append_query(self.query[:5])
        _verify_columns_correct(self.series[:10], self.query[:5], m, euclid, True)

        euclid.append_series(self.series[10: 15])
        euclid.append_query(self.query[5: 10])
        _verify_columns_correct(self.series[:15], self.query[:10], m, euclid, True)

        euclid.append_series(self.series[15: 25])
        euclid.append_query(self.query[10: 18])
        _verify_columns_correct(self.series[5: 25], self.query[3: 18], m, euclid, True)

        euclid.append_query(self.query[18: 20])
        _verify_columns_correct(self.series[5: 25], self.query[5: 20], m, euclid, True)

        euclid.append_series(self.series[25:30])
        _verify_columns_correct(self.series[10: 30], self.query[5: 20], m, euclid, True)

    def test_calc_column_cache(self):
        m = 5
        euclid = Euclidean().prepare_streaming(m, 20, 15)

        euclid.append_series(self.series[:10])
        euclid.append_query(self.query[:5])
        _verify_columns_correct(self.series[:10], self.query[:5], m, euclid, False)

        euclid.append_series(self.series[10: 15])
        euclid.append_query(self.query[5: 10])
        _verify_columns_correct(self.series[:15], self.query[:10], m, euclid, False)

        euclid.append_series(self.series[15: 25])
        euclid.append_query(self.query[10: 18])
        _verify_columns_correct(self.series[5: 25], self.query[3: 18], m, euclid, False)

        euclid.append_query(self.query[18: 20])
        _verify_columns_correct(self.series[5: 25], self.query[5: 20], m, euclid, False)

        euclid.append_series(self.series[25:30])
        _verify_columns_correct(self.series[10: 30], self.query[5: 20], m, euclid, False)


class TestStreamingEuclideanSelfJoin(TestCase):
    def setUp(self):
        self.series = np.array(
            [0.5578463, 0.4555404, 0.18124978, 0.252396, 0.60623881,
             0.5546021, 0.13714127, 0.903246, 0.03695094, 0.23420792,
             0.27482897, 0.57765821, 0.23571178, 0.65772705, 0.00292154,
             0.87258653, 0.29869269, 0.91492178, 0.69096235, 0.6786107,
             0.85413687, 0.19725933, 0.39460891, 0.32650366, 0.35188833,
             0.92658149, 0.07503563, 0.37864432, 0.9415974, 0.62313779])

    def test_calc_diagonal(self):
        m = 5
        euclid = Euclidean().prepare_streaming(m, 20)

        euclid.append_series(self.series[:10])
        _verify_diagonals_correct(self.series[:10], self.series[:10], m, euclid)

        euclid.append_series(self.series[10: 15])
        _verify_diagonals_correct(self.series[:15], self.series[:15], m, euclid)

        euclid.append_series(self.series[15: 25])
        _verify_diagonals_correct(self.series[5: 25], self.series[5: 25], m, euclid)

        euclid.append_series(self.series[25:30])
        _verify_diagonals_correct(self.series[10: 30], self.series[10: 30], m, euclid)

    def test_calc_column_no_cache(self):
        m = 5
        euclid = Euclidean().prepare_streaming(m, 20)

        euclid.append_series(self.series[:10])
        _verify_columns_correct(self.series[:10], self.series[:10], m, euclid, True)

        euclid.append_series(self.series[10: 15])
        _verify_columns_correct(self.series[:15], self.series[:15], m, euclid, True)

        euclid.append_series(self.series[15: 25])
        _verify_columns_correct(self.series[5: 25], self.series[5: 25], m, euclid, True)

        euclid.append_series(self.series[25:30])
        _verify_columns_correct(self.series[10: 30], self.series[10: 30], m, euclid, True)

    def test_calc_column_cache(self):
        m = 5
        euclid = Euclidean().prepare_streaming(m, 20)

        euclid.append_series(self.series[:10])
        _verify_columns_correct(self.series[:10], self.series[:10], m, euclid, False)

        euclid.append_series(self.series[10: 15])
        _verify_columns_correct(self.series[:15], self.series[:15], m, euclid, False)

        euclid.append_series(self.series[15: 25])
        _verify_columns_correct(self.series[5: 25], self.series[5: 25], m, euclid, False)

        euclid.append_series(self.series[25:30])
        _verify_columns_correct(self.series[10: 30], self.series[10: 30], m, euclid, False)


def _verify_diagonals_correct(series, query, m, euclid):
    h = len(query) - m + 1
    w = len(series) - m + 1
    bf_distance_matrix = _bruteforce_euclidean_distance_matrix(series, query, m)

    for i in range(-h + 1, w):
        result = euclid.calc_diagonal(i)
        expected = bf_distance_matrix[diag_indices(h, w, i)]
        npt.assert_allclose(result, expected)


def _verify_columns_correct(series, query, m, euclid, backwards):
    w = len(series) - m + 1
    bf_distance_matrix = _bruteforce_euclidean_distance_matrix(series, query, m)

    if backwards:
        r = range(w - 1, -1, -1)
    else:
        r = range(w)

    for i in r:
        result = euclid.calc_column(i)
        expected = bf_distance_matrix[:, i]
        npt.assert_allclose(result, expected, err_msg="Mismatch for row {row}".format(row=i))


def _bruteforce_euclidean_distance_matrix(series, query, m):
    num_cols = len(series) - m + 1
    num_rows = len(query) - m + 1
    distance_matrix = np.zeros((num_rows, num_cols))

    for row in range(num_rows):
        for col in range(num_cols):
            distance_matrix[row, col] = _euclidean_distance(
                query[row: row + m],
                series[col: col + m])

    return distance_matrix


def _euclidean_distance(s1, s2):
    return np.sqrt(np.sum(np.square(s1 - s2)))
