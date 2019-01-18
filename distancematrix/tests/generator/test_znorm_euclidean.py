import numpy as np
from unittest import TestCase
import numpy.testing as npt
from abc import abstractmethod

from distancematrix.util import diag_indices
from distancematrix.generator.znorm_euclidean import ZNormEuclidean


class AbstractGeneratorTest(object):
    def setUp(self):
        self.series = np.array(
            [0.2488674, 0.1547179, 2, 2, 2,
             2, 2, 2, 0.02841, 0.371845,
             0.5578463, 0.4555404, 0.18124978, 0.252396, 0.60623881,
             0.5546021, 0.13714127, 0.903246, 0.03695094, 0.23420792,
             0.27482897, 0.57765821, 0.23571178, 0.65772705, 0.00292154,
             0.87258653, 0.29869269, 0.91492178, 0.69096235, 0.6786107,
             0.85413687, 0.19725933, 0.39460891, 0.32650366, 0.35188833,
             0.92658149, 0.07503563, 0.37864432, 0.9415974, 0.62313779])

        self.query = np.array(
            [6., 6., 6., 6., 6.,
             0.03737861, 0.53931239, 0.06194507, 0.0938707, 0.95875364,
             0.09495936, 0.12392364, 0.81358582, 0.56507776, 0.61620183,
             0.24720462, 0.83886639, 0.38130506, 0.13693176, 0.90555723,
             0.23274948, 0.31526678, 0.28504739, 0.45200344, 0.9867946])

        self.m = 5

    @abstractmethod
    def create_generator(self):
        pass

    @abstractmethod
    def bruteforce_matrix(self, m, series, query):
        pass

    def test_non_streaming_calc_diagonal(self):
        gen = self.create_generator().prepare(self.m, self.series, self.query)
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series, self.query)
        _verify_diagonals_correct(bf_dist_matrix, gen)

    def test_non_streaming_calc_column_no_cache(self):
        gen = self.create_generator().prepare(self.m, self.series, self.query)
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series, self.query)
        _verify_columns_correct(bf_dist_matrix, gen, True)

    def test_non_streaming_calc_column_cache(self):
        gen = self.create_generator().prepare(self.m, self.series, self.query)
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series, self.query)
        _verify_columns_correct(bf_dist_matrix, gen, False)

    def test_non_streaming_self_join_calc_diagonal(self):
        gen = self.create_generator().prepare(self.m, self.series)
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series, self.series)
        _verify_diagonals_correct(bf_dist_matrix, gen)

    def test_non_streaming_self_join_calc_column_no_cache(self):
        gen = self.create_generator().prepare(self.m, self.series)
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series, self.series)
        _verify_columns_correct(bf_dist_matrix, gen, True)

    def test_non_streaming_self_join_calc_column_cache(self):
        gen = self.create_generator().prepare(self.m, self.series, self.series)
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series, self.series)
        _verify_columns_correct(bf_dist_matrix, gen, False)

    def test_streaming_calc_diagonal(self):
        gen = self.create_generator().prepare_streaming(self.m, 20, 15)

        gen.append_series(self.series[:10])
        gen.append_query(self.query[:5])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:10], self.query[:5])
        _verify_diagonals_correct(bf_dist_matrix, gen)

        gen.append_series(self.series[10: 15])
        gen.append_query(self.query[5: 10])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:15], self.query[:10])
        _verify_diagonals_correct(bf_dist_matrix, gen)

        gen.append_query(self.query[10: 15])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:15], self.query[:15])
        _verify_diagonals_correct(bf_dist_matrix, gen)

        gen.append_series(self.series[15: 25])
        gen.append_query(self.query[15: 25])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[5: 25], self.query[10: 25])
        _verify_diagonals_correct(bf_dist_matrix, gen)

        gen.append_series(self.series[25:40])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[20: 40], self.query[10: 25])
        _verify_diagonals_correct(bf_dist_matrix, gen)

    def test_streaming_calc_column_no_cache(self):
        gen = self.create_generator().prepare_streaming(self.m, 20, 15)

        gen.append_series(self.series[:10])
        gen.append_query(self.query[:5])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:10], self.query[:5])
        _verify_columns_correct(bf_dist_matrix, gen, True)

        gen.append_series(self.series[10: 15])
        gen.append_query(self.query[5: 10])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:15], self.query[:10])
        _verify_columns_correct(bf_dist_matrix, gen, True)

        gen.append_query(self.query[10: 15])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:15], self.query[:15])
        _verify_columns_correct(bf_dist_matrix, gen, True)

        gen.append_series(self.series[15: 25])
        gen.append_query(self.query[15: 25])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[5: 25], self.query[10: 25])
        _verify_columns_correct(bf_dist_matrix, gen, True)

        gen.append_series(self.series[25:40])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[20: 40], self.query[10: 25])
        _verify_columns_correct(bf_dist_matrix, gen, True)

    def test_streaming_calc_column_cache(self):
        gen = self.create_generator().prepare_streaming(self.m, 20, 15)

        gen.append_series(self.series[:10])
        gen.append_query(self.query[:5])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:10], self.query[:5])
        _verify_columns_correct(bf_dist_matrix, gen, False)

        gen.append_series(self.series[10: 15])
        gen.append_query(self.query[5: 10])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:15], self.query[:10])
        _verify_columns_correct(bf_dist_matrix, gen, False)

        gen.append_query(self.query[10: 15])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:15], self.query[:15])
        _verify_columns_correct(bf_dist_matrix, gen, False)

        gen.append_series(self.series[15: 25])
        gen.append_query(self.query[15: 25])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[5: 25], self.query[10: 25])
        _verify_columns_correct(bf_dist_matrix, gen, False)

        gen.append_series(self.series[25:40])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[20: 40], self.query[10: 25])
        _verify_columns_correct(bf_dist_matrix, gen, False)

    def test_streaming_self_join_calc_diagonal(self):
        gen = self.create_generator().prepare_streaming(self.m, 20)

        gen.append_series(self.series[:10])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:10], self.series[:10])
        _verify_diagonals_correct(bf_dist_matrix, gen)

        gen.append_series(self.series[10: 15])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:15], self.series[:15])
        _verify_diagonals_correct(bf_dist_matrix, gen)

        gen.append_series(self.series[15: 16])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:16], self.series[:16])
        _verify_diagonals_correct(bf_dist_matrix, gen)

        gen.append_series(self.series[16:40])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[20: 40], self.series[20: 40])
        _verify_diagonals_correct(bf_dist_matrix, gen)

    def test_streaming_self_join_calc_column_no_cache(self):
        gen = self.create_generator().prepare_streaming(self.m, 20)

        gen.append_series(self.series[:10])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:10], self.series[:10])
        _verify_columns_correct(bf_dist_matrix, gen, True)

        gen.append_series(self.series[10: 15])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:15], self.series[:15])
        _verify_columns_correct(bf_dist_matrix, gen, True)

        gen.append_series(self.series[15: 16])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:16], self.series[:16])
        _verify_columns_correct(bf_dist_matrix, gen, True)

        gen.append_series(self.series[16:40])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[20: 40], self.series[20: 40])
        _verify_columns_correct(bf_dist_matrix, gen, True)

    def test_streaming_self_join_calc_column_cache(self):
        gen = self.create_generator().prepare_streaming(self.m, 20)

        gen.append_series(self.series[:10])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:10], self.series[:10])
        _verify_columns_correct(bf_dist_matrix, gen, False)

        gen.append_series(self.series[10: 15])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:15], self.series[:15])
        _verify_columns_correct(bf_dist_matrix, gen, False)

        gen.append_series(self.series[15: 16])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[:16], self.series[:16])
        _verify_columns_correct(bf_dist_matrix, gen, False)

        gen.append_series(self.series[16:40])
        bf_dist_matrix = self.bruteforce_matrix(self.m, self.series[20: 40], self.series[20: 40])
        _verify_columns_correct(bf_dist_matrix, gen, False)


class TestZnormEuclidean(AbstractGeneratorTest, TestCase):
    def create_generator(self):
        return ZNormEuclidean()

    def bruteforce_matrix(self, m, series, query):
        return _bruteforce_zeuclidean_distance_matrix(series, query, m, 0.)


class TestZnormEuclideanNoiseElimination(AbstractGeneratorTest, TestCase):
    def create_generator(self):
        return ZNormEuclidean(noise_std=0.2)

    def bruteforce_matrix(self, m, series, query):
        return _bruteforce_zeuclidean_distance_matrix(series, query, m, 0.2)


def _verify_diagonals_correct(bf_distance_matrix, zeuclid):
    h, w = bf_distance_matrix.shape

    for i in range(-h + 1, w):
        result = zeuclid.calc_diagonal(i)
        expected = bf_distance_matrix[diag_indices(h, w, i)]
        npt.assert_allclose(result, expected, atol=1e-10)


def _verify_columns_correct(bf_distance_matrix, euclid, backwards):
    w = bf_distance_matrix.shape[1]

    if backwards:
        r = range(w - 1, -1, -1)
    else:
        r = range(w)

    for i in r:
        result = euclid.calc_column(i)
        expected = bf_distance_matrix[:, i]
        npt.assert_allclose(result, expected, atol=1e-10, err_msg="Mismatch for row {row}".format(row=i))


def _bruteforce_zeuclidean_distance_matrix(series, query, m, noise_std=0.):
    num_cols = len(series) - m + 1
    num_rows = len(query) - m + 1
    distance_matrix = np.zeros((num_rows, num_cols))

    for row in range(num_rows):
        for col in range(num_cols):
            distance_matrix[row, col] = _euclidean_znorm_distance(
                query[row: row + m],
                series[col: col + m],
                m,
                noise_std
            )

    return distance_matrix


def _euclidean_znorm_distance(s1, s2, m, noise_std=0.):
    sq_dist = np.sum(
        np.square(_znorm(s1) - _znorm(s2)))

    if noise_std != 0.:
        max_std = np.maximum(np.std(s1), np.std(s2))
        if max_std != 0:
            sq_dist -= (2 * (m + 1) * np.square(noise_std) /
                        np.square(max_std))
            sq_dist = np.maximum(sq_dist, 0)

    return np.sqrt(sq_dist)


def _znorm(a):
    std = np.std(a)
    if std == 0:
        std = 1
    return (a - np.mean(a)) / std
