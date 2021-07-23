from typing import Iterable, Tuple

import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.util import diag_indices_of
from distancematrix.consumer.contextual_matrix_profile import ContextualMatrixProfile
from distancematrix.consumer.contextmanager import GeneralStaticManager
from distancematrix.consumer.contextmanager import AbstractContextManager


class TestContextualMatrixProfile(TestCase):

    def setUp(self):
        self.dist_matrix = np.array([
            [8.67, 1.10, 1.77, 1.26, 1.91, 4.29, 6.32, 4.24, 4.64, 5.06, 6.41, 4.07, 4.67, 9.32, 5.09],
            [4.33, 4.99, 0.14, 2.79, 2.10, 6.26, 9.40, 4.14, 5.53, 4.26, 8.21, 5.91, 6.83, 9.26, 6.19],
            [0.16, 9.05, 1.35, 4.78, 7.01, 4.36, 5.24, 8.81, 7.90, 5.84, 8.90, 7.88, 3.37, 4.70, 6.94],
            [0.94, 8.70, 3.87, 6.29, 0.32, 1.79, 5.80, 2.61, 1.43, 6.32, 1.62, 0.20, 2.28, 7.11, 2.15],
            [9.90, 4.51, 2.11, 2.83, 5.52, 8.55, 6.90, 0.24, 1.58, 4.26, 8.75, 3.71, 9.93, 8.33, 0.38],
            [7.30, 5.84, 9.63, 1.95, 3.76, 3.61, 9.42, 5.56, 5.09, 7.07, 1.90, 4.78, 1.06, 0.69, 3.67],
            [2.17, 8.37, 3.99, 4.28, 4.37, 2.86, 8.61, 3.39, 8.37, 6.95, 6.57, 1.79, 7.40, 4.41, 7.64],
            [6.26, 0.29, 6.44, 8.84, 1.24, 2.52, 6.25, 3.07, 5.55, 3.19, 8.16, 5.32, 9.01, 0.39, 9.],
            [4.67, 8.88, 3.05, 3.06, 2.36, 8.34, 4.91, 5.46, 9.25, 9.78, 0.03, 5.64, 5.10, 3.58, 6.92],
            [1.01, 0.91, 6.28, 7.79, 0.68, 5.50, 6.72, 5.11, 0.80, 9.30, 9.77, 4.71, 3.26, 7.29, 6.26]])

        self.m = 5

    def mock_initialise(self, cdm):
        cdm.initialise(1, self.dist_matrix.shape[0], self.dist_matrix.shape[1])

    def bruteforce_cdm(self, dist_matrix, query_ranges, series_ranges):
        """
        Brute force calculation of contextual distance matrix.
        :param dist_matrix: 2D matrix
        :param query_ranges: list of length m containing: ranges or list of ranges
        :param series_ranges: list of length n containing: ranges or list of ranges
        :return: 2D matrix (m by n)
        """

        correct = np.full((len(query_ranges), len(series_ranges)), np.inf, dtype=float)
        correct_qi = np.full((len(query_ranges), len(series_ranges)), -1, dtype=int)
        correct_si = np.full((len(query_ranges), len(series_ranges)), -1, dtype=int)

        for i, r0s in enumerate(query_ranges):
            if not isinstance(r0s, list):
                r0s = [r0s]

            for r0 in r0s:
                r0 = slice(r0.start, r0.stop)
                for j, r1s in enumerate(series_ranges):
                    if not isinstance(r1s, list):
                        r1s = [r1s]

                    for r1 in r1s:
                        r1 = slice(r1.start, r1.stop)
                        view = dist_matrix[r0, :][:, r1]

                        if view.size == 0:
                            continue

                        min_value = np.min(view)

                        if correct[i, j] > min_value:
                            correct[i, j] = np.min(view)
                            correct_qi[i, j], correct_si[i, j] = next(zip(*np.where(view == np.min(view))))
                            correct_qi[i, j] += r0.start
                            correct_si[i, j] += r1.start

        return correct, correct_qi, correct_si

    def test_process_diagonal(self):
        query_ranges = [range(0, 4), range(7, 8)]
        series_ranges = [range(1, 4), range(6, 8)]

        correct, correct_qi, correct_si = self.bruteforce_cdm(self.dist_matrix, query_ranges, series_ranges)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for diag in range(-self.dist_matrix.shape[0] + 1, self.dist_matrix.shape[1]):
            diag_ind = diag_indices_of(self.dist_matrix, diag)
            cdm.process_diagonal(diag, np.atleast_2d(self.dist_matrix[diag_ind]))

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)

    def test_process_diagonal_partial_calculation(self):
        # Ranges selected so that the diagonals have a mix of the number of numbers that are filled in
        query_ranges = [range(0, 2), range(2, 5), range(7, 8), range(8, 9), range(9, 10)]
        series_ranges = [range(0, 2), range(2, 4), range(4, 10), range(13, 14)]

        part_dist_matrix = np.full_like(self.dist_matrix, np.inf)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for diag in range(-8, self.dist_matrix.shape[1], 4):
            diag_ind = diag_indices_of(self.dist_matrix, diag)
            cdm.process_diagonal(diag, np.atleast_2d(self.dist_matrix[diag_ind]))
            part_dist_matrix[diag_ind] = self.dist_matrix[diag_ind]

        correct, correct_qi, correct_si = self.bruteforce_cdm(part_dist_matrix, query_ranges, series_ranges)

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)

    def test_process_diagonal_complex_ranges(self):
        # Overlapping ranges and contexts consisting of multiple ranges
        query_ranges = [range(0, 10), range(1, 5), range(1, 2), range(4, 5),
                        [range(1, 2), range(3, 4), range(7, 9)]]
        series_ranges = [range(0, 2), range(1, 3), range(2, 4), range(3, 6), range(4, 8), range(4, 10),
                         [range(0, 3), range(3, 5), range(13, 15)]]

        correct, correct_qi, correct_si = self.bruteforce_cdm(self.dist_matrix, query_ranges, series_ranges)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for diag in range(-self.dist_matrix.shape[0] + 1, self.dist_matrix.shape[1]):
            diag_ind = diag_indices_of(self.dist_matrix, diag)
            cdm.process_diagonal(diag, np.atleast_2d(self.dist_matrix[diag_ind]))

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)

    def test_process_diagonal_context_goes_beyond_distancematrix(self):
        query_ranges = [range(0, 8), range(8, 16)]
        series_ranges = [range(0, 10), range(10, 20)]

        correct, correct_qi, correct_si = self.bruteforce_cdm(self.dist_matrix, query_ranges, series_ranges)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for diag in range(-self.dist_matrix.shape[0] + 1, self.dist_matrix.shape[1]):
            diag_ind = diag_indices_of(self.dist_matrix, diag)
            cdm.process_diagonal(diag, np.atleast_2d(self.dist_matrix[diag_ind]))

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)

    def test_process_diagonal_context_goes_beyond_distancematrix_2(self):
        self.dist_matrix = self.dist_matrix.T

        query_ranges = [range(0, 8), range(13, 20)]
        series_ranges = [range(0, 10), range(9, 13)]

        correct, correct_qi, correct_si = self.bruteforce_cdm(self.dist_matrix, query_ranges, series_ranges)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for diag in range(-self.dist_matrix.shape[0] + 1, self.dist_matrix.shape[1]):
            diag_ind = diag_indices_of(self.dist_matrix, diag)
            cdm.process_diagonal(diag, np.atleast_2d(self.dist_matrix[diag_ind]))

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)

    def test_process_diagonal_context_falls_outside_distancematrix(self):
        query_ranges = [range(0, 8), range(8, 16), range(20, 30)]
        series_ranges = [range(0, 10), range(10, 20), range(30, 40)]

        correct, correct_qi, correct_si = self.bruteforce_cdm(self.dist_matrix, query_ranges, series_ranges)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for diag in range(-self.dist_matrix.shape[0] + 1, self.dist_matrix.shape[1]):
            diag_ind = diag_indices_of(self.dist_matrix, diag)
            cdm.process_diagonal(diag, np.atleast_2d(self.dist_matrix[diag_ind]))

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)

    def test_process_column(self):
        query_ranges = [range(0, 4), range(7, 8)]
        series_ranges = [range(1, 4), range(6, 8)]

        correct, correct_qi, correct_si = self.bruteforce_cdm(self.dist_matrix, query_ranges, series_ranges)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for column in range(0, self.dist_matrix.shape[1]):
            cdm.process_column(column, np.atleast_2d(self.dist_matrix[:, column]))

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)

    def test_process_column_partial_calculation(self):
        # Ranges selected so that the contexts will receive 0, 1, 2 or 3 columns
        query_ranges = [range(0, 2), range(2, 5), range(7, 10)]
        series_ranges = [range(0, 2), range(2, 4), range(4, 6), range(9, 14)]

        part_dist_matrix = np.full_like(self.dist_matrix, np.inf)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for column in [2, 3, 4, 5, 10, 11, 12]:
            cdm.process_column(column, np.atleast_2d(self.dist_matrix[:, column]))
            part_dist_matrix[:, column] = self.dist_matrix[:, column]

        correct, correct_qi, correct_si = self.bruteforce_cdm(part_dist_matrix, query_ranges, series_ranges)

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)

    def test_process_column_complex_ranges(self):
        # Overlapping ranges and contexts consisting of multiple ranges
        query_ranges = [range(0, 10), range(1, 5), range(1, 2), range(4, 5),
                        [range(1, 2), range(3, 4), range(7, 9)]]
        series_ranges = [range(0, 2), range(1, 3), range(2, 4), range(3, 6), range(4, 8), range(4, 10),
                         [range(0, 3), range(3, 5), range(13, 15)]]

        correct, correct_qi, correct_si = self.bruteforce_cdm(self.dist_matrix, query_ranges, series_ranges)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for column in range(0, self.dist_matrix.shape[1]):
            cdm.process_column(column, np.atleast_2d(self.dist_matrix[:, column]))

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)

    def test_process_column_context_goes_beyond_distancematrix(self):
        query_ranges = [range(0, 8), range(8, 16)]
        series_ranges = [range(0, 10), range(10, 20)]

        correct, correct_qi, correct_si = self.bruteforce_cdm(self.dist_matrix, query_ranges, series_ranges)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for column in range(0, self.dist_matrix.shape[1]):
            cdm.process_column(column, np.atleast_2d(self.dist_matrix[:, column]))

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)

    def test_process_column_context_falls_outside_distancematrix(self):
        query_ranges = [range(0, 8), range(8, 16), range(20, 30)]
        series_ranges = [range(0, 10), range(10, 20), range(30, 40)]

        correct, correct_qi, correct_si = self.bruteforce_cdm(self.dist_matrix, query_ranges, series_ranges)

        cdm = ContextualMatrixProfile(GeneralStaticManager(series_ranges, query_ranges))
        self.mock_initialise(cdm)

        for column in range(0, self.dist_matrix.shape[1]):
            cdm.process_column(column, np.atleast_2d(self.dist_matrix[:, column]))

        npt.assert_allclose(cdm.distance_matrix, correct)
        npt.assert_equal(cdm.match_index_query, correct_qi)
        npt.assert_equal(cdm.match_index_series, correct_si)


class MockContextManager(AbstractContextManager):
    def __init__(self) -> None:
        self.series_shift = 0
        self.query_shift = 0

    def context_matrix_shape(self) -> (int, int):
        return 2, 4

    def shift_query(self, amount: int) -> int:
        self.query_shift += amount
        if self.query_shift == 2:
            return 0
        elif self.query_shift == 4:
            return 1
        else:
            raise RuntimeError("Invalid test state")

    def shift_series(self, amount: int) -> int:
        self.series_shift += amount
        if self.series_shift == 1:
            return 0
        elif self.series_shift == 2:
            return 1
        else:
            raise RuntimeError("Invalid test state")

    def series_contexts(self, start: int, stop: int) -> Iterable[Tuple[int, int, int]]:
        if self.series_shift == 0:
            ctxs = [(0, 2, 0), (2, 4, 1), (4, 6, 2)]
        elif self.series_shift == 1:
            ctxs = [(0, 1, 0), (1, 3, 1), (3, 5, 2), (5, 6, 3)]
        elif self.series_shift == 2:
            ctxs = [(0, 2, 0), (2, 4, 1), (4, 6, 2)]
        else:
            raise RuntimeError("Invalid test state")
        return filter(lambda c: c[0] < stop and c[1] > start, ctxs)

    def query_contexts(self, start: int, stop: int) -> Iterable[Tuple[int, int, int]]:
        if self.query_shift == 0:
            ctxs = [(0, 3, 0), (3, 10, 1)]
        elif self.query_shift == 2:
            ctxs = [(0, 1, 0), (1, 10, 1)]
        elif self.query_shift == 4:
            ctxs = [(0, 10, 0), (2, 8, 1)]
        else:
            raise RuntimeError("Invalid test state")
        return filter(lambda c: c[0] < stop and c[1] > start, ctxs)


class TestStreamingContextualMatrixProfile(TestCase):
    def setUp(self):
        self.dist_matrix = np.array([
            [9.2, 4.6, 0.5, 0.0, 9.7, 8.4, 8.8, 7.2],
            [8.4, 3.3, 3.8, 6.2, 2.3, 7.3, 1.0, 9.7],
            [7.2, 3.1, 6.1, 3.2, 8.0, 5.9, 2.7, 8.1],
            [5.3, 7.3, 5.8, 9.0, 2.6, 1.0, 2.9, 8.8],
            [8.9, 4.8, 6.4, 6.7, 2.1, 6.4, 1.2, 5.2],
            [4.2, 8.1, 2.3, 4.4, 3.4, 6.6, 1.5, 4.2],
            [1.7, 8.8, 1.2, 0.4, 3.7, 9.3, 4.4, 3.7],
            [1.6, 7.8, 4.0, 8.9, 0.1, 7.5, 2.5, 8.]
        ])

    def test_streaming_process_column(self):
        cdm = ContextualMatrixProfile(MockContextManager())
        cdm.initialise(1, 4, 6)

        cdm.process_column(0, np.atleast_2d(self.dist_matrix[0:4, 0]))
        cdm.process_column(2, np.atleast_2d(self.dist_matrix[0:4, 2]))

        npt.assert_equal(cdm.distance_matrix, [[7.2, 0.5, np.Inf, np.Inf], [5.3, 5.8, np.Inf, np.Inf]])
        npt.assert_equal(cdm.match_index_query, [[2, 0, -1, -1], [3, 3, -1, -1]])
        npt.assert_equal(cdm.match_index_series, [[0, 2, -1, -1], [0, 2, -1, -1]])

        cdm.shift_series(1)
        cdm.shift_query(2)

        npt.assert_equal(cdm.distance_matrix, [[7.2, 0.5, np.Inf, np.Inf], [5.3, 5.8, np.Inf, np.Inf]])
        npt.assert_equal(cdm.match_index_query, [[2, 0, -1, -1], [3, 3, -1, -1]])
        npt.assert_equal(cdm.match_index_series, [[0, 2, -1, -1], [0, 2, -1, -1]])

        cdm.process_column(0, np.atleast_2d(self.dist_matrix[2:6, 1]))
        cdm.process_column(1, np.atleast_2d(self.dist_matrix[2:6, 2]))
        cdm.process_column(3, np.atleast_2d(self.dist_matrix[2:6, 4]))

        npt.assert_equal(cdm.distance_matrix, [[3.1, 0.5, 8.0, np.Inf], [4.8, 2.3, 2.1, np.Inf]])
        npt.assert_equal(cdm.match_index_query, [[2, 0, 2, -1], [4, 5, 4, -1]])
        npt.assert_equal(cdm.match_index_series, [[1, 2, 4, -1], [1, 2, 4, -1]])

        cdm.shift_series(1)
        cdm.shift_query(2)

        npt.assert_equal(cdm.distance_matrix, [[2.3, 2.1, np.Inf, np.Inf], [np.Inf, np.Inf, np.Inf, np.Inf]])
        npt.assert_equal(cdm.match_index_query, [[5, 4, -1, -1], [-1, -1, -1, -1]])
        npt.assert_equal(cdm.match_index_series, [[2, 4, -1, -1], [-1, -1, -1, -1]])

        cdm.process_column(0, np.atleast_2d(self.dist_matrix[4:8, 2]))
        cdm.process_column(3, np.atleast_2d(self.dist_matrix[4:8, 5]))
        cdm.process_column(4, np.atleast_2d(self.dist_matrix[4:8, 6]))
        cdm.process_column(5, np.atleast_2d(self.dist_matrix[4:8, 7]))

        npt.assert_equal(cdm.distance_matrix, [[1.2, 2.1, 1.2, np.Inf], [1.2, 7.5, 2.5, np.Inf]])
        npt.assert_equal(cdm.match_index_query, [[6, 4, 4, -1], [6, 7, 7, -1]])
        npt.assert_equal(cdm.match_index_series, [[2, 4, 6, -1], [2, 5, 6, -1]])

    def test_streaming_process_diagonal(self):
        cdm = ContextualMatrixProfile(MockContextManager())
        cdm.initialise(1, 4, 6)

        v = self.dist_matrix[0:4, 0:6]
        cdm.process_diagonal(0, np.atleast_2d(v[diag_indices_of(v, 0)]))
        cdm.process_diagonal(2, np.atleast_2d(v[diag_indices_of(v, 2)]))
        cdm.process_diagonal(-1, np.atleast_2d(v[diag_indices_of(v, -1)]))

        npt.assert_equal(cdm.distance_matrix, [[3.1, 0.5, 8., np.Inf], [np.Inf, 5.8, 1., np.Inf]])
        npt.assert_equal(cdm.match_index_query, [[2, 0, 2, -1], [-1, 3, 3, -1]])
        npt.assert_equal(cdm.match_index_series, [[1, 2, 4, -1], [-1, 2, 5, -1]])

        cdm.shift_series(1)
        cdm.shift_query(2)
        v = self.dist_matrix[2:6, 1:7]

        npt.assert_equal(cdm.distance_matrix, [[3.1, 0.5, 8., np.Inf], [np.Inf, 5.8, 1., np.Inf]])
        npt.assert_equal(cdm.match_index_query, [[2, 0, 2, -1], [-1, 3, 3, -1]])
        npt.assert_equal(cdm.match_index_series, [[1, 2, 4, -1], [-1, 2, 5, -1]])

        cdm.process_diagonal(5, np.atleast_2d(v[diag_indices_of(v, 5)]))
        cdm.process_diagonal(1, np.atleast_2d(v[diag_indices_of(v, 1)]))
        cdm.process_diagonal(0, np.atleast_2d(v[diag_indices_of(v, 0)]))
        cdm.process_diagonal(-3, np.atleast_2d(v[diag_indices_of(v, -3)]))

        npt.assert_equal(cdm.distance_matrix, [[3.1, 0.5, 8., 2.7], [8.1, 5.8, 1., np.Inf]])
        npt.assert_equal(cdm.match_index_query, [[2, 0, 2, 2], [5, 3, 3, -1]])
        npt.assert_equal(cdm.match_index_series, [[1, 2, 4, 6], [1, 2, 5, -1]])

        cdm.shift_series(1)
        cdm.shift_query(2)
        v = self.dist_matrix[4:8, 2:8]

        npt.assert_equal(cdm.distance_matrix, [[5.8, 1., np.Inf, np.Inf], [np.Inf, np.Inf, np.Inf, np.Inf]])
        npt.assert_equal(cdm.match_index_query, [[3, 3, -1, -1], [-1, -1, -1, -1]])
        npt.assert_equal(cdm.match_index_series, [[2, 5, -1, -1], [-1, -1, -1, -1]])

        cdm.process_diagonal(-2, np.atleast_2d(v[diag_indices_of(v, -2)]))
        cdm.process_diagonal(0, np.atleast_2d(v[diag_indices_of(v, 0)]))
        cdm.process_diagonal(1, np.atleast_2d(v[diag_indices_of(v, 1)]))
        cdm.process_diagonal(2, np.atleast_2d(v[diag_indices_of(v, 2)]))
        cdm.process_diagonal(3, np.atleast_2d(v[diag_indices_of(v, 3)]))
        cdm.process_diagonal(4, np.atleast_2d(v[diag_indices_of(v, 4)]))

        npt.assert_equal(cdm.distance_matrix, [[1.2, 1., 1.2, np.Inf], [1.2, 3.7, 2.5, np.Inf]])
        npt.assert_equal(cdm.match_index_query, [[6, 3, 4, -1], [6, 6, 7, -1]])
        npt.assert_equal(cdm.match_index_series, [[2, 5, 6, -1], [2, 4, 6, -1]])
