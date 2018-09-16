import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.tests.generator.mock_generator import MockGenerator
from distancematrix.generator.filter_generator import _invalid_data_to_invalid_subseq
from distancematrix.generator.filter_generator import FilterGenerator
from distancematrix.generator.filter_generator import is_not_finite


class TestFilterGenerator(TestCase):
    def test_data_points_are_filtered_for_different_query_and_series(self):
        mock_gen = MockGenerator(np.arange(12).reshape((3, 4)))
        filter_gen = FilterGenerator(mock_gen, invalid_data_function=is_not_finite, invalid_subseq_function=None)

        filter_gen.prepare(
            np.array([1, np.inf, 3, 4, 5, np.inf]),
            np.array([np.inf, 2, 3, 4, np.inf]),
            3
        )

        npt.assert_equal(mock_gen.series, [1, 0, 3, 4, 5, 0])
        npt.assert_equal(mock_gen.query, [0, 2, 3, 4, 0])

    def test_data_points_are_filtered_for_same_query_and_series_conserving_equal_array(self):
        mock_gen = MockGenerator(np.arange(9).reshape((3, 3)))
        filter_gen = FilterGenerator(mock_gen, invalid_data_function=is_not_finite, invalid_subseq_function=None)

        data = np.array([np.inf, 2, 3, 4, np.inf])
        filter_gen.prepare(data, data, 3)

        npt.assert_equal(mock_gen.series, [0, 2, 3, 4, 0])
        npt.assert_equal(mock_gen.query, [0, 2, 3, 4, 0])
        npt.assert_(mock_gen.series is mock_gen.query)

    def test_data_points_are_not_filtered_for_invalid_subseqs(self):
        mock_gen = MockGenerator(np.arange(9).reshape((3, 3)))

        def invalid_ss(data, m):
            return np.zeros(3, dtype=np.bool)

        filter_gen = FilterGenerator(mock_gen, invalid_data_function=None, invalid_subseq_function=invalid_ss)

        data = np.array([np.inf, 2, 3, 4, np.inf])
        filter_gen.prepare(data, data, 3)

        npt.assert_equal(mock_gen.series, [np.inf, 2, 3, 4, np.inf])
        npt.assert_equal(mock_gen.query, [np.inf, 2, 3, 4, np.inf])
        npt.assert_(mock_gen.series is mock_gen.query)

    def test_calc_column_with_invalid_data(self):
        mock_gen = MockGenerator(np.arange(12, dtype=np.float).reshape((3, 4)))
        filter_gen = FilterGenerator(mock_gen, invalid_data_function=is_not_finite, invalid_subseq_function=None)

        filter_gen.prepare(
            np.array([1, np.inf, 3, 4, 5, 6], dtype=np.float),
            np.array([1, 2, 3, 4, np.inf], dtype=np.float),
            3
        )

        npt.assert_equal(filter_gen.calc_column(0), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(1), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(2), [2, 6, np.inf])
        npt.assert_equal(filter_gen.calc_column(3), [3, 7, np.inf])

    def test_calc_diag_with_invalid_data(self):
        mock_gen = MockGenerator(np.arange(12, dtype=np.float).reshape((3, 4)))
        filter_gen = FilterGenerator(mock_gen, invalid_data_function=is_not_finite, invalid_subseq_function=None)

        filter_gen.prepare(
            np.array([1, np.inf, 3, 4, 5, 6], dtype=np.float),
            np.array([1, 2, 3, 4, np.inf], dtype=np.float),
            3
        )

        # i i 2 3
        # i i 6 7
        # i i i i
        npt.assert_equal(filter_gen.calc_diagonal(-2), [np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(-1), [np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(0), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(1), [np.inf, 6, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(2), [2, 7])
        npt.assert_equal(filter_gen.calc_diagonal(3), [3])

    def test_calc_column_with_invalid_subseq(self):
        mock_gen = MockGenerator(np.arange(12, dtype=np.float).reshape((3, 4)))

        def invalid_ss(data, m):
            if data.shape[0] == 6:
                return np.array([1, 0, 0, 1], dtype=np.bool)  # Invalid series subseqs
            if data.shape[0] == 5:
                return np.array([0, 0, 1, ], dtype=np.bool)  # Invalid query subseqs
            raise RuntimeError()

        filter_gen = FilterGenerator(mock_gen, invalid_data_function=None, invalid_subseq_function=invalid_ss)

        filter_gen.prepare(
            np.array([1, 2, 3, 4, 5, 6], dtype=np.float),
            np.array([1, 2, 3, 4, 5], dtype=np.float),
            3
        )

        npt.assert_equal(filter_gen.calc_column(0), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(1), [1, 5, np.inf])
        npt.assert_equal(filter_gen.calc_column(2), [2, 6, np.inf])
        npt.assert_equal(filter_gen.calc_column(3), [np.inf, np.inf, np.inf])

    def test_calc_diag_with_invalid_subseq(self):
        mock_gen = MockGenerator(np.arange(12, dtype=np.float).reshape((3, 4)))

        def invalid_ss(data, m):
            if data.shape[0] == 6:
                return np.array([1, 0, 0, 1], dtype=np.bool)  # Invalid series subseqs
            if data.shape[0] == 5:
                return np.array([0, 0, 1, ], dtype=np.bool)  # Invalid query subseqs
            raise RuntimeError()

        filter_gen = FilterGenerator(mock_gen, invalid_data_function=None, invalid_subseq_function=invalid_ss)

        filter_gen.prepare(
            np.array([1, 2, 3, 4, 5, 6], dtype=np.float),
            np.array([1, 2, 3, 4, 5], dtype=np.float),
            3
        )

        # i 1 2 i
        # i 5 6 i
        # i i i i
        npt.assert_equal(filter_gen.calc_diagonal(-2), [np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(-1), [np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(0), [np.inf, 5, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(1), [1, 6, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(2), [2, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(3), [np.inf])


class TestHelperMethods(TestCase):
    def test_invalid_data_to_invalid_subseq(self):
        data = np.array([0, 0, 0, 0, 0, 0], dtype=np.bool)
        corr = np.array([0, 0, 0, 0], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([1, 0, 0, 0, 0, 0], dtype=np.bool)
        corr = np.array([1, 0, 0, 0], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 1, 0, 0, 0, 0], dtype=np.bool)
        corr = np.array([1, 1, 0, 0], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 0, 1, 0, 0, 0], dtype=np.bool)
        corr = np.array([1, 1, 1, 0], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 0, 0, 1, 0, 0], dtype=np.bool)
        corr = np.array([0, 1, 1, 1], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 0, 0, 0, 1, 0], dtype=np.bool)
        corr = np.array([0, 0, 1, 1], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 0, 0, 0, 0, 1], dtype=np.bool)
        corr = np.array([0, 0, 0, 1], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([1, 0, 1, 0, 0, 0], dtype=np.bool)
        corr = np.array([1, 1, 1, 0], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 0, 1, 0, 1, 0], dtype=np.bool)
        corr = np.array([1, 1, 1, 1], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)