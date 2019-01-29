import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.util import diag_indices_of
from distancematrix.consumer.distance_matrix import DistanceMatrix
from distancematrix.calculator import AnytimeCalculator
from distancematrix.calculator import StreamingCalculator
from distancematrix.tests.generator.mock_generator import MockGenerator
from distancematrix.consumer.abstract_consumer import AbstractConsumer


class SummingConsumer(AbstractConsumer):
    """
    Consumer for testing purposes, which simply sums all data channels into a single distance matrix.
    """

    def __init__(self):
        self.distance_matrix = None

    def initialise(self, dims, query_subseq, series_subseq):
        self.distance_matrix = np.full((query_subseq, series_subseq), np.nan, dtype=np.float)

    def process_diagonal(self, diagonal_index, values):
        indices = diag_indices_of(self.distance_matrix, diagonal_index)
        self.distance_matrix[indices] = np.sum(values, axis=0)

    def process_column(self, column_index, values):
        self.distance_matrix[:, column_index] = np.sum(values, axis=0)


class TestAnytimeCalculator(TestCase):
    def test_simple_calculate_columns(self):
        query = np.arange(13)
        series = np.arange(23)
        m = 4

        distance_matrix1 = np.arange(200.).reshape((10, 20))
        distance_matrix2 = np.ones((10, 20), dtype=np.float)
        distance_matrix3 = np.full((10, 20), np.nan, dtype=np.float)
        distance_matrix4 = np.full((10, 20), 5., dtype=np.float)

        calc = AnytimeCalculator(m, series, query)
        calc.add_generator(0, MockGenerator(distance_matrix1))
        calc.add_generator(0, MockGenerator(distance_matrix2))
        calc.add_generator(0, MockGenerator(distance_matrix3))
        calc.add_generator(0, MockGenerator(distance_matrix4))

        consumer1 = DistanceMatrix()
        consumer2 = DistanceMatrix()
        consumer3 = SummingConsumer()
        calc.add_consumer([0], consumer1)
        calc.add_consumer([3], consumer2)
        calc.add_consumer([0, 1, 3], consumer3)

        calc.calculate_columns()

        npt.assert_equal(consumer1.distance_matrix, distance_matrix1)
        npt.assert_equal(consumer2.distance_matrix, distance_matrix4)
        npt.assert_equal(consumer3.distance_matrix, distance_matrix1 + distance_matrix2 + distance_matrix4)

    def test_simple_calculate_columns_partial(self):
        query = np.arange(13)
        series = np.arange(23)
        m = 4

        distance_matrix1 = np.arange(200.).reshape((10, 20))
        distance_matrix2 = np.ones((10, 20), dtype=np.float)
        distance_matrix3 = np.full((10, 20), np.nan, dtype=np.float)
        distance_matrix4 = np.full((10, 20), 5., dtype=np.float)
        summed_matrix = distance_matrix1 + distance_matrix2 + distance_matrix4

        calc = AnytimeCalculator(m, series, query)
        calc.add_generator(0, MockGenerator(distance_matrix1))
        calc.add_generator(0, MockGenerator(distance_matrix2))
        calc.add_generator(0, MockGenerator(distance_matrix3))
        calc.add_generator(0, MockGenerator(distance_matrix4))

        consumer1 = DistanceMatrix()
        consumer2 = DistanceMatrix()
        consumer3 = SummingConsumer()
        calc.add_consumer([0], consumer1)
        calc.add_consumer([3], consumer2)
        calc.add_consumer([0, 1, 3], consumer3)

        # Calculates [3..12[
        calc.calculate_columns(start=3, upto=0.6)
        npt.assert_equal(consumer1.distance_matrix, copy_columns(distance_matrix1, range(3, 12)))
        npt.assert_equal(consumer2.distance_matrix, copy_columns(distance_matrix4, range(3, 12)))
        npt.assert_equal(consumer3.distance_matrix, copy_columns(summed_matrix, range(3, 12)))

        # Calculates [12..15[
        calc.calculate_columns(upto=15)
        npt.assert_equal(consumer1.distance_matrix, copy_columns(distance_matrix1, range(3, 15)))
        npt.assert_equal(consumer2.distance_matrix, copy_columns(distance_matrix4, range(3, 15)))
        npt.assert_equal(consumer3.distance_matrix, copy_columns(summed_matrix, range(3, 15)))

        # Calculates [15..20[
        calc.calculate_columns()
        npt.assert_equal(consumer1.distance_matrix, copy_columns(distance_matrix1, range(3, 20)))
        npt.assert_equal(consumer2.distance_matrix, copy_columns(distance_matrix4, range(3, 20)))
        npt.assert_equal(consumer3.distance_matrix, copy_columns(summed_matrix, range(3, 20)))

        # Calculates [0..3[
        calc.calculate_columns(start=0., upto=3)
        npt.assert_equal(consumer1.distance_matrix, distance_matrix1)
        npt.assert_equal(consumer2.distance_matrix, distance_matrix4)
        npt.assert_equal(consumer3.distance_matrix, summed_matrix)

    def test_simple_calculate_diagonals(self):
        query = np.arange(13)
        series = np.arange(23)
        m = 4

        distance_matrix1 = np.arange(200.).reshape((10, 20))
        distance_matrix2 = np.ones((10, 20), dtype=np.float)
        distance_matrix3 = np.full((10, 20), np.nan, dtype=np.float)
        distance_matrix4 = np.full((10, 20), 5., dtype=np.float)
        summed_matrix = distance_matrix1 + distance_matrix2 + distance_matrix4

        calc = AnytimeCalculator(m, series, query)
        calc.add_generator(0, MockGenerator(distance_matrix1))
        calc.add_generator(0, MockGenerator(distance_matrix2))
        calc.add_generator(0, MockGenerator(distance_matrix3))
        calc.add_generator(0, MockGenerator(distance_matrix4))

        consumer1 = DistanceMatrix()
        consumer2 = DistanceMatrix()
        consumer3 = SummingConsumer()
        calc.add_consumer([0], consumer1)
        calc.add_consumer([3], consumer2)
        calc.add_consumer([0, 1, 3], consumer3)

        calc.calculate_diagonals()

        npt.assert_equal(consumer1.distance_matrix, distance_matrix1)
        npt.assert_equal(consumer2.distance_matrix, distance_matrix4)
        npt.assert_equal(consumer3.distance_matrix, summed_matrix)

    def test_simple_calculate_diagonals_partial(self):
        query = np.arange(13)
        series = np.arange(23)
        m = 4

        distance_matrix1 = np.arange(200.).reshape((10, 20))
        distance_matrix2 = np.ones((10, 20), dtype=np.float)
        distance_matrix3 = np.full((10, 20), np.nan, dtype=np.float)
        distance_matrix4 = np.full((10, 20), 5., dtype=np.float)
        summed_matrix = distance_matrix1 + distance_matrix2 + distance_matrix4
        max_diag = min(len(query) - m + 1, len(series) - m + 1)  # Maximum length of a diagonal

        calc = AnytimeCalculator(m, series, query)
        calc.add_generator(0, MockGenerator(distance_matrix1))
        calc.add_generator(0, MockGenerator(distance_matrix2))
        calc.add_generator(0, MockGenerator(distance_matrix3))
        calc.add_generator(0, MockGenerator(distance_matrix4))

        consumer1 = DistanceMatrix()
        consumer2 = DistanceMatrix()
        consumer3 = SummingConsumer()
        calc.add_consumer([0], consumer1)
        calc.add_consumer([3], consumer2)
        calc.add_consumer([0, 1, 3], consumer3)

        calc.calculate_diagonals(partial=20)
        npt.assert_(20 <= np.count_nonzero(~np.isnan(consumer1.distance_matrix) < 20 + max_diag))
        npt.assert_(20 <= np.count_nonzero(~np.isnan(consumer2.distance_matrix) < 20 + max_diag))
        npt.assert_(20 <= np.count_nonzero(~np.isnan(consumer3.distance_matrix) < 20 + max_diag))

        # For 20 items, at least 3 diagonals are calculated
        for diagonal in calc._diagonal_calc_order[:3]:
            diag_indices = diag_indices_of(distance_matrix1, diagonal)
            npt.assert_equal(consumer1.distance_matrix[diag_indices], distance_matrix1[diag_indices])

        calc.calculate_diagonals(partial=.8)
        npt.assert_(160 <= np.count_nonzero(~np.isnan(consumer1.distance_matrix) < 160 + max_diag))
        npt.assert_(160 <= np.count_nonzero(~np.isnan(consumer2.distance_matrix) < 160 + max_diag))
        npt.assert_(160 <= np.count_nonzero(~np.isnan(consumer3.distance_matrix) < 160 + max_diag))

        calc.calculate_diagonals()
        npt.assert_(200 == np.count_nonzero(~np.isnan(consumer1.distance_matrix)))
        npt.assert_(200 == np.count_nonzero(~np.isnan(consumer2.distance_matrix)))
        npt.assert_(200 == np.count_nonzero(~np.isnan(consumer3.distance_matrix)))
        npt.assert_equal(consumer1.distance_matrix, distance_matrix1)
        npt.assert_equal(consumer2.distance_matrix, distance_matrix4)
        npt.assert_equal(consumer3.distance_matrix, summed_matrix)

    def test_simple_calculate_self_join_diagonals(self):
        series = np.arange(23)
        m = 4
        buffer = 2

        # This test verifies that no values below the main diagonal are used.
        distance_matrix = np.triu(np.arange(1., 401.).reshape((20, 20)), buffer + 1)

        calc = AnytimeCalculator(m, series, trivial_match_buffer=buffer)
        calc.add_generator(0, MockGenerator(distance_matrix))

        consumer = DistanceMatrix()
        calc.add_consumer([0], consumer)

        npt.assert_array_less(np.full(17, buffer), calc._diagonal_calc_order)

        calc.calculate_diagonals()
        expected = distance_matrix + distance_matrix.T
        expected[expected == 0] = np.nan
        npt.assert_equal(consumer.distance_matrix, expected)

    def test_simple_calculate_self_join_columns(self):
        series = np.arange(23)
        m = 4
        buffer = 2

        distance_matrix = np.arange(1., 401.).reshape((20, 20))

        calc = AnytimeCalculator(m, series, trivial_match_buffer=buffer)
        calc.add_generator(0, MockGenerator(distance_matrix))

        consumer = DistanceMatrix()
        calc.add_consumer([0], consumer)

        calc.calculate_columns()

        expected = distance_matrix.copy()
        for diag in range(-buffer, buffer + 1):
            expected[diag_indices_of(expected, diag)] = np.inf

        npt.assert_equal(consumer.distance_matrix, expected)


class TestStreamingCalculator(TestCase):
    def test_streaming_calculate_columns(self):
        dist_matrix = np.arange(1200).astype(dtype=np.float).reshape((30, 40))
        calc = StreamingCalculator(6, 25, 20)
        calc.add_generator(0, MockGenerator(dist_matrix))

        consumer = DistanceMatrix()
        calc.add_consumer([0], consumer)

        calc.calculate_columns()

        expected = np.full((15, 20), np.nan)
        npt.assert_equal(consumer.distance_matrix, expected)

        calc.append_series(np.zeros((1, 15)))
        calc.append_query(np.zeros((1, 20)))
        calc.calculate_columns()

        expected[:, 0:10] = dist_matrix[0:15, 0:10]
        npt.assert_equal(consumer.distance_matrix, expected)

        calc.append_series(np.zeros((1, 10)))
        npt.assert_equal(consumer.distance_matrix, expected)

        calc.calculate_columns()
        expected = dist_matrix[0:15, 0:20]
        npt.assert_equal(consumer.distance_matrix, expected)

        calc.append_series(np.zeros((1, 10)))
        calc.append_query(np.zeros((1, 5)))
        expected = dist_matrix[5:20, 10:30].copy()
        expected[-5:, :] = np.nan
        expected[:, -10:] = np.nan
        npt.assert_equal(consumer.distance_matrix, expected)

        calc.calculate_columns()
        expected = dist_matrix[5:20, 10:30]
        npt.assert_equal(consumer.distance_matrix, expected)

        calc.append_query(np.zeros((1, 5)))
        expected = dist_matrix[10:25, 10:30].copy()
        expected[-5:, :] = np.nan
        npt.assert_equal(consumer.distance_matrix, expected)

        # Expect same result after full calculation, because query shift does not change last column calculated
        calc.calculate_columns()
        npt.assert_equal(consumer.distance_matrix, expected)

        # Correct results when performing full recalculation
        calc.calculate_columns(0, 1.)
        expected = dist_matrix[10:25, 10:30]
        npt.assert_equal(consumer.distance_matrix, expected)


def copy_columns(array, columns):
    """
    Returns an array of the same size as array, with the columns copied, and nan for any other value.
    """
    result = np.full_like(array, np.nan, dtype=np.float)
    result[..., columns] = array[..., columns]
    return result
