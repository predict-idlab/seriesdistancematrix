import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.util import diag_indices_of
from distancematrix.consumer.distance_matrix import DistanceMatrix
from distancematrix.calculator import Calculator
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


class TestCalculator(TestCase):
    def test_simple_calculate_columns(self):
        query = np.arange(13)
        series = np.arange(23)
        m = 4

        distance_matrix1 = np.arange(200.).reshape((10, 20))
        distance_matrix2 = np.ones((10, 20), dtype=np.float)
        distance_matrix3 = np.full((10, 20), np.nan, dtype=np.float)
        distance_matrix4 = np.full((10, 20), 5., dtype=np.float)

        calc = Calculator(query, series, m)
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

        calc = Calculator(query, series, m)
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

        calc = Calculator(query, series, m)
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

        calc = Calculator(query, series, m)
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


def copy_columns(array, columns):
    """
    Returns an array of the same size as array, with the columns copied, and nan for any other value.
    """
    result = np.full_like(array, np.nan, dtype=np.float)
    result[..., columns] = array[..., columns]
    return result
