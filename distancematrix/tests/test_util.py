from unittest import TestCase
import numpy as np
import numpy.testing as npt

from distancematrix.util import diag_length
from distancematrix.util import diag_indices


class TestEuclidean(TestCase):
    def test_diag_length_square_matrix(self):
        self.assertEqual(diag_length(5, 5, 0), 5)
        self.assertEqual(diag_length(5, 5, 1), 4)
        self.assertEqual(diag_length(5, 5, -2), 3)
        self.assertEqual(diag_length(5, 5, 4), 1)
        self.assertEqual(diag_length(5, 5, 5), 0)
        self.assertEqual(diag_length(5, 5, 6), 0)

    def test_diag_length_rect_matrix(self):
        self.assertEqual(diag_length(5, 3, 0), 3)
        self.assertEqual(diag_length(5, 3, 1), 2)
        self.assertEqual(diag_length(5, 3, 2), 1)
        self.assertEqual(diag_length(5, 3, 3), 0)
        self.assertEqual(diag_length(5, 3, 4), 0)

        self.assertEqual(diag_length(5, 3, -1), 3)
        self.assertEqual(diag_length(5, 3, -2), 3)
        self.assertEqual(diag_length(5, 3, -3), 2)
        self.assertEqual(diag_length(5, 3, -4), 1)
        self.assertEqual(diag_length(5, 3, -5), 0)
        self.assertEqual(diag_length(5, 3, -6), 0)

        self.assertEqual(diag_length(3, 5, 0), 3)
        self.assertEqual(diag_length(3, 5, 1), 3)
        self.assertEqual(diag_length(3, 5, 2), 3)
        self.assertEqual(diag_length(3, 5, 3), 2)
        self.assertEqual(diag_length(3, 5, 4), 1)
        self.assertEqual(diag_length(3, 5, 5), 0)
        self.assertEqual(diag_length(3, 5, 6), 0)

        self.assertEqual(diag_length(3, 5, -1), 2)
        self.assertEqual(diag_length(3, 5, -2), 1)
        self.assertEqual(diag_length(3, 5, -3), 0)
        self.assertEqual(diag_length(3, 5, -4), 0)

    def test_diag_indices_square(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        npt.assert_equal(data[diag_indices(3, 3, -3)], [])
        npt.assert_equal(data[diag_indices(3, 3, -2)], [7])
        npt.assert_equal(data[diag_indices(3, 3, -1)], [4, 8])
        npt.assert_equal(data[diag_indices(3, 3, 0)], [1, 5, 9])
        npt.assert_equal(data[diag_indices(3, 3, 1)], [2, 6])
        npt.assert_equal(data[diag_indices(3, 3, 2)], [3])
        npt.assert_equal(data[diag_indices(3, 3, 3)], [])

    def test_diag_indices_rect(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        npt.assert_equal(data[diag_indices(2, 3, -2)], [])
        npt.assert_equal(data[diag_indices(2, 3, -1)], [4])
        npt.assert_equal(data[diag_indices(2, 3, 0)], [1, 5])
        npt.assert_equal(data[diag_indices(2, 3, 1)], [2, 6])
        npt.assert_equal(data[diag_indices(2, 3, 2)], [3])
        npt.assert_equal(data[diag_indices(2, 3, 3)], [])
