from unittest import TestCase
import numpy as np
import numpy.testing as npt

from distancematrix.util import diag_length
from distancematrix.util import diag_indices
from distancematrix.util import diag_indices_of
from distancematrix.util import cross_count
from distancematrix.util import norm_cross_count



class TestUtil(TestCase):
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

    def test_diag_indices_of_rect(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        npt.assert_equal(data[diag_indices_of(data, -2)], [])
        npt.assert_equal(data[diag_indices_of(data, -1)], [4])
        npt.assert_equal(data[diag_indices_of(data, 0)], [1, 5])
        npt.assert_equal(data[diag_indices_of(data, 1)], [2, 6])
        npt.assert_equal(data[diag_indices_of(data, 2)], [3])
        npt.assert_equal(data[diag_indices_of(data, 3)], [])

    def test_cross_count(self):
        data = np.array([0, 1, 2, 3, 4, 5, 6])
        npt.assert_equal(cross_count(data), [0, 0, 0, 0, 0, 0, 0])
        npt.assert_equal(cross_count(data[::-1]), [2, 4, 6, 6, 4, 2, 0])

        npt.assert_equal(data[diag_indices_of(data, 0)], [1, 5])
        npt.assert_equal(data[diag_indices_of(data, 1)], [2, 6])
        npt.assert_equal(data[diag_indices_of(data, 2)], [3])
        npt.assert_equal(data[diag_indices_of(data, 3)], [])

    def test_norm_cross_count(self):
        data = np.arange(20)
        npt.assert_equal(norm_cross_count(data,2), np.ones(data.shape[0]))

        npt.assert_equal(norm_cross_count(data[::-1],2), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        data = [ 2,  4,  6,  8, 10, 12, 14, 16, 16, 14, 14, 16, 14, 14, 14, 14, 12, 10,
                 12, 12, 14, 14, 12, 12, 10,  8,  6,  4,  2,  0]

        npt.assert_allclose(norm_cross_count(data,2), [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                       0.89669421, 0.98412698, 0.83461538, 0.81578947, 0.8037037,
                                                       0.79779412, 0.68382353, 0.57407407, 0.69924812, 0.71538462,
                                                       0.86111111, 1., 1., 1., 1., 1., 1., 1., 1., 1.])

        npt.assert_allclose(norm_cross_count(data[::-1],2), [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                             0.76859504, 0.73809524, 0.59615385, 0.69924812, 0.8037037,
                                                             0.79779412, 0.79779412, 0.8037037 , 0.93233083, 0.83461538,
                                                             0.86111111, 1., 1., 1., 1., 1., 1., 1., 1., 1.])
