from unittest import TestCase
import numpy as np
import numpy.testing as npt

from distancematrix.util import diag_length
from distancematrix.util import diag_indices
from distancematrix.util import diag_indices_of
from distancematrix.util import cut_indices_of
from distancematrix.util import shortest_path_distances
from distancematrix.util import shortest_path

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

    def test_cut_indices_of(self):
        data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        npt.assert_equal(data[cut_indices_of(data, 0)], [1])
        npt.assert_equal(data[cut_indices_of(data, 1)], [4, 2])
        npt.assert_equal(data[cut_indices_of(data, 2)], [7, 5, 3])
        npt.assert_equal(data[cut_indices_of(data, 3)], [10, 8, 6])
        npt.assert_equal(data[cut_indices_of(data, 4)], [11, 9])
        npt.assert_equal(data[cut_indices_of(data, 5)], [12])

        npt.assert_equal(data[cut_indices_of(data, 6)], [])

        data = np.array([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]
        ])

        npt.assert_equal(data[cut_indices_of(data, 0)], [0])
        npt.assert_equal(data[cut_indices_of(data, 1)], [5, 1])
        npt.assert_equal(data[cut_indices_of(data, 2)], [6, 2])
        npt.assert_equal(data[cut_indices_of(data, 3)], [7, 3])
        npt.assert_equal(data[cut_indices_of(data, 4)], [8, 4])
        npt.assert_equal(data[cut_indices_of(data, 5)], [9])

    def test_shortest_path_distances(self):
        data = np.array([
            [1, 2, 1, 0, 3],
            [1, 3, 0, 1, 1],
            [0, 1, 1, 4, 0],
            [2, 5, 5, 2, 2],
            [0, 1, 2, 3, 9]
        ], dtype=np.float)

        expected = np.array([
            [1, 3, 4, 4, 7],
            [2, 4, 3, 4, 5],
            [2, 3, 4, 7, 4],
            [4, 7, 8, 6, 6],
            [4, 5, 7, 9, 15]
        ], dtype=np.float)

        result = shortest_path_distances(data)
        npt.assert_equal(result, expected)

        result = shortest_path_distances(data[:3, :])
        npt.assert_equal(result, expected[:3, :])

        result = shortest_path_distances(data[:, :3])
        npt.assert_equal(result, expected[:, :3])

    def test_shortest_path(self):
        data = np.array([
            [1, 2, 1, 0, 3],
            [1, 3, 3, 1, 1],
            [4, 3, 8, 4, 0],
            [2, 2, 5, 2, 5],
            [0, 1, 1, 3, 2],
            [0, 1, 1, 5, 9]
        ], dtype=np.float)

        result = shortest_path(data)
        npt.assert_equal(result, [[0, 0], [1, 0], [2, 1], [3, 1], [4, 2], [4, 3], [5, 4]])