import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.util import diag_indices_of
from distancematrix.consumer.distance_matrix import DistanceMatrix


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

    def mock_initialise(self, dm):
        dm.initialise(1, self.dist_matrix.shape[0], self.dist_matrix.shape[1])

    def test_process_diagonal(self):
        dm = DistanceMatrix()
        self.mock_initialise(dm)

        for diag in range(-self.dist_matrix.shape[0] + 1, self.dist_matrix.shape[1]):
            diag_ind = diag_indices_of(self.dist_matrix, diag)
            dm.process_diagonal(diag, self.dist_matrix[diag_ind])

        npt.assert_equal(dm.distance_matrix, self.dist_matrix)

    def test_process_diagonal_partial_calculation(self):
        dm = DistanceMatrix()
        self.mock_initialise(dm)

        correct = np.full_like(self.dist_matrix, np.nan, dtype=np.float)

        for diag in range(-8, self.dist_matrix.shape[1], 3):
            diag_ind = diag_indices_of(self.dist_matrix, diag)
            dm.process_diagonal(diag, self.dist_matrix[diag_ind])
            correct[diag_ind] = self.dist_matrix[diag_ind]

        npt.assert_equal(dm.distance_matrix, correct)

    def test_process_column(self):
        dm = DistanceMatrix()
        self.mock_initialise(dm)

        for column in range(0, self.dist_matrix.shape[1]):
            dm.process_column(column, self.dist_matrix[:, column])

        npt.assert_equal(dm.distance_matrix, self.dist_matrix)

    def test_process_column_partial_calculation(self):
        dm = DistanceMatrix()
        self.mock_initialise(dm)

        correct = np.full_like(self.dist_matrix, np.nan, dtype=np.float)

        for column in [2, 3, 4, 5, 10, 11, 12]:
            dm.process_column(column, self.dist_matrix[:, column])
            correct[:, column] = self.dist_matrix[:, column]

        npt.assert_equal(dm.distance_matrix, correct)
