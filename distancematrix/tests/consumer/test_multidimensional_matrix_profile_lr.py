import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.util import diag_indices_of
from distancematrix.consumer.multidimensional_matrix_profile_lr import MultidimensionalMatrixProfileLR


class TestContextualMatrixProfile(TestCase):

    def setUp(self):
        self.dist_matrix = np.array([[
            [8.67, 1.10, 1.77, 1.26, 1.91, 4.29, 6.32, 4.24, 4.64, 5.06, 6.41, 4.07, 4.67, 9.32, 5.09],
            [4.33, 4.99, 0.14, 2.79, 2.10, 6.26, 9.40, 4.14, 5.53, 4.26, 8.21, 5.91, 6.83, 9.26, 6.19],
            [0.16, 9.05, 1.35, 4.78, 7.01, 4.36, 5.24, 8.81, 7.90, 5.84, 8.90, 7.88, 3.37, 4.70, 6.94],
            [0.94, 8.70, 3.87, 6.29, 0.32, 1.79, 5.80, 2.61, 1.43, 6.32, 1.62, 0.20, 2.28, 7.11, 2.15],
            [9.90, 4.51, 2.11, 2.83, 5.52, 8.55, 6.90, 0.24, 1.58, 4.26, 8.75, 3.71, 9.93, 8.33, 0.38],
            [7.30, 5.84, 9.63, 1.95, 3.76, 3.61, 9.42, 5.56, 5.09, 7.07, 1.90, 4.78, 1.06, 0.69, 3.67],
            [2.17, 8.37, 3.99, 4.28, 4.37, 2.86, 8.61, 3.39, 8.37, 6.95, 6.57, 1.79, 7.40, 4.41, 7.64],
            [6.26, 0.29, 6.44, 8.84, 1.24, 2.52, 6.25, 3.07, 5.55, 3.19, 8.16, 5.32, 9.01, 0.39, 9.],
            [4.67, 8.88, 3.05, 3.06, 2.36, 8.34, 4.91, 5.46, 9.25, 9.78, 0.03, 5.64, 5.10, 3.58, 6.92],
            [1.01, 0.91, 6.28, 7.79, 0.68, 5.50, 6.72, 5.11, 0.80, 9.30, 9.77, 4.71, 3.26, 7.29, 6.26]],

            [[5.89, 3.38, 2.54, 3.61, 6.17, 3.90, 3.41, 2.94, 5.02, 3.24, 0.65, 4.29, 7.93, 8.39, 0.34],
             [9.07, 6.32, 9.13, 0.22, 6.60, 9.55, 2.96, 6.94, 4.12, 4.06, 0.64, 0.55, 6.09, 0.47, 2.39],
             [6.53, 0.98, 2.08, 1.82, 3.91, 6.91, 0.90, 9.13, 4.49, 7.40, 1.67, 5.70, 3.04, 1.66, 8.26],
             [9.66, 5.53, 9.81, 8.96, 8.84, 2.23, 5.42, 6.41, 2.19, 8.69, 8.41, 9.63, 6.26, 2.77, 6.88],
             [5.68, 1.47, 3.30, 1.51, 0.97, 2.82, 6.85, 1.01, 0.86, 8.32, 0.83, 5.52, 4.61, 5.95, 9.56],
             [5.63, 5.17, 5.04, 2.77, 3.22, 8.44, 1.05, 2.06, 4.69, 0.16, 1.29, 2.02, 2.82, 1.16, 7.46],
             [4.39, 9.87, 2.19, 6.36, 6.88, 5.44, 7.18, 0.16, 6.62, 1.88, 6.46, 7.27, 7.59, 6.13, 5.34],
             [4.96, 8.46, 0.26, 7.01, 9.36, 0.61, 8.21, 1.82, 9.81, 1.28, 3.41, 3.20, 1.40, 8.20, 0.95],
             [6.36, 4.40, 7.43, 7.41, 8.58, 2.95, 5.04, 5.48, 1.33, 7.79, 1.53, 7.30, 2.02, 9.20, 3.19],
             [5.81, 5.85, 2.30, 5.09, 9.57, 8.08, 2.75, 2.99, 2.70, 5.14, 0.29, 7.95, 9.90, 9.71, 5.84]],

            [[7.93, 0.35, 1.02, 1.19, 6.96, 4.55, 1.85, 0.70, 6.80, 8.49, 0.54, 5.47, 7.57, 3.03, 5.93],
             [2.84, 9.46, 9.19, 4.49, 0.42, 2.26, 7.95, 4.36, 1.18, 9.94, 7.37, 4.88, 8.04, 6.01, 1.26],
             [2.73, 4.03, 2.50, 9.90, 3.51, 9.53, 1.03, 2.01, 0.71, 3.09, 1.24, 1.97, 4.78, 8.69, 4.2],
             [3.45, 4.46, 0.19, 0.54, 5.21, 9.66, 2.28, 5.84, 9.80, 9.72, 6.16, 5.17, 0.51, 8.07, 9.98],
             [5.82, 5.66, 0.02, 2.28, 8.50, 0.06, 5.98, 1.91, 3.69, 0.98, 7.10, 5.13, 6.22, 3.20, 1.05],
             [4.71, 2.63, 0.47, 6.70, 3.99, 5.36, 3.19, 3.50, 6.79, 7.85, 6.38, 2.26, 6.00, 5.31, 6.69],
             [9.67, 8.39, 7.69, 9.63, 1.80, 7.40, 0.53, 3.41, 4.18, 5.06, 6.33, 8.79, 4.18, 5.56, 4.18],
             [8.34, 8.47, 9.03, 9.66, 0.29, 6.72, 9.32, 0.56, 2.08, 6.74, 6.67, 4.30, 8.74, 8.59, 7.73],
             [3.83, 5.72, 4.78, 9.77, 8.38, 6.96, 0.90, 5.01, 9.55, 2.26, 3.10, 3.78, 2.83, 3.21, 1.7],
             [1.41, 7.79, 8.91, 0.13, 4.94, 9.35, 8.13, 4.78, 0.15, 4.76, 2.10, 7.92, 8.54, 2.19, 2.71]]
        ])

        self.m = 5

    def mock_initialise(self, mmp):
        mock_series = np.zeros((3, self.dist_matrix.shape[2] + self.m - 1))
        mock_query = np.zeros((3, self.dist_matrix.shape[1] + self.m - 1))
        mmp.initialise(mock_series, mock_query, self.m)

    def bruteforce_mmp(self, dist_matrix):
        """
        Brute force calculation of multidimensional distance matrix.
        :param dist_matrix: 2D matrix
        :return: a tuple (profile, index, dimensions)
        """

        correct = np.full((dist_matrix.shape[0], dist_matrix.shape[2]), np.inf, dtype=np.float)
        correct_index = np.full((dist_matrix.shape[0], dist_matrix.shape[2]), -1, dtype=np.int)
        correct_dims = [np.full((i + 1, dist_matrix.shape[2]), -1, dtype=np.int) for i in range(dist_matrix.shape[0])]

        correct[0], correct_index[0], correct_dims[0] = self._bruteforce_mmp_for_dims(dist_matrix, ([0], [1], [2]))
        correct[1], correct_index[1], correct_dims[1] = self._bruteforce_mmp_for_dims(dist_matrix, ([0, 1], [0, 2], [1, 2]))
        correct[2], correct_index[2], correct_dims[2] = self._bruteforce_mmp_for_dims(dist_matrix, ([0, 1, 2],))

        return correct, correct_index, correct_dims

    def _bruteforce_mmp_for_dims(self, dist_matrix, dimensions):
        mp = np.full(dist_matrix.shape[2], np.inf, dtype=np.float)
        index = np.full(dist_matrix.shape[2], -1, dtype=np.int)
        dims = np.full((len(dimensions[0]), dist_matrix.shape[2]), -1, dtype=np.int)

        for selected_dims in dimensions:
            dim_view = np.sum(dist_matrix[selected_dims], axis=0) / len(selected_dims)
            dim_mp = np.min(dim_view, axis=0)
            dim_mp_better = dim_mp < mp
            mp[dim_mp_better] = dim_mp[dim_mp_better]
            index[dim_mp_better] = np.argmin(dim_view, axis=0)[dim_mp_better]
            dims[:, dim_mp_better] = np.asarray(selected_dims)[:, None]

        return mp, index, dims

    def test_process_diagonal(self):
        mmp = MultidimensionalMatrixProfileLR()
        self.mock_initialise(mmp)

        for diag in range(-self.dist_matrix.shape[1] + 1, self.dist_matrix.shape[2]):
            diag_ind = diag_indices_of(self.dist_matrix[0], diag)
            mmp.process_diagonal(diag, self.dist_matrix[:, diag_ind[0], diag_ind[1]])

        correct_mp, correct_index, correct_dims = self.bruteforce_mmp(self.dist_matrix)

        npt.assert_allclose(mmp.md_matrix_profile(), correct_mp)
        npt.assert_equal(mmp.md_profile_index(), correct_index)
        for i, dims in enumerate(correct_dims):
            sorted_mmp_dims = np.sort(mmp.md_profile_dimensions()[i], axis=0)
            npt.assert_equal(sorted_mmp_dims, dims)

    def test_process_diagonal_partial_calculation(self):
        part_dist_matrix = np.full_like(self.dist_matrix, np.inf)

        mmp = MultidimensionalMatrixProfileLR()
        self.mock_initialise(mmp)

        for diag in range(-8, self.dist_matrix.shape[1], 4):
            diag_ind = diag_indices_of(self.dist_matrix[0], diag)
            mmp.process_diagonal(diag, self.dist_matrix[:, diag_ind[0], diag_ind[1]])
            part_dist_matrix[:, diag_ind[0], diag_ind[1]] = self.dist_matrix[:, diag_ind[0], diag_ind[1]]

        correct_mp, correct_index, correct_dims = self.bruteforce_mmp(part_dist_matrix)

        npt.assert_allclose(mmp.md_matrix_profile(), correct_mp)
        npt.assert_equal(mmp.md_profile_index(), correct_index)
        for i, dims in enumerate(correct_dims):
            sorted_mmp_dims = np.sort(mmp.md_profile_dimensions()[i], axis=0)
            npt.assert_equal(sorted_mmp_dims, dims)

    def test_process_diagonal_lr(self):
        l_dist_matrix = self.dist_matrix.copy()
        r_dist_matrix = self.dist_matrix.copy()

        mmp = MultidimensionalMatrixProfileLR()
        self.mock_initialise(mmp)

        for diag in range(-self.dist_matrix.shape[1] + 1, self.dist_matrix.shape[2]):
            diag_ind = diag_indices_of(self.dist_matrix[0], diag)
            if diag < 0:
                l_dist_matrix[:, diag_ind[0], diag_ind[1]] = np.inf
            else:
                r_dist_matrix[:, diag_ind[0], diag_ind[1]] = np.inf
            mmp.process_diagonal(diag, self.dist_matrix[:, diag_ind[0], diag_ind[1]])

        correct_lmp, correct_lindex, correct_ldims = self.bruteforce_mmp(l_dist_matrix)
        correct_rmp, correct_rindex, correct_rdims = self.bruteforce_mmp(r_dist_matrix)

        npt.assert_allclose(mmp.md_matrix_profile_left, correct_lmp)
        npt.assert_allclose(mmp.md_matrix_profile_right, correct_rmp)
        npt.assert_equal(mmp.md_profile_index_left, correct_lindex)
        npt.assert_equal(mmp.md_profile_index_right, correct_rindex)
        for i, dims in enumerate(correct_ldims):
            sorted_mmp_ldims = np.sort(mmp.md_profile_dimension_left[i], axis=0)
            sorted_mmp_rdims = np.sort(mmp.md_profile_dimension_right[i], axis=0)
            npt.assert_equal(sorted_mmp_ldims, correct_ldims[i])
            npt.assert_equal(sorted_mmp_rdims, correct_rdims[i])

    def test_process_column(self):
        mmp = MultidimensionalMatrixProfileLR()
        self.mock_initialise(mmp)

        for column in range(0, self.dist_matrix.shape[2]):
            mmp.process_column(column, self.dist_matrix[:, :, column])

        correct_mp, correct_index, correct_dims = self.bruteforce_mmp(self.dist_matrix)

        npt.assert_allclose(mmp.md_matrix_profile(), correct_mp)
        npt.assert_equal(mmp.md_profile_index(), correct_index)
        for i, dims in enumerate(correct_dims):
            sorted_mmp_dims = np.sort(mmp.md_profile_dimensions()[i], axis=0)
            npt.assert_equal(sorted_mmp_dims, dims)

    def test_process_column_partial_calculation(self):
        part_dist_matrix = np.full_like(self.dist_matrix, np.inf)

        mmp = MultidimensionalMatrixProfileLR()
        self.mock_initialise(mmp)

        for column in [0, 3, 4, 6, 9, 12, 13]:
            mmp.process_column(column, self.dist_matrix[:, :, column])
            part_dist_matrix[:, :, column] = self.dist_matrix[:, :, column]

        correct_mp, correct_index, correct_dims = self.bruteforce_mmp(part_dist_matrix)

        npt.assert_allclose(mmp.md_matrix_profile(), correct_mp)
        npt.assert_equal(mmp.md_profile_index(), correct_index)
        for i, dims in enumerate(correct_dims):
            sorted_mmp_dims = np.sort(mmp.md_profile_dimensions()[i], axis=0)
            npt.assert_equal(sorted_mmp_dims, dims)

    def test_process_column_lr(self):
        l_dist_matrix = self.dist_matrix.copy()
        r_dist_matrix = self.dist_matrix.copy()

        mmp = MultidimensionalMatrixProfileLR()
        self.mock_initialise(mmp)

        for diag in range(-self.dist_matrix.shape[1] + 1, self.dist_matrix.shape[2]):
            diag_ind = diag_indices_of(self.dist_matrix[0], diag)
            if diag < 0:
                l_dist_matrix[:, diag_ind[0], diag_ind[1]] = np.inf
            else:
                r_dist_matrix[:, diag_ind[0], diag_ind[1]] = np.inf

        for column in range(self.dist_matrix.shape[2]):
            mmp.process_column(column, self.dist_matrix[:, :, column])

        correct_lmp, correct_lindex, correct_ldims = self.bruteforce_mmp(l_dist_matrix)
        correct_rmp, correct_rindex, correct_rdims = self.bruteforce_mmp(r_dist_matrix)

        npt.assert_allclose(mmp.md_matrix_profile_left, correct_lmp)
        npt.assert_allclose(mmp.md_matrix_profile_right, correct_rmp)
        npt.assert_equal(mmp.md_profile_index_left, correct_lindex)
        npt.assert_equal(mmp.md_profile_index_right, correct_rindex)
        for i, dims in enumerate(correct_ldims):
            sorted_mmp_ldims = np.sort(mmp.md_profile_dimension_left[i], axis=0)
            sorted_mmp_rdims = np.sort(mmp.md_profile_dimension_right[i], axis=0)
            npt.assert_equal(sorted_mmp_ldims, correct_ldims[i])
            npt.assert_equal(sorted_mmp_rdims, correct_rdims[i])
