import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.util import diag_indices_of
from distancematrix.util import diag_indices
from distancematrix.consumer.matrix_profile_lr import MatrixProfileLR
from distancematrix.consumer.matrix_profile_lr import MatrixProfileLRReservoir


class TestMatrixProfileLR(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dist_matrix = np.array([
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

        h, w = cls.dist_matrix.shape

        dm_left = cls.dist_matrix.copy()
        for diag in range(-1, -h, -1):
            dm_left[diag_indices(h, w, diag)] = np.inf
        dm_right = cls.dist_matrix.copy()
        for diag in range(0, w):
            dm_right[diag_indices(h, w, diag)] = np.inf

        left_mp = np.full(w, np.inf, dtype=np.float)
        right_mp = np.full(w, np.inf, dtype=np.float)
        left_indices = np.full(w, -1, dtype=np.int)
        right_indices = np.full(w, -1, dtype=np.int)
        mp = np.full(w, np.inf, dtype=np.float)
        indices = np.full(w, -1, dtype=np.int)

        for i in range(w):
            left_mp[i] = np.min(dm_left[:, i])
            if np.isfinite(left_mp[i]):
                left_indices[i] = np.argmin(dm_left[:, i])
            right_mp[i] = np.min(dm_right[:, i])
            if np.isfinite(right_mp[i]):
                right_indices[i] = np.argmin(dm_right[:, i])
            mp[i] = np.min(cls.dist_matrix[:, i])
            if np.isfinite(mp[i]):
                indices[i] = np.argmin(cls.dist_matrix[:, i])

        cls.correct_mp = mp
        cls.correct_indices = indices
        cls.correct_mp_left = left_mp
        cls.correct_indices_left = left_indices
        cls.correct_mp_right = right_mp
        cls.correct_indices_right = right_indices

    def setUp(self):
        m = 5
        dm = TestMatrixProfileLR.dist_matrix

        self.mplr = MatrixProfileLR()
        self.mplr.initialise(1, dm.shape[0], dm.shape[1])

    def test_process_diagonal(self):
        dm = TestMatrixProfileLR.dist_matrix

        for diag in range(-dm.shape[0] + 1, dm.shape[1]):
            diag_ind = diag_indices_of(dm, diag)
            self.mplr.process_diagonal(diag, np.atleast_2d(dm[diag_ind]))

        npt.assert_allclose(self.mplr.matrix_profile_right, TestMatrixProfileLR.correct_mp_right)
        npt.assert_allclose(self.mplr.matrix_profile_left, TestMatrixProfileLR.correct_mp_left)
        npt.assert_equal(self.mplr.profile_index_right, TestMatrixProfileLR.correct_indices_right)
        npt.assert_equal(self.mplr.profile_index_left, TestMatrixProfileLR.correct_indices_left)

        npt.assert_allclose(self.mplr.matrix_profile(), TestMatrixProfileLR.correct_mp)
        npt.assert_equal(self.mplr.profile_index(), TestMatrixProfileLR.correct_indices)

    def test_process_column(self):
        dm = TestMatrixProfileLR.dist_matrix

        for column in range(0, dm.shape[1]):
            self.mplr.process_column(column, np.atleast_2d(dm[:, column]))

        npt.assert_allclose(self.mplr.matrix_profile_right, TestMatrixProfileLR.correct_mp_right)
        npt.assert_allclose(self.mplr.matrix_profile_left, TestMatrixProfileLR.correct_mp_left)
        npt.assert_equal(self.mplr.profile_index_right, TestMatrixProfileLR.correct_indices_right)
        npt.assert_equal(self.mplr.profile_index_left, TestMatrixProfileLR.correct_indices_left)

        npt.assert_allclose(self.mplr.matrix_profile(), TestMatrixProfileLR.correct_mp)
        npt.assert_equal(self.mplr.profile_index(), TestMatrixProfileLR.correct_indices)


class TestMatrixProfileLRReservoir(TestMatrixProfileLR):
    def setUp(self):
        super(TestMatrixProfileLR, self).setUp()

        dm = TestMatrixProfileLR.dist_matrix

        self.mplr = MatrixProfileLRReservoir(random_seed=0)
        self.mplr.initialise(1, dm.shape[0], dm.shape[1])

    @classmethod
    def setUpClass(cls):
        TestMatrixProfileLR.setUpClass()

    def test_reservoir_sampling_columns(self):
        dm = np.zeros((4, 1000))

        self.mplr = MatrixProfileLRReservoir(random_seed=0)
        self.mplr.initialise(1, dm.shape[0], dm.shape[1])

        for i in range(dm.shape[1]):
            self.mplr.process_column(i, np.atleast_2d(dm[:, i]))

        # Check correct value of matrix profile
        npt.assert_equal(self.mplr.matrix_profile(), np.zeros(dm.shape[1]))

        # Check uniform distribution of selected indices
        mp_index = self.mplr.profile_index()
        for i in range(dm.shape[0]):
            npt.assert_allclose(np.count_nonzero(mp_index == i), dm.shape[1] / dm.shape[0], rtol=0.1)

    def test_reservoir_sampling_diagonals(self):
        dm = np.zeros((4, 1000))

        self.mplr = MatrixProfileLRReservoir(random_seed=0)
        self.mplr.initialise(1, dm.shape[0], dm.shape[1])

        for i in range(-dm.shape[0] + 1, dm.shape[1]):
            self.mplr.process_diagonal(i, np.atleast_2d(dm[diag_indices_of(dm, i)]))

        # Check correct value of matrix profile
        npt.assert_equal(self.mplr.matrix_profile(), np.zeros(dm.shape[1]))

        # Check uniform distribution of selected indices
        mp_index = self.mplr.profile_index()
        for i in range(dm.shape[0]):
            npt.assert_allclose(np.count_nonzero(mp_index == i), dm.shape[1] / dm.shape[0], rtol=0.1)
