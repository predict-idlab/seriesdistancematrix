import numpy as np
from numpy import nan
from unittest import TestCase
import numpy.testing as npt

from distancematrix.util import diag_indices_of
from distancematrix.util import diag_indices
from distancematrix.consumer.matrix_profile_lr import MatrixProfileLR
from distancematrix.consumer.matrix_profile_lr import ShiftingMatrixProfileLR
from distancematrix.consumer.matrix_profile_lr import MatrixProfileLRReservoir


def brute_force_mp(dist_matrix):
    h, w = dist_matrix.shape

    mp = np.full(w, np.inf)
    mpi = np.full(w, -1)

    for col in range(w):
        mp[col] = np.nanmin(dist_matrix[:, col])
        if np.isfinite(mp[col]):
            mpi[col] = np.nanargmin(dist_matrix[:, col])

    return mp, mpi


def fill_diagonals(matrix, diagonals, value):
    result = matrix.copy()
    for diag in diagonals:
        result[diag_indices_of(result, diag)] = value
    return result


class TestMatrixProfileLR(TestCase):
    def setUp(self):
        self.dm = np.array([
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

        self.mplr = MatrixProfileLR()
        self.mplr.initialise(1, self.dm.shape[0], self.dm.shape[1])

    def test_process_diagonal(self):
        for diag in range(-self.dm.shape[0] + 1, self.dm.shape[1]):
            diag_ind = diag_indices_of(self.dm, diag)
            self.mplr.process_diagonal(diag, np.atleast_2d(self.dm[diag_ind]))

        rmp, rmpi = brute_force_mp(fill_diagonals(self.dm, range(0, self.dm.shape[1]), np.inf))
        lmp, lmpi = brute_force_mp(fill_diagonals(self.dm, range(-self.dm.shape[1]+1, 0), np.inf))
        mp, mpi = brute_force_mp(self.dm)

        npt.assert_allclose(self.mplr.matrix_profile_right, rmp)
        npt.assert_allclose(self.mplr.matrix_profile_left, lmp)
        npt.assert_equal(self.mplr.profile_index_right, rmpi)
        npt.assert_equal(self.mplr.profile_index_left, lmpi)

        npt.assert_allclose(self.mplr.matrix_profile(), mp)
        npt.assert_equal(self.mplr.profile_index(), mpi)

    def test_process_column(self):
        for column in range(0, self.dm.shape[1]):
            self.mplr.process_column(column, np.atleast_2d(self.dm[:, column]))

        rmp, rmpi = brute_force_mp(fill_diagonals(self.dm, range(0, self.dm.shape[1]), np.inf))
        lmp, lmpi = brute_force_mp(fill_diagonals(self.dm, range(-self.dm.shape[1] + 1, 0), np.inf))
        mp, mpi = brute_force_mp(self.dm)

        npt.assert_allclose(self.mplr.matrix_profile_right, rmp)
        npt.assert_allclose(self.mplr.matrix_profile_left, lmp)
        npt.assert_equal(self.mplr.profile_index_right, rmpi)
        npt.assert_equal(self.mplr.profile_index_left, lmpi)

        npt.assert_allclose(self.mplr.matrix_profile(), mp)
        npt.assert_equal(self.mplr.profile_index(), mpi)


class TestMatrixProfileLRReservoir(TestMatrixProfileLR):
    # Also runs tests in base class!
    def setUp(self):
        super().setUp()

        self.mplr = MatrixProfileLRReservoir(random_seed=0)
        self.mplr.initialise(1, self.dm.shape[0], self.dm.shape[1])

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


class TestShiftingMatrixProfileLR(TestMatrixProfileLR):
    def setUp(self):
        # Needed to run all tests in TestMatrixProfileLR
        super().setUp()
        self.mplr = ShiftingMatrixProfileLR()
        self.mplr.initialise(1, self.dm.shape[0], self.dm.shape[1])

    def test_calculate_columns_with_series_shifting(self):
        dm = np.array([
            [2., 2., 3., 1., -1., -5],
            [4., 1., 2., 4., -2., -6.],
            [1., 3., 4., 2., -3., -7.],
            [3., 4., 1., 3., -4., 0.]])
        #   =============== : window

        mplr = ShiftingMatrixProfileLR()
        mplr.initialise(1, 4, 4)

        mplr.process_column(0, np.atleast_2d(dm[:, 0]))
        mplr.process_column(1, np.atleast_2d(dm[:, 1]))

        npt.assert_equal(mplr.matrix_profile_left, [2., 1, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_left, [0, 1, -1, -1])
        npt.assert_equal(mplr.matrix_profile_right, [1., 3, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_right, [2, 2, -1, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., np.inf, np.inf])
        npt.assert_equal(mplr.profile_index(), [2, 1, -1, -1])

        mplr.shift_series(1)
        npt.assert_equal(mplr.matrix_profile_left, [1, np.inf, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_left, [1, -1, -1, -1])
        npt.assert_equal(mplr.matrix_profile_right, [3, np.inf, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_right, [2, -1, -1, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., np.inf, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index(), [1, -1, -1, -1])

        mplr.process_column(1, np.atleast_2d(dm[:, 2]))
        npt.assert_equal(mplr.matrix_profile_left, [1, 2, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_left, [1, 1, -1, -1])
        npt.assert_equal(mplr.matrix_profile_right, [3, 1, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_right, [2, 3, -1, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index(), [1, 3, -1, -1])

        mplr.shift_series(1)
        mplr.process_column(1, np.atleast_2d(dm[:, 3]))
        npt.assert_equal(mplr.matrix_profile_left, [2., 1, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_left, [1, 0, -1, -1])
        npt.assert_equal(mplr.matrix_profile_right, [1., np.inf, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_right, [3, -1, -1, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., np.inf, np.inf])
        npt.assert_equal(mplr.profile_index(), [3, 0, -1, -1])

        mplr.process_column(2, np.atleast_2d(dm[:, 4]))
        mplr.process_column(3, np.atleast_2d(dm[:, 5]))

        npt.assert_equal(mplr.matrix_profile_left, [2., 1, -4, -7])
        npt.assert_equal(mplr.profile_index_left, [1, 0, 3, 2])
        npt.assert_equal(mplr.matrix_profile_right, [1., np.inf, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_right, [3, -1, -1, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., -4, -7])
        npt.assert_equal(mplr.profile_index(), [3, 0, 3, 2])

    def test_calculate_columns_with_series_and_query_shifting(self):
        dm = np.array([
            [2.,  2.,  3.,  1.,  nan, nan],
            [4.,  1.,  2.,  4.,  nan, nan],
            [1.,  3.,  4.,  2.,  -1., -5.],
            [3.,  4.,  1.,  3.,  -2., -6.],
            [nan, nan, nan, 8., -3., -7.],
            [nan, nan, nan, 7., -4., 0.]
        ])

        mplr = ShiftingMatrixProfileLR()
        mplr.initialise(1, 4, 4)

        for col in range(4):
            mplr.process_column(col, np.atleast_2d(dm[0:4, col]))

        npt.assert_equal(mplr.matrix_profile_left, [2., 1, 2., 1])
        npt.assert_equal(mplr.profile_index_left, [0, 1, 1, 0])
        npt.assert_equal(mplr.matrix_profile_right, [1., 3, 1., np.inf])
        npt.assert_equal(mplr.profile_index_right, [2, 2, 3, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., 1., 1.])
        npt.assert_equal(mplr.profile_index(), [2, 1, 3, 0])

        mplr.shift_query(2)
        mplr.shift_series(2)
        npt.assert_equal(mplr.matrix_profile_left, [2., 1, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_left, [1, 0, -1, -1])
        npt.assert_equal(mplr.matrix_profile_right, [1., np.inf, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_right, [3, -1, -1, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., np.inf, np.inf])
        npt.assert_equal(mplr.profile_index(), [3, 0, -1, -1])

        mplr.process_column(1, np.atleast_2d(dm[2:6, 3]))
        mplr.process_column(2, np.atleast_2d(dm[2:6, 4]))
        mplr.process_column(3, np.atleast_2d(dm[2:6, 5]))

        npt.assert_equal(mplr.matrix_profile_left, [2., 1, -3, -7])
        npt.assert_equal(mplr.profile_index_left, [1, 0, 4, 4])
        npt.assert_equal(mplr.matrix_profile_right, [1., 7., -4, np.inf])
        npt.assert_equal(mplr.profile_index_right, [3, 5, 5, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., -4, -7])
        npt.assert_equal(mplr.profile_index(), [3, 0, 5, 4])

    def test_calculate_columns_with_query_shifting(self):
        dm = np.array([
            [2., 1.],
            [0., 2.],
            [1., 3.],
            [3., 4.],
            [4., 0.],
            [2., 9.]
        ])

        mplr = ShiftingMatrixProfileLR()
        mplr.initialise(1, 4, 2)

        mplr.process_column(0, np.atleast_2d(dm[0:4, 0]))
        mplr.process_column(1, np.atleast_2d(dm[0:4, 1]))

        npt.assert_equal(mplr.matrix_profile_left, [2., 1.])
        npt.assert_equal(mplr.profile_index_left, [0, 0])
        npt.assert_equal(mplr.matrix_profile_right, [0., 3.])
        npt.assert_equal(mplr.profile_index_right, [1, 2])
        npt.assert_equal(mplr.matrix_profile(), [0., 1.])
        npt.assert_equal(mplr.profile_index(), [1, 0])

        mplr.shift_query(1)
        npt.assert_equal(mplr.matrix_profile_left, [2., 1.])
        npt.assert_equal(mplr.profile_index_left, [0, 0])
        npt.assert_equal(mplr.matrix_profile_right, [0., 3.])
        npt.assert_equal(mplr.profile_index_right, [1, 2])
        npt.assert_equal(mplr.matrix_profile(), [0., 1.])
        npt.assert_equal(mplr.profile_index(), [1, 0])

        mplr.process_column(0, np.atleast_2d(dm[1:5, 0]))
        mplr.process_column(1, np.atleast_2d(dm[1:5, 1]))
        npt.assert_equal(mplr.matrix_profile_left, [2., 1.])
        npt.assert_equal(mplr.profile_index_left, [0, 0])
        npt.assert_equal(mplr.matrix_profile_right, [0., 0.])
        npt.assert_equal(mplr.profile_index_right, [1, 4])
        npt.assert_equal(mplr.matrix_profile(), [0., 0.])
        npt.assert_equal(mplr.profile_index(), [1, 4])

        mplr.shift_query(1)
        mplr.process_column(0, np.atleast_2d(dm[2:6, 0]))
        mplr.process_column(1, np.atleast_2d(dm[2:6, 1]))
        npt.assert_equal(mplr.matrix_profile_left, [2., 1.])
        npt.assert_equal(mplr.profile_index_left, [0, 0])
        npt.assert_equal(mplr.matrix_profile_right, [0., 0.])
        npt.assert_equal(mplr.profile_index_right, [1, 4])
        npt.assert_equal(mplr.matrix_profile(), [0., 0.])
        npt.assert_equal(mplr.profile_index(), [1, 4])

    def test_calculate_diagonals_with_series_shifting(self):
        dm = np.array([
            [2., 2., 3., 1., -1., -5],
            [4., 1., 2., 4., -2., -6.],
            [1., 3., 4., 2., -3., -7.],
            [3., 4., 1., 3., -4., 0.]])
        #   =============== : window

        mplr = ShiftingMatrixProfileLR()
        mplr.initialise(1, 4, 4)

        for diag in range(-3, 0):
            mplr.process_diagonal(diag, np.atleast_2d(dm[:, :4][diag_indices(4, 4, diag)]))
        for diag in range(1, 4):
            mplr.process_diagonal(diag, np.atleast_2d(dm[:, :4][diag_indices(4, 4, diag)]))

        npt.assert_equal(mplr.matrix_profile_left, [np.inf, 2, 2., 1])
        npt.assert_equal(mplr.profile_index_left, [-1, 0, 1, 0])
        npt.assert_equal(mplr.matrix_profile_right, [1., 3, 1., np.inf])
        npt.assert_equal(mplr.profile_index_right, [2, 2, 3, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 2., 1., 1.])
        npt.assert_equal(mplr.profile_index(), [2, 0, 3, 0])

        mplr.shift_series(1)
        mplr.process_diagonal(-1, np.atleast_2d(dm[:, 1:5][diag_indices(4, 4, -1)]))
        npt.assert_equal(mplr.matrix_profile_left, [1., 2., 1, np.inf])
        npt.assert_equal(mplr.profile_index_left, [1, 1, 0, -1])
        npt.assert_equal(mplr.matrix_profile_right, [3, 1., np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_right, [2, 3, -1, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., 1., np.inf])
        npt.assert_equal(mplr.profile_index(), [1, 3, 0, -1])

        mplr.shift_series(1)
        npt.assert_equal(mplr.matrix_profile_left, [2., 1, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_left, [1, 0, -1, -1])
        npt.assert_equal(mplr.matrix_profile_right, [1., np.inf, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_right, [3, -1, -1, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., np.inf, np.inf])
        npt.assert_equal(mplr.profile_index(), [3, 0, -1, -1])

        for diag in range(-1, 4):
            mplr.process_diagonal(diag, np.atleast_2d(dm[:, 2:][diag_indices(4, 4, diag)]))

        npt.assert_equal(mplr.matrix_profile_left, [2., 1, -4, -7])
        npt.assert_equal(mplr.profile_index_left, [1, 0, 3, 2])
        npt.assert_equal(mplr.matrix_profile_right, [1., np.inf, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_right, [3, -1, -1, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., -4, -7])
        npt.assert_equal(mplr.profile_index(), [3, 0, 3, 2])

    def test_calculate_diagonals_with_series_and_query_shifting(self):
        dm = np.array([
            [2.,  2.,  3.,  1.,  nan, nan],
            [4.,  1.,  2.,  4.,  nan, nan],
            [1.,  3.,  4.,  2.,  -1., -5.],
            [3.,  4.,  1.,  3.,  -2., -6.],
            [nan, nan, nan, -1., -3., -7.],
            [nan, nan, nan, nan, -4., 0.]
        ])

        mplr = ShiftingMatrixProfileLR()
        mplr.initialise(1, 4, 4)

        for diag in range(-3, 4):
            mplr.process_diagonal(diag, np.atleast_2d(dm[:, :4][diag_indices(4, 4, diag)]))

        npt.assert_equal(mplr.matrix_profile_left, [2., 1, 2., 1])
        npt.assert_equal(mplr.profile_index_left, [0, 1, 1, 0])
        npt.assert_equal(mplr.matrix_profile_right, [1., 3, 1., np.inf])
        npt.assert_equal(mplr.profile_index_right, [2, 2, 3, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., 1., 1.])
        npt.assert_equal(mplr.profile_index(), [2, 1, 3, 0])

        mplr.shift_query(2)
        mplr.shift_series(2)
        npt.assert_equal(mplr.matrix_profile_left, [2., 1, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_left, [1, 0, -1, -1])
        npt.assert_equal(mplr.matrix_profile_right, [1., np.inf, np.inf, np.inf])
        npt.assert_equal(mplr.profile_index_right, [3, -1, -1, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., 1., np.inf, np.inf])
        npt.assert_equal(mplr.profile_index(), [3, 0, -1, -1])

        for diag in range(-1, 4):
            mplr.process_diagonal(diag, np.atleast_2d(dm[2:, 2:][diag_indices(4, 4, diag)]))

        npt.assert_equal(mplr.matrix_profile_left, [2., 1, -3, -7])
        npt.assert_equal(mplr.profile_index_left, [1, 0, 4, 4])
        npt.assert_equal(mplr.matrix_profile_right, [1., -1, -4, np.inf])
        npt.assert_equal(mplr.profile_index_right, [3, 4, 5, -1])
        npt.assert_equal(mplr.matrix_profile(), [1., -1., -4, -7])
        npt.assert_equal(mplr.profile_index(), [3, 4, 5, 4])
