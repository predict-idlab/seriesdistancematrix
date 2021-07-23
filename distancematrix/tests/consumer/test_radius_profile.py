import numpy as np
from unittest import TestCase
import numpy.testing as npt
from itertools import takewhile

from distancematrix.consumer.radius_profile import RadiusProfile0
from distancematrix.consumer.radius_profile import RadiusProfile
from distancematrix.insights import lowest_value_idxs


class TestRadiusProfile0(TestCase):
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


    @staticmethod
    def bruteforce_calc(array, target_idxs):
        target_idxs = sorted(target_idxs)
        result = np.zeros((len(target_idxs), array.shape[1]), dtype=float)

        for col in range(array.shape[1]):
            sorted_col_values = np.sort(array[:, col])
            for i, target_idx in enumerate(target_idxs):
                if target_idx < len(sorted_col_values):
                    result[i, col] = sorted_col_values[target_idx]
                else:
                    result[i, col] = np.nan

        return result

    def test_process_diagonal(self):
        tracker = RadiusProfile0(0)
        tracker.initialise(1, self.dm.shape[0], self.dm.shape[1])

        with self.assertRaises(NotImplementedError):
            tracker.process_diagonal(0, np.zeros(10))

    def test_process_column_single_value(self):
        tracker = RadiusProfile0(2)
        tracker.initialise(1, self.dm.shape[0], self.dm.shape[1])

        for column in range(0, self.dm.shape[1]):
            tracker.process_column(column, np.atleast_2d(self.dm[:, column]))

        npt.assert_equal(tracker.values, self.bruteforce_calc(self.dm, [2]))

    def test_process_column_multiple_value(self):
        track_idxs = [2, 5, 0, 9, len(self.dm)]
        tracker = RadiusProfile0(track_idxs)
        tracker.initialise(1, self.dm.shape[0], self.dm.shape[1])

        for column in range(0, self.dm.shape[1]):
            tracker.process_column(column, np.atleast_2d(self.dm[:, column]))

        npt.assert_equal(tracker.values, self.bruteforce_calc(self.dm, track_idxs))
        npt.assert_equal(tracker.values[-1, :], np.full(self.dm.shape[1], np.nan))


class TestRadiusProfile(TestCase):
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

    @staticmethod
    def bruteforce_calc(array, target_idxs, exclusion):
        target_idxs = np.array(target_idxs)
        target_idxs.sort()

        result = np.zeros((len(target_idxs), array.shape[1]), dtype=float)

        for col in range(array.shape[1]):
            for i, target_idx in enumerate(target_idxs):
                lowest_idxs = list(lowest_value_idxs(array[:, col], exclusion))  # takes care of the exclusion distance
                lowest_values = array[:, col][lowest_idxs]

                if target_idx < len(lowest_values):
                    result[i, col] = lowest_values[target_idx]
                else:
                    result[i, col] = np.nan

        return result

    def test_process_diagonal(self):
        tracker = RadiusProfile(2, 2)
        tracker.initialise(1, self.dm.shape[0], self.dm.shape[1])

        with self.assertRaises(NotImplementedError):
            tracker.process_diagonal(0, np.zeros(10))

    def test_process_column_single_value(self):
        tracker = RadiusProfile(2, 2)
        tracker.initialise(1, self.dm.shape[0], self.dm.shape[1])

        for column in range(0, self.dm.shape[1]):
            tracker.process_column(column, np.atleast_2d(self.dm[:, column]))

        npt.assert_equal(tracker.values, self.bruteforce_calc(self.dm, [2], 2))

    def test_process_column_multiple_value(self):
        track_idxs = [2, 0, len(self.dm)]
        tracker = RadiusProfile(track_idxs, 1)
        tracker.initialise(1, self.dm.shape[0], self.dm.shape[1])

        for column in range(0, self.dm.shape[1]):
            tracker.process_column(column, np.atleast_2d(self.dm[:, column]))

        npt.assert_equal(tracker.values, self.bruteforce_calc(self.dm, track_idxs, 1))
