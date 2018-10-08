import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.insights import lowest_value_idxs
from distancematrix.insights import highest_value_idxs


class TestSlidingMeanStd(TestCase):
    def test_lowest_value_idxs(self):
        a = np.array([1, 5, 3, 9, 4, 7, 6, 0, 2, 8], dtype=np.float)

        npt.assert_equal(list(lowest_value_idxs(a, 0)), np.argsort(a))
        npt.assert_equal(list(lowest_value_idxs(a, 1)), [7, 0, 2, 4, 9])
        npt.assert_equal(list(lowest_value_idxs(a, 2)), [7, 0, 4])
        npt.assert_equal(list(lowest_value_idxs(a, 3)), [7, 0])

    def test_highest_value_idxs(self):
        a = np.array([4, 8, 6, 1, 0, 3, 7, 9, 2, 5], dtype=np.float)

        npt.assert_equal(list(highest_value_idxs(a, 0)), np.argsort(a)[::-1])
        npt.assert_equal(list(highest_value_idxs(a, 1)), [7, 1, 9, 5, 3])
        npt.assert_equal(list(highest_value_idxs(a, 2)), [7, 1, 4])
        npt.assert_equal(list(highest_value_idxs(a, 3)), [7, 1])
