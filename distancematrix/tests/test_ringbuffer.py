import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.ringbuffer import RingBuffer


class TestRingBuffer(TestCase):
    def test_one_dimensional(self):
        buffer = RingBuffer([0, 1, 2, 3, 4])
        npt.assert_equal(buffer.view, np.array([0, 1, 2, 3, 4]))

        buffer.push([])
        npt.assert_equal(buffer.view, np.array([0, 1, 2, 3, 4]))
        self.assertEqual(buffer[0], 0)

        buffer.push([5])
        npt.assert_equal(buffer.view, np.array([1, 2, 3, 4, 5]))
        self.assertEqual(buffer[0], 1)

        buffer.push([6, 7])
        npt.assert_equal(buffer.view, np.array([3, 4, 5, 6, 7]))
        self.assertEqual(buffer[0], 3)

        buffer.push([8, 9, 10])
        npt.assert_equal(buffer.view, np.array([6, 7, 8, 9, 10]))
        self.assertEqual(buffer[0], 6)

        buffer.push([11, 12, 13, 14])
        npt.assert_equal(buffer.view, np.array([10, 11, 12, 13, 14]))
        self.assertEqual(buffer[0], 10)

        buffer.push([15, 16, 17, 18, 19])
        npt.assert_equal(buffer.view, np.array([15, 16, 17, 18, 19]))
        self.assertEqual(buffer[0], 15)

        buffer.push([20, 21, 22, 23, 24, 25])
        npt.assert_equal(buffer.view, np.array([21, 22, 23, 24, 25]))
        self.assertEqual(buffer[0], 21)

    def test_multi_dimensional(self):
        buffer = RingBuffer([[0, 1, 2, 3, 4], [0, -1, -2, -3, -4]])
        npt.assert_equal(buffer.view, np.array([[0, 1, 2, 3, 4], [0, -1, -2, -3, -4]]))

        buffer.push([[], []])
        npt.assert_equal(buffer.view, np.array([[0, 1, 2, 3, 4], [0, -1, -2, -3, -4]]))
        npt.assert_equal(buffer[:, 0], [0, 0])

        buffer.push([[5], [-5]])
        npt.assert_equal(buffer.view, np.array([[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]]))
        npt.assert_equal(buffer[:, 0], [1, -1])

        buffer.push([[6, 7], [-6, -7]])
        npt.assert_equal(buffer.view, np.array([[3, 4, 5, 6, 7], [-3, -4, -5, -6, -7]]))
        npt.assert_equal(buffer[:, 0], [3, -3])

        buffer.push([[8, 9, 10], [-8, -9, -10]])
        npt.assert_equal(buffer.view, np.array([[6, 7, 8, 9, 10], [-6, -7, -8, -9, -10]]))
        npt.assert_equal(buffer[:, 0], [6, -6])

        buffer.push([[11, 12, 13, 14], [-11, -12, -13, -14]])
        npt.assert_equal(buffer.view, np.array([[10, 11, 12, 13, 14], [-10, -11, -12, -13, -14]]))
        npt.assert_equal(buffer[:, 0], [10, -10])

        buffer.push([[15, 16, 17, 18, 19], [-15, -16, -17, -18, -19]])
        npt.assert_equal(buffer.view, np.array([[15, 16, 17, 18, 19], [-15, -16, -17, -18, -19]]))
        npt.assert_equal(buffer[:, 0], [15, -15])

        buffer.push([[20, 21, 22, 23, 24, 25], [-20, -21, -22, -23, -24, -25]])
        npt.assert_equal(buffer.view, np.array([[21, 22, 23, 24, 25], [-21, -22, -23, -24, -25]]))
        npt.assert_equal(buffer[:, 0], [21, -21])

    def test_empty_intialization(self):
        buffer = RingBuffer(None, shape=(5,), dtype=np.int)

        npt.assert_equal(buffer.view, np.array([]))

        buffer.push([1])
        npt.assert_equal(buffer.view, np.array([1]))
        self.assertEqual(buffer[0], 1)

        buffer.push([2, 3])
        npt.assert_equal(buffer.view, np.array([1, 2, 3]))
        self.assertEqual(buffer[0], 1)

        buffer.push([4, 5, 6])
        npt.assert_equal(buffer.view, np.array([2, 3, 4, 5, 6]))
        self.assertEqual(buffer[0], 2)

    def test_partial_intialization(self):
        buffer = RingBuffer([1, 2], shape=(5,), dtype=np.int)

        npt.assert_equal(buffer.view, np.array([1, 2]))
        self.assertEqual(buffer[0], 1)

        buffer.push([3])
        npt.assert_equal(buffer.view, np.array([1, 2, 3]))
        self.assertEqual(buffer[0], 1)

        buffer.push([4, 5, 6])
        npt.assert_equal(buffer.view, np.array([2, 3, 4, 5, 6]))
        self.assertEqual(buffer[0], 2)

    def test_oversized_initialization(self):
        buffer = RingBuffer([1, 2, 3, 4, 5, 6], shape=(5,), dtype=np.int)

        npt.assert_equal(buffer.view, np.array([2, 3, 4, 5, 6]))
        self.assertEqual(buffer[0], 2)

        buffer.push([7])
        npt.assert_equal(buffer.view, np.array([3, 4, 5, 6, 7]))
        self.assertEqual(buffer[0], 3)

        buffer.push([8, 9, 10])
        npt.assert_equal(buffer.view, np.array([6, 7, 8, 9, 10]))
        self.assertEqual(buffer[0], 6)
