from unittest import TestCase
import numpy.testing as npt
from itertools import zip_longest

from distancematrix.consumer.contextmanager import GeneralStaticManager


class TestGeneralStaticManager(TestCase):
    def test_does_not_return_empty_contexts(self):
        r = [range(1, 5), range(0, 0), range(5, 10)]
        m = GeneralStaticManager(r)

        _assert_equal_iteration(m.series_contexts(0, 1), [])
        _assert_equal_iteration(m.series_contexts(0, 4), [(1, 5, 0)])
        _assert_equal_iteration(m.series_contexts(0, 8), [(1, 5, 0), (5, 10, 2)])
        _assert_equal_iteration(m.series_contexts(0, 12), [(1, 5, 0), (5, 10, 2)])
        _assert_equal_iteration(m.series_contexts(5, 12), [(5, 10, 2)])

        _assert_equal_iteration(m.query_contexts(0, 1), [])
        _assert_equal_iteration(m.query_contexts(0, 4), [(1, 5, 0)])
        _assert_equal_iteration(m.query_contexts(0, 8), [(1, 5, 0), (5, 10, 2)])
        _assert_equal_iteration(m.query_contexts(0, 12), [(1, 5, 0), (5, 10, 2)])
        _assert_equal_iteration(m.query_contexts(5, 12), [(5, 10, 2)])


def _assert_equal_iteration(actual, expected, msg=''):
    """
    Assert function similar to TestCase.assertSequenceEqual, but that actually treats 2D numpy arrays as iterables.
    """
    sentinel = object()
    for actual_value, expected_value in zip_longest(actual, expected, fillvalue=sentinel):
        if sentinel is actual_value:
            raise AssertionError("Actual iterator is shorter, does not include " + str(expected_value))

        if sentinel is expected_value:
            raise AssertionError("Actual iterator is longer, contained " + str(actual_value))

        npt.assert_equal(actual_value, expected_value, msg)
