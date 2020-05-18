from abc import ABC, abstractmethod
from typing import Iterable, Tuple
import collections
import numpy as np


class AbstractContextManager(ABC):
    @abstractmethod
    def query_contexts(self, start: int, stop: int) -> Iterable[Tuple[int, int, int]]:
        """
        Return all non-empty query context definitions that fall in the given range of the distance matrix query axis.

        :param start: start of the range
        :param stop: end of the range
        :return: iterable of tuples (start of context, end of context, context id)
        """
        pass

    @abstractmethod
    def series_contexts(self, start: int, stop: int) -> Iterable[Tuple[int, int, int]]:
        """
        Return all non-empty series context definitions that fall in the given range of the distance matrix series axis.

        :param start: start of the range
        :param stop: end of the range
        :return: iterable of tuples (start of context, end of context, context id)
        """
        pass

    @abstractmethod
    def context_matrix_shape(self) -> (int, int):
        """
        Returns the shape of the contextual distance matrix

        :return: upper bound for any context id returned by this manager, for query and series axis
        """
        pass

    def shift_query(self, amount: int) -> int:
        """
        Informs the manager that the distance matrix has shifted along the query axis.

        :param amount: amount of values shifted
        :return: the amount of values that the contextual distance matrix should shift along the query axis
        """
        raise RuntimeError("This generator does not support query shifting.")

    def shift_series(self, amount: int) -> int:
        """
        Informs the manager that the distance matrix has shifted along the series axis.

        :param amount: amount of values shifted
        :return: the amount of values that the contextual distance matrix should shift along the series axis
        """
        raise RuntimeError("This generator does not support series shifting.")


class GeneralStaticManager(AbstractContextManager):
    """
    General purpose context manager for contextual matrix profile. This manager does not support streaming data.
    """

    def __init__(self, series_contexts, query_contexts=None):
        """
        Creates a new context manager.

        :param series_contexts: an iterable of ranges, each range defines one context. You can also
          use lists of ranges, to specify non-consecutive contexts.
        :param query_contexts: iterable of ranges, defaults to None, meaning to use the same contexts as the series
        """
        _verify_ranges([r for i, r in _enumerate_flattened(series_contexts)])

        if query_contexts is None:
            query_contexts = series_contexts
        else:
            _verify_ranges([r for i, r in _enumerate_flattened(query_contexts)])

        self._series_contexts = np.array(
            [(r.start, r.stop, i) for i, r in _filter_empty(_enumerate_flattened(series_contexts))], dtype=np.int)
        self._query_contexts = np.array(
            [(r.start, r.stop, i) for i, r in _filter_empty(_enumerate_flattened(query_contexts))], dtype=np.int)

        self._qc_sorted_start = self._query_contexts[np.argsort(self._query_contexts[:, 0])]
        self._qc_sorted_stop = self._query_contexts[np.argsort(self._query_contexts[:, 1])]

    def context_matrix_shape(self) -> (int, int):
        num_query_contexts = np.max(self._query_contexts[:, 2]) + 1
        num_series_contexts = np.max(self._series_contexts[:, 2]) + 1

        return num_query_contexts, num_series_contexts

    def series_contexts(self, start, stop):
        return self._series_contexts[np.logical_and(
            self._series_contexts[:, 0] < stop,  # Start of context is before stop
            self._series_contexts[:, 1] > start  # End of context is after start
        )]

    def query_contexts(self, start, stop):
        if start <= self._qc_sorted_start[0, 0] and stop >= self._qc_sorted_stop[-1, 1]:
            return self._query_contexts

        if start == 0:
            # All contexts that start before stop
            contexts = self._qc_sorted_start[0: np.searchsorted(self._qc_sorted_start[:, 0], stop)]
            return filter(lambda c: c[1] > 0, contexts)
        elif stop >= self._qc_sorted_stop[-1, 1]:
            # All contexts that end after start
            contexts = self._qc_sorted_stop[np.searchsorted(self._qc_sorted_stop[:, 1], start, side="right"):]
            return filter(lambda c: c[0] < stop, contexts)
        else:
            return self._query_contexts[np.logical_and(
                self._query_contexts[:, 0] < stop,  # Start of context is before stop
                self._query_contexts[:, 1] > start  # End of context is after start
            )]


def _verify_ranges(ranges):
    for r in ranges:
        if r.step != 1:
            raise RuntimeError("Only ranges with step 1 supported.")
        if r.start < 0:
            raise RuntimeError("Range start should not be negative.")


def _enumerate_flattened(l):
    """
    Converts a list of elements and lists into tuples (index, element), so that elements in nested lists
    have the same index.

    Eg: [1, [2,3], 4] => (0, 1), (1, 2), (1, 3), (2, 4)
    """
    for i, el in enumerate(l):
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, range):
            for r in el:
                yield i, r
        else:
            yield i, el


def _filter_empty(iter):
    for i, r in iter:
        if r.start < r.stop:
            yield (i, r)
