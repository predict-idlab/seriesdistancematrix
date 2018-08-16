import collections
import numpy as np
from .abstract_consumer import AbstractConsumer


class ContextualMatrixProfile(AbstractConsumer):
    """
    A consumer that constructs the conceptual matrix profile. The conceptual matrix profile is formed by
    taking the minimum of rectangles across the full distance matrix (where the matrix profile takes the
    minimum across columns).
    """

    def __init__(self, series_contexts, query_contexts=None):
        """
        Creates a new consumer that calculates a contextual matrix profile, along with corresponding indices.

        :param series_contexts: an iterable of ranges, each range defines one context. You can also
          use lists of ranges, to specify non-consecutive contexts.
        :param query_contexts: iterable of ranges, defaults to None, meaning to use the same contexts as the series
        """
        self._num_series_subseq = None
        self._num_query_subseq = None
        self._range = None

        self._verify_ranges([r for i, r in _enumerate_flattened(series_contexts)])

        if query_contexts is None:
            query_contexts = series_contexts
        else:
            self._verify_ranges([r for i, r in _enumerate_flattened(query_contexts)])

        self._series_contexts = np.array(
            [(r.start, r.stop, i) for i, r in _enumerate_flattened(series_contexts)], dtype=np.int)
        self._query_contexts = np.array(
            [(r.start, r.stop, i) for i, r in _enumerate_flattened(query_contexts)], dtype=np.int)

        self._qc_sorted_start = self._query_contexts[np.argsort(self._query_contexts[:, 0])]
        self._qc_sorted_stop = self._query_contexts[np.argsort(self._query_contexts[:, 1])]

        self.distance_matrix = None
        self.match_index_series = None
        self.match_index_query = None

    def initialise(self, dims, query_subseq, series_subseq):
        self._num_series_subseq = series_subseq
        self._num_query_subseq = query_subseq
        self._range = np.arange(0, max(series_subseq, query_subseq), dtype=np.int)

        num_series_contexts = np.max(self._series_contexts[:, 2]) + 1
        num_query_contexts = np.max(self._query_contexts[:, 2]) + 1

        self.distance_matrix = np.full((num_query_contexts, num_series_contexts), np.Inf, dtype=np.float)
        self.match_index_series = np.full((num_query_contexts, num_series_contexts), -1, dtype=np.int)
        self.match_index_query = np.full((num_query_contexts, num_series_contexts), -1, dtype=np.int)

    def _verify_ranges(self, ranges):
        for r in ranges:
            if r.step != 1:
                raise RuntimeError("Only ranges with step 1 supported.")
            if r.start < 0:
                raise RuntimeError("Range start should not be negative.")

    def process_diagonal(self, diag, values):
        values = values[0]
        num_values = len(values)

        if diag >= 0:
            values_idx0_start = 0  # Absolute index where values belong in the distance matrix (first index)
            values_idx1_start = diag  # Absolute index where values belong in the distance matrix (2nd index)
            # All contexts that start before last value passed
            context0_idxs = self._qc_sorted_start[0: np.searchsorted(self._qc_sorted_start[:, 0], num_values)]
        else:
            values_idx0_start = -diag
            values_idx1_start = 0
            # All contexts whose end is on or after the first value passed
            context0_idxs = self._qc_sorted_stop[
                            np.searchsorted(self._qc_sorted_stop[:, 1], values_idx0_start, side="right"):]

        for c0_start, c0_end, c0_identifier in context0_idxs:
            # We now have a sub-sequence (ss) defined by the first context on the query axis
            # In absolute coordinates, start/end of this subsequence on 2nd axis (series axis)
            ss1_start = max(0, c0_start + diag)
            ss1_end = min(self._num_series_subseq, min(self._num_query_subseq, c0_end) + diag)

            context1_idxs = self._series_contexts[np.logical_and(
                self._series_contexts[:, 0] < ss1_end,  # Start of context is before end of sequence
                self._series_contexts[:, 1] > ss1_start  # End of context is after start of sequence
            )]

            for c1_start, c1_end, c1_identifier in context1_idxs:
                # In absolute coordinates, start/end of the subsequence on 2nd axis defined by both contexts
                sss1_start = max(ss1_start, c1_start)
                sss1_end = min(ss1_end, c1_end)

                # Values that belong to both contexts
                sss_values = values[sss1_start - values_idx1_start: sss1_end - values_idx1_start]

                # Compare if better than current
                min_sss_value = np.min(sss_values)
                is_better = min_sss_value < self.distance_matrix[c0_identifier, c1_identifier]

                if is_better:
                    self.distance_matrix[c0_identifier, c1_identifier] = min_sss_value
                    rel_indices = np.argmin(sss_values)
                    sss0_start = sss1_start - diag
                    self.match_index_query[c0_identifier, c1_identifier] = rel_indices + sss0_start
                    self.match_index_series[c0_identifier, c1_identifier] = rel_indices + sss1_start

    def process_column(self, column_index, values):
        values = values[0]

        context1_idxs = self._series_contexts[np.logical_and(
            self._series_contexts[:, 0] < column_index + 1,  # Start of context is on or before column
            self._series_contexts[:, 1] > column_index  # End of context is after column
        )]

        for _, _, c1_identifier in context1_idxs:
            for c0_start, c0_end, c0_identifier in self._query_contexts:
                subseq = values[c0_start: c0_end]
                best_value = np.min(subseq)

                if best_value < self.distance_matrix[c0_identifier, c1_identifier]:
                    self.distance_matrix[c0_identifier, c1_identifier] = best_value
                    self.match_index_query[c0_identifier, c1_identifier] = np.argmin(subseq) + c0_start
                    self.match_index_series[c0_identifier, c1_identifier] = column_index


def _enumerate_flattened(l):
    """
    Converts a list of elements and lists into tuples (index, element), so that elements in nested lists
    have the same index.

    Eg: [1, [2,3], 4] => (0, 1), (1, 2), (1, 3), (2, 4)
    """
    for i, el in enumerate(l):
        if isinstance(el, collections.Iterable) and not isinstance(el, range):
            for r in el:
                yield i, r
        else:
            yield i, el
