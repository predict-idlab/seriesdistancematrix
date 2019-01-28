import numpy as np

from distancematrix.consumer.abstract_consumer import AbstractConsumer
from distancematrix.consumer.contextmanager import AbstractContextManager


class ContextualMatrixProfile(AbstractConsumer):
    """
    A consumer that constructs the contextual matrix profile. The contextual matrix profile is formed by
    taking the minimum of rectangles across the full distance matrix (where the matrix profile takes the
    minimum across columns).
    """

    def __init__(self, context_manager: AbstractContextManager):
        """
        Creates a new consumer that calculates a contextual matrix profile,
        according to the contexts defined by the manager.

        :param context_manager: object responsible for defining the spans of each context over the query and series axis
        """
        self._num_series_subseq = None
        self._num_query_subseq = None
        self._range = None

        self._contexts = context_manager

        self.distance_matrix = None
        self.match_index_series = None
        self.match_index_query = None

    def initialise(self, dims, query_subseq, series_subseq):
        self._num_series_subseq = series_subseq
        self._num_query_subseq = query_subseq
        self._range = np.arange(0, max(series_subseq, query_subseq), dtype=np.int)

        num_query_contexts, num_series_contexts = self._contexts.context_matrix_shape()

        self.distance_matrix = np.full((num_query_contexts, num_series_contexts), np.Inf, dtype=np.float)
        self.match_index_series = np.full((num_query_contexts, num_series_contexts), -1, dtype=np.int)
        self.match_index_query = np.full((num_query_contexts, num_series_contexts), -1, dtype=np.int)

    def process_diagonal(self, diag, values):
        values = values[0]
        num_values = len(values)

        if diag >= 0:
            values_idx1_start = diag
            context0_idxs = self._contexts.query_contexts(0, num_values)
        else:
            values_idx1_start = 0
            context0_idxs = self._contexts.query_contexts(-diag, self._num_query_subseq)

        for c0_start, c0_end, c0_identifier in context0_idxs:
            # We now have a sub-sequence (ss) defined by the first context on the query axis
            # In absolute coordinates, start/end of this subsequence on 2nd axis (series axis)
            ss1_start = min(max(0, c0_start + diag), self._num_series_subseq)
            ss1_end = min(self._num_series_subseq, min(self._num_query_subseq, c0_end) + diag)

            if ss1_start == ss1_end:
                continue

            context1_idxs = self._contexts.series_contexts(ss1_start, ss1_end)

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
        context1_idxs = self._contexts.series_contexts(column_index, column_index + 1)

        for _, _, c1_identifier in context1_idxs:
            query_contexts = self._contexts.query_contexts(0, self._num_query_subseq)

            for c0_start, c0_end, c0_identifier in query_contexts:
                subseq = values[c0_start: c0_end]
                best_value = np.min(subseq)

                if best_value < self.distance_matrix[c0_identifier, c1_identifier]:
                    self.distance_matrix[c0_identifier, c1_identifier] = best_value
                    self.match_index_query[c0_identifier, c1_identifier] = np.argmin(subseq) + c0_start
                    self.match_index_series[c0_identifier, c1_identifier] = column_index
