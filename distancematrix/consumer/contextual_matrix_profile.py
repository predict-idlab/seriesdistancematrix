import numpy as np

from distancematrix.ringbuffer import RingBuffer
from distancematrix.consumer.abstract_consumer import AbstractStreamingConsumer
from distancematrix.consumer.contextmanager import AbstractContextManager


class ContextualMatrixProfile(AbstractStreamingConsumer):
    """
    A consumer that constructs the contextual matrix profile. The contextual matrix profile is formed by
    taking the minimum of rectangles across the full distance matrix (where the matrix profile takes the
    minimum across columns).

    This consumer supports streaming if the provided context manager does.
    """

    def __init__(self, context_manager: AbstractContextManager, rb_scale_factor=2.):
        """
        Creates a new consumer that calculates a contextual matrix profile,
        according to the contexts defined by the manager.

        :param context_manager: object responsible for defining the spans of each context over the query and series axis
        :param rb_scale_factor: scaling factor used for RingBuffers in case of streaming data (should be >= 1),
            this allows choosing a balance between less memory (low values) and reduced data copying (higher values)
        """
        if rb_scale_factor < 1.:
            raise ValueError("rb_scale_factor should be >= 1, it was: " + str(rb_scale_factor))

        self._num_series_subseq = None
        self._num_query_subseq = None
        self._range = None

        self._contexts = context_manager
        self._query_shift = 0
        self._series_shift = 0

        self._distance_matrix = None
        self._match_index_series = None
        self._match_index_query = None

        self._rb_scale_factor = rb_scale_factor

    def initialise(self, dims, query_subseq, series_subseq):
        self._num_series_subseq = series_subseq
        self._num_query_subseq = query_subseq
        self._range = np.arange(0, max(series_subseq, query_subseq), dtype=int)

        num_query_contexts, num_series_contexts = self._contexts.context_matrix_shape()

        self._distance_matrix = RingBuffer(np.full((num_query_contexts, num_series_contexts), np.Inf, dtype=float),
                                           scaling_factor=self._rb_scale_factor)
        self._match_index_series = RingBuffer(np.full((num_query_contexts, num_series_contexts), -1, dtype=int),
                                              scaling_factor=self._rb_scale_factor)
        self._match_index_query = RingBuffer(np.full((num_query_contexts, num_series_contexts), -1, dtype=int),
                                             scaling_factor=self._rb_scale_factor)

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
                is_better = min_sss_value < self._distance_matrix[c0_identifier, c1_identifier]

                if is_better:
                    self._distance_matrix[c0_identifier, c1_identifier] = min_sss_value
                    rel_indices = np.argmin(sss_values)
                    sss0_start = sss1_start - diag
                    self._match_index_query[c0_identifier, c1_identifier] = rel_indices + sss0_start + self._query_shift
                    self._match_index_series[c0_identifier, c1_identifier] = rel_indices + sss1_start + self._series_shift

    def process_column(self, column_index, values):
        values = values[0]
        context1_idxs = self._contexts.series_contexts(column_index, column_index + 1)

        for _, _, c1_identifier in context1_idxs:
            query_contexts = self._contexts.query_contexts(0, self._num_query_subseq)

            for c0_start, c0_end, c0_identifier in query_contexts:
                subseq = values[c0_start: c0_end]
                best_value = np.min(subseq)

                if best_value < self._distance_matrix[c0_identifier, c1_identifier]:
                    self._distance_matrix[c0_identifier, c1_identifier] = best_value
                    self._match_index_query[c0_identifier, c1_identifier] = np.argmin(subseq) + c0_start + self._query_shift
                    self._match_index_series[c0_identifier, c1_identifier] = column_index + self._series_shift

    def shift_series(self, amount):
        context_shift = self._contexts.shift_series(amount)
        self._series_shift += amount

        if context_shift > 0:
            height = self._distance_matrix.max_shape[0]
            self._distance_matrix.push(np.full((height, context_shift), np.Inf, dtype=float))
            self._match_index_series.push(np.full((height, context_shift), -1, dtype=int))
            self._match_index_query.push(np.full((height, context_shift), -1, dtype=int))

    def shift_query(self, amount):
        context_shift = self._contexts.shift_query(amount)
        self._query_shift += amount

        if context_shift > 0:
            # Note: This could be more efficient using a 2D Ringbuffer.
            height = min(context_shift, self._distance_matrix.max_shape[0])
            self._distance_matrix.view = np.roll(self._distance_matrix.view, context_shift, axis=0)
            self._distance_matrix[-height:, :] = np.Inf
            self._match_index_series.view = np.roll(self._match_index_series.view, context_shift, axis=0)
            self._match_index_series[-height:, :] = -1
            self._match_index_query.view = np.roll(self._match_index_query.view, context_shift, axis=0)
            self._match_index_query[-height:, :] = -1

    @property
    def match_index_query(self):
        return self._match_index_query.view

    @property
    def match_index_series(self):
        return self._match_index_series.view

    @property
    def distance_matrix(self):
        return self._distance_matrix.view
