import numpy as np

from distancematrix.util import diag_indices_of
from distancematrix.generator.abstract_generator import AbstractGenerator
from distancematrix.generator.abstract_generator import AbstractBoundStreamingGenerator


class MockGenerator(AbstractGenerator):
    """
    Mock generator for testing purposes. Simply returns distances from a given distance matrix.
    """

    def __init__(self, dist_matrix):
        """
        Creates a new mock generator that will return distances from the provided distance matrix.

        :param dist_matrix: distances to return.
        """
        self._dist_matrix = dist_matrix

        # Storage for parameters used for prepare and prepare_streaming
        self.m = None
        self.series_window = None
        self.query_window = None
        self.series = None
        self.query = None
        self.bound_gen = None

    def prepare_streaming(self, m, series_window, query_window=None):
        self.m = m
        self.series_window = series_window
        self.query_window = query_window


        if query_window is None:
            query_window = series_window
            self_join = True
        else:
            self_join = False

        s_subseqs = series_window - m + 1
        q_subseqs = query_window - m + 1
        self.bound_gen = BoundMockGenerator(self._dist_matrix, s_subseqs, q_subseqs,
                                            self_join, -m - s_subseqs + 1, -m - q_subseqs + 1)

        return self.bound_gen

    def prepare(self, m, series, query=None):
        self.m = m
        self.series = series
        self.query = query

        s_win = len(series) - m + 1
        if query is None:
            q_win = s_win
            self_join = True
        else:
            q_win = len(query) - m + 1
            self_join = False

        self.bound_gen = BoundMockGenerator(self._dist_matrix, s_win, q_win, self_join, 0, 0)
        return self.bound_gen


class BoundMockGenerator(AbstractBoundStreamingGenerator):
    """
    Mock generator for testing purposes. Simply returns distances from a given distance matrix.
    """
    def __init__(self, dist_matrix, s_win, q_win, self_join, s_view_index, q_view_index):
        """
        Creates a new mock generator that will return distances from the provided distance matrix.

        :param dist_matrix: 2D matrix, base distance values to use, a view will be used to determine
            which values to return for mocked calculations
        :param s_win: window size of the view over the series axis
        :param q_win: window size of the view over the query axis
        :param self_join: are we doing a self-join (does adding series data also implicitly add query data)
        :param s_view_index: start index of the view of dist_matrix (for series)
        :param q_view_index: start index of the view of dist_matrix (for query)
        """
        self._dist_matrix = dist_matrix
        self._s_win = s_win
        self._q_win = q_win
        self._self_join = self_join

        self._s_index = s_view_index
        self._q_index = q_view_index

        self.appended_series = np.empty((0,), dtype=np.float)
        self.appended_query = np.empty((0,), dtype=np.float)

    def calc_diagonal(self, diag):
        view = self._dist_matrix[
               max(self._q_index, 0): self._q_index + self._q_win,
               max(self._s_index, 0): self._s_index + self._s_win
               ]
        return view[diag_indices_of(view, diag)]

    def calc_column(self, column):
        view = self._dist_matrix[
               max(self._q_index, 0): self._q_index + self._q_win,
               max(self._s_index, 0): self._s_index + self._s_win
               ]
        return view[:, column]

    def append_series(self, values):
        self.appended_series = np.concatenate([self.appended_series, values])
        self._s_index += len(values)
        if self._self_join:
            self._q_index += len(values)

    def append_query(self, values):
        if self._self_join:
            raise RuntimeError("Should not append query if self-joining.")

        self.appended_query = np.concatenate([self.appended_query, values])
        self._q_index += len(values)
