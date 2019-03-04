import numpy as np

from distancematrix.util import diag_length
from distancematrix.util import sliding_window_view
from distancematrix.ringbuffer import RingBuffer
from distancematrix.generator.abstract_generator import AbstractGenerator
from distancematrix.generator.abstract_generator import AbstractBoundStreamingGenerator

EPSILON = 1e-15


class Euclidean(AbstractGenerator):
    """
    Class capable of efficiently calculating parts of the euclidean distance matrix between two series,
    where each entry in the distance matrix equals the euclidean distance between 2 subsequences of both series.

    This generator can handle streaming data.
    """
    def prepare_streaming(self, m, series_window, query_window=None):
        series = RingBuffer(None, (series_window,), dtype=np.float)

        if query_window is not None:
            query = RingBuffer(None, (query_window,), dtype=np.float)
            self_join = False
        else:
            query = series
            self_join = True

        return BoundStreamingEuclidean(m, series, query, self_join)

    def prepare(self, m, series, query=None):
        if series.ndim != 1:
            raise RuntimeError("Series should be 1D")
        if query is not None and query.ndim != 1:
            raise RuntimeError("Query should be 1D")

        series = RingBuffer(series, scaling_factor=1)
        if query is not None:
            query = RingBuffer(query, scaling_factor=1)
            self_join = False
        else:
            query = series
            self_join = True
        return BoundStreamingEuclidean(m, series, query, self_join)


class BoundStreamingEuclidean(AbstractBoundStreamingGenerator):
    def __init__(self, m, series, query, self_join):
        self.m = m
        self.series = series
        self.query = query
        self.self_join = self_join

        self.first_row = None
        self.first_row_backlog = 0  # The number of values not yet processed for the first row cache
        self.prev_calc_column_index = None
        self.prev_calc_column_sq_dist = None

    def append_series(self, values):
        if len(values) == 0:
            return

        data_dropped = self.series.push(values)
        num_dropped = len(values) - (self.series.max_shape[0] - self.series.view.shape[0])
        self.first_row_backlog += len(values)

        if self.prev_calc_column_index is not None and num_dropped > 0:
            self.prev_calc_column_index -= num_dropped

        if self.self_join:
            if data_dropped:
                self.first_row = None  # The first row was dropped by new data
            self.prev_calc_column_index = None

    def append_query(self, values):
        if self.self_join:
            raise RuntimeError("Cannot append query data in case of a self join.")

        if len(values) == 0:
            return

        if self.query.push(values):
            self.first_row = None  # The first row was dropped by new data
        self.prev_calc_column_index = None

    def calc_diagonal(self, diag):
        dl = diag_length(len(self.query.view), len(self.series.view), diag)
        cumsum = np.zeros(dl + 1, dtype=np.float)

        if diag >= 0:
            # Eg: for diag = 2:
            # D = (y0 - x2)², (y1 - x3)², (y2 - x4)²...
            # cumsum = 0, D0, D0+D1, D0+D1+D2, ...
            cumsum[1:] = np.cumsum(np.square(self.query[:dl] - self.series[diag: diag + dl]))
        else:
            # Eg: for diag = -2:
            # D = (y2 - x0)², (y3 - x1)², (y4 - x2)²...
            # cumsum = 0, D0, D0+D1, D0+D1+D2, ...
            cumsum[1:] = np.cumsum(np.square(self.query[-diag: -diag + dl] - self.series[:dl]))

        return np.sqrt(cumsum[self.m:] - cumsum[:len(cumsum) - self.m])

    def calc_column(self, column):
        if self.prev_calc_column_index != column - 1 or column == 0:
            # Previous column not cached or data for incremental calculation not available: full calculation
            sq_dist = _euclidean_distance_squared(self.query.view, self.series[column:column + self.m])
        else:
            # Previous column cached, reuse it
            if self.first_row is None:
                self.first_row = RingBuffer(_euclidean_distance_squared(self.series.view, self.query[0: self.m]),
                                            shape=(self.series.max_shape[0] - self.m + 1,))
                self.first_row_backlog = 0
            elif self.first_row_backlog > 0:
                # Series has been updated since last calculation of first_row
                elems_to_recalc = self.first_row_backlog + self.m - 1
                self.first_row.push(_euclidean_distance_squared(self.series[-elems_to_recalc:], self.query[0: self.m]))
                self.first_row_backlog = 0

            sq_dist = self.prev_calc_column_sq_dist  # work in same array
            sq_dist[1:] = (self.prev_calc_column_sq_dist[:-1]
                           - np.square(self.series[column - 1] - self.query[:len(self.query.view)-self.m])
                           + np.square(self.series[column + self.m - 1] - self.query[self.m:]))
            sq_dist[0] = self.first_row[column]

        self.prev_calc_column_sq_dist = sq_dist
        self.prev_calc_column_index = column

        return np.sqrt(sq_dist)


def _euclidean_distance_squared(series, sequence):
    """
    Calculates the squared euclidean distance between the given sequence and each possible subsequence of the series
    (using a sliding window of the same length as the sequence).

    :param series: 1D numpy array of length n
    :param sequence: 1D numpy array of length m
    :return: a 1D numpy array of length n-m+1 containing the squared euclidean distance
    """
    if series.ndim != 1:
        raise RuntimeError("Series should be 1D")
    if sequence.ndim != 1:
        raise RuntimeError("Sequence should be 1D")

    m = len(sequence)

    sliding_view = sliding_window_view(series, [m])

    # (X - Y)^2 = X^2 - 2XY + Y^2
    # Here, einsum is used to calculate dot products over sliding window to prevent memory copying.
    # Using the normal euclidean distance calculation over the sliding window (x - y)^2 would result in copying
    # each window, which leads to memory errors for long series.
    dist = np.einsum('ij,ij->i', sliding_view, sliding_view)  # Dot product of every window with itself
    dist -= 2 * np.einsum('ij,j->i', sliding_view, sequence)  # Dot product of every window with sequence
    dist += np.dot(sequence, sequence)  # Dot product of sequence with itself
    dist[dist < EPSILON] = 0  # Avoid very small negative numbers due to rounding

    # Simple implementation, this takes double as long to calculate as the einsum approach, though it contains
    # no approximations. For very long series (100k when testing), suddenly takes 10 times as long, most likely
    # due to cpu caching that cannot contain the entire series (could be circumvented by batching):
    # num_sub_seq = len(series) - m + 1
    # dist = np.zeros(num_sub_seq)
    # for i in range(m):
    #     dist += np.square(series[i:num_sub_seq + i] - sequence[i])

    return dist
