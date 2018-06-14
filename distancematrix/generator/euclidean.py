import numpy as np

from distancematrix.util import diag_length


class Euclidean(object):
    def __init__(self):
        self.m = None
        self.series = None
        self.query = None
        self.n = None

        self.first_row = None
        self.prev_calc_column_index = None
        self.prev_calc_column_sq_dist = None

    def prepare(self, series, query, m):
        self.series = np.array(series, dtype=np.float, copy=True)
        self.query = np.array(query, dtype=np.float, copy=True)
        self.m = m
        self.n = len(series)

        if series.ndim != 1:
            raise RuntimeError("Series should be 1D")
        if query.ndim != 1:
            raise RuntimeError("Query should be 1D")

    def calc_diagonal(self, diag):
        dl = diag_length(len(self.query), self.n, diag)
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
        if self.prev_calc_column_index != column - 1:
            # Previous column not cached, full calculation
            sq_dist = _euclidean_distance_squared(self.query, self.series[column:column + self.m])
        else:
            # Previous column cached, reuse it
            if self.first_row is None:
                self.first_row = _euclidean_distance_squared(self.series, self.query[0: self.m])

            sq_dist = self.prev_calc_column_sq_dist  # work in same array
            sq_dist[1:] = (self.prev_calc_column_sq_dist[:-1]
                           - np.square(self.series[column - 1] - self.query[:len(self.query)-self.m])
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
    num_sub_seq = len(series) - m + 1

    # Simple implementation:
    dist = np.zeros(num_sub_seq)
    for i in range(m):
        dist += np.square(series[i:num_sub_seq + i] - sequence[i])

    return dist
