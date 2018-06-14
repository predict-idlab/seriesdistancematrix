import numpy as np
from scipy.signal import fftconvolve

from distancematrix.util import diag_length
from distancematrix.math_tricks import sliding_mean_std

_EPS = 1e-12


class ZNormEuclidean(object):
    """
    Class capable of efficiently calculating parts of the z-normalized distance matrix between two series,
    where each entry in the distance matrix equals the euclidean distance between 2 z-normalized
    (zero mean and unit variance) subsequences of both series.
    """

    def __init__(self, noise_std):
        """
        Creates a new instance.

        :param noise_std: standard deviation of measurement noise, if not zero, the resulting distances will
          be adjusted to eliminate the influence of the noise.
        """
        # Core values
        self.m = None
        self.series = None
        self.query = None
        self.noise_std = noise_std

        # Derivated values
        self.mu_s = None
        self.mu_q = None
        self.std_s = None
        self.std_q = None

        # Caching
        self.first_row = None
        self.prev_calc_column_index = None
        self.prev_calc_column_dot_prod = None

    def prepare(self, series, query, m):
        self.series = np.array(series, dtype=np.float, copy=True)
        self.query = np.array(query, dtype=np.float, copy=True)
        self.m = m

        self.mu_s, self.std_s = sliding_mean_std(series, m)
        self.mu_q, self.std_q = sliding_mean_std(query, m)

        if series.ndim != 1:
            raise RuntimeError("Series should be 1D")
        if query.ndim != 1:
            raise RuntimeError("Query should be 1D")

    def calc_diagonal(self, diag):
        dl = diag_length(len(self.query), len(self.series), diag)  # Number of affected data points
        dlr = dl - self.m + 1  # Number of entries in diagonal
        cumsum = np.zeros(dl + 1, dtype=np.float)

        if diag >= 0:
            # Eg: for diag = 2:
            # D = (y0 * x2), (y1 * x3), (y2 * x4)...
            # cumsum = 0, D0, D0+D1, D0+D1+D2, ...
            cumsum[1:] = np.cumsum(self.query[:dl] * self.series[diag: diag + dl])
            q_range = range(0, dlr)
            s_range = range(diag, diag + dlr)
        else:
            # Eg: for diag = -2:
            # D = (y2 * x0), (y3 * x1), (y4 * x2)...
            # cumsum = 0, D0, D0+D1, D0+D1+D2, ...
            cumsum[1:] = np.cumsum(self.query[-diag: -diag + dl] * self.series[:dl])
            s_range = range(0, dlr)
            q_range = range(-diag, -diag + dlr)

        mean_q = self.mu_q[q_range]
        mean_s = self.mu_s[s_range]
        std_q = self.std_q[q_range]
        std_s = self.std_s[s_range]

        dot_prod = cumsum[self.m:] - cumsum[:dlr]

        dist_sq = np.zeros(dlr, dtype=np.float)
        non_zero_std_q = std_q != 0
        non_zero_std_s = std_s != 0

        # For subsequences where both signals are stable (std = 0), we define the distance as zero.
        # This is covered by the initialization of the dist array.
        # For subsequences where exactly one signal is stable, the distance is sqrt(m) by definition.
        dist_sq[np.logical_xor(non_zero_std_q, non_zero_std_s)] = self.m

        # Formula for regular (non-stable) subsequences
        mask = np.logical_and(non_zero_std_q, non_zero_std_s)
        dist_sq[mask] = (2 * (self.m - (dot_prod[mask] - self.m * mean_q[mask] * mean_s[mask]) /
                              (std_q[mask] * std_s[mask])))

        # Noise correction - See paper "Eliminating noise in the matrix profile"
        if self.noise_std != 0.:
            mask = np.logical_or(non_zero_std_q, non_zero_std_s)
            dist_sq[mask] -= (2 * (self.m + 1) * np.square(self.noise_std) /
                              np.square(np.maximum(std_s[mask], std_q[mask])))

        # Before the noise correction, small negative values are possible due to rounding.
        # After the noise, larger negative values are also possible.
        # Correct all negative values to zero.
        dist_sq[dist_sq < _EPS] = 0

        return np.sqrt(dist_sq)

    def calc_column(self, column):
        dist_sq = np.zeros(len(self.query) - self.m + 1, dtype=np.float)
        series_subseq = self.series[column: column + self.m]

        if self.prev_calc_column_index != column - 1:
            # Previous column not cached, full calculation
            dot_prod = fftconvolve(self.query, series_subseq[::-1], 'valid')
        else:
            # Previous column cached, reuse it
            if self.first_row is None:
                first_query = self.query[0:self.m]
                self.first_row = fftconvolve(self.series, first_query[::-1], 'valid')

            dot_prod = self.prev_calc_column_dot_prod  # work in same array
            dot_prod[1:] = (self.prev_calc_column_dot_prod[:-1]
                            - self.series[column - 1] * self.query[:len(self.query) - self.m]
                            + self.series[column + self.m - 1] * self.query[self.m:])
            dot_prod[0] = self.first_row[column]

        self.prev_calc_column_dot_prod = dot_prod
        self.prev_calc_column_index = column

        if self.std_s[column] != 0:
            q_valid = self.std_q != 0

            # Series subsequence is not stable, if query subsequence is stable, the distance is sqrt(m) by definition.
            dist_sq[~q_valid] = self.m

            dist_sq[q_valid] = 2 * (self.m - (dot_prod[q_valid] - self.m * self.mu_q[q_valid] * self.mu_s[column]) /
                                    (self.std_q[q_valid] * self.std_s[column]))
        else:
            # Series subsequence is stable, results are either sqrt(m) or 0, depending on whether or not
            # query subsequences are stable as well.

            dist_sq[self.std_q != 0] = self.m
            # dist_sq[self.std_q == 0] = 0  # Covered by array initialization

        # Noise correction - See paper "Eliminating noise in the matrix profile"
        if self.noise_std != 0.:
            if self.std_s[column] != 0:
                mask = slice(None)
            else:
                mask = self.std_q != 0

            dist_sq[mask] -= (2 * (self.m + 1) * np.square(self.noise_std) /
                              np.square(np.maximum(self.std_s[column], self.std_q[mask])))

        # Before the noise correction, small negative values are possible due to rounding.
        # After the noise, larger negative values are also possible.
        # Correct all negative values to zero.
        dist_sq[dist_sq < _EPS] = 0

        return np.sqrt(dist_sq)
