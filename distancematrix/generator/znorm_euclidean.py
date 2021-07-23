import numpy as np
from scipy.signal import convolve

from distancematrix.util import diag_length
from distancematrix.math_tricks import sliding_mean_std
from distancematrix.generator.abstract_generator import AbstractGenerator
from distancematrix.generator.abstract_generator import AbstractBoundStreamingGenerator
from distancematrix.ringbuffer import RingBuffer

_EPS = 1e-12


class ZNormEuclidean(AbstractGenerator):
    """
    Class capable of efficiently calculating parts of the z-normalized distance matrix between two series,
    where each entry in the distance matrix equals the euclidean distance between 2 z-normalized
    (zero mean and unit variance) subsequences of both series.

    This generator can handle streaming data.
    """

    def __init__(self, noise_std=0., rb_scale_factor=2.):
        """
        Creates a new instance.

        :param noise_std: standard deviation of measurement noise, if not zero, the resulting distances will
            be adjusted to eliminate the influence of the noise.
        :param rb_scale_factor: scaling factor used for RingBuffers in case of streaming data (should be >= 1),
            this allows choosing a balance between less memory (low values) and reduced data copying (higher values)
        """

        if noise_std < 0.:
            raise ValueError("noise_std should be >= 0, it was: " + str(noise_std))
        if rb_scale_factor < 1.:
            raise ValueError("rb_scale_factor should be >= 1, it was: " + str(rb_scale_factor))

        self.noise_std = noise_std
        self._rb_scale_factor = rb_scale_factor

    def prepare_streaming(self, m, series_window, query_window=None):
        series = RingBuffer(None, (series_window,), dtype=float, scaling_factor=self._rb_scale_factor)

        if query_window is not None:
            query = RingBuffer(None, (query_window,), dtype=float, scaling_factor=self._rb_scale_factor)
            self_join = False
        else:
            query = series
            self_join = True

        num_subseq_s = series.max_shape[-1] - m + 1
        mu_s = RingBuffer(None, shape=(num_subseq_s,), dtype=float, scaling_factor=self._rb_scale_factor)
        std_s = RingBuffer(None, shape=(num_subseq_s,), dtype=float, scaling_factor=self._rb_scale_factor)
        std_s_nonzero = RingBuffer(None, shape=(num_subseq_s,), dtype=bool, scaling_factor=self._rb_scale_factor)

        if not self_join:
            num_subseq_q = query.max_shape[-1] - m + 1
            mu_q = RingBuffer(None, shape=(num_subseq_q,), dtype=float, scaling_factor=self._rb_scale_factor)
            std_q = RingBuffer(None, shape=(num_subseq_q,), dtype=float, scaling_factor=self._rb_scale_factor)
            std_q_nonzero = RingBuffer(None, shape=(num_subseq_q,), dtype=bool, scaling_factor=self._rb_scale_factor)
        else:
            mu_q = mu_s
            std_q = std_s
            std_q_nonzero = std_s_nonzero

        return BoundZNormEuclidean(m, series, query, self_join, self.noise_std,
                                   mu_s, std_s, std_s_nonzero, mu_q, std_q, std_q_nonzero)

    def prepare(self, m, series, query=None):
        if series.ndim != 1:
            raise RuntimeError("Series should be 1D")
        if query is not None and query.ndim != 1:
            raise RuntimeError("Query should be 1D")

        num_subseq_s = series.shape[-1] - m + 1
        series_buffer = RingBuffer(None, shape=series.shape, dtype=float, scaling_factor=1)
        mu_s = RingBuffer(None, shape=(num_subseq_s,), dtype=float, scaling_factor=1)
        std_s = RingBuffer(None, shape=(num_subseq_s,), dtype=float, scaling_factor=1)
        std_s_nonzero = RingBuffer(None, shape=(num_subseq_s,), dtype=bool, scaling_factor=1)

        if query is not None:
            num_subseq_q = query.shape[-1] - m + 1
            query_buffer = RingBuffer(None, shape=query.shape, dtype=float, scaling_factor=1)
            mu_q = RingBuffer(None, shape=(num_subseq_q,), dtype=float, scaling_factor=1)
            std_q = RingBuffer(None, shape=(num_subseq_q,), dtype=float, scaling_factor=1)
            std_q_nonzero = RingBuffer(None, shape=(num_subseq_q,), dtype=bool, scaling_factor=1)
            self_join = False
        else:
            query_buffer = series_buffer
            mu_q = mu_s
            std_q = std_s
            std_q_nonzero = std_s_nonzero
            self_join = True

        result = BoundZNormEuclidean(m, series_buffer, query_buffer, self_join, self.noise_std,
                                     mu_s, std_s, std_s_nonzero, mu_q, std_q, std_q_nonzero)

        result.append_series(series)
        if not self_join:
            result.append_query(query)

        return result


class BoundZNormEuclidean(AbstractBoundStreamingGenerator):
    def __init__(self, m, series, query, self_join, noise_std, series_mu, series_std, series_std_nz,
                 query_mu, query_std, query_std_nz,):
        """
        :param m: subsequence length to consider for distance calculations
        :param series: empty ringbuffer, properly sized to contain the desired window for series
        :param query: empty ringbuffer, properly sized to contain the desired window for query, or the same buffer
          as series in case of a self-join
        :param self_join: whether or not a self-join should be done
        :param noise_std: standard deviation of noise on series/query, zero to disable noise cancellation
        """

        # Core values
        self.m = m
        self.series = series
        self.query = query
        self.noise_std = noise_std
        self.self_join = self_join

        # Derivated values
        self.mu_s = series_mu
        self.std_s = series_std
        self.std_s_nonzero = series_std_nz

        self.mu_q = query_mu
        self.std_q = query_std
        self.std_q_nonzero = query_std_nz

        # Caching
        self.first_row = None
        self.first_row_backlog = 0
        self.prev_calc_column_index = None
        self.prev_calc_column_dot_prod = None

    def append_series(self, values):
        if len(values) == 0:
            return

        data_dropped = self.series.push(values)
        num_dropped = len(values) - (self.series.max_shape[0] - self.series.view.shape[0])
        self.first_row_backlog += len(values)

        if len(self.series.view) >= self.m:
            num_affected = len(values) + self.m - 1
            new_mu, new_std = sliding_mean_std(self.series[-num_affected:], self.m)
            self.mu_s.push(new_mu)
            self.std_s.push(new_std)
            self.std_s_nonzero.push(new_std != 0.)

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

        if len(self.query.view) >= self.m:
            num_affected = len(values) + self.m - 1
            new_mu, new_std = sliding_mean_std(self.query[-num_affected:], self.m)
            self.mu_q.push(new_mu)
            self.std_q.push(new_std)
            self.std_q_nonzero.push(new_std != 0.)

    def calc_diagonal(self, diag):
        dl = diag_length(len(self.query.view), len(self.series.view), diag)  # Number of affected data points
        dlr = dl - self.m + 1  # Number of entries in diagonal
        cumsum = np.zeros(dl + 1, dtype=float)

        if diag >= 0:
            # Eg: for diag = 2:
            # D = (y0 * x2), (y1 * x3), (y2 * x4)...
            # cumsum = 0, D0, D0+D1, D0+D1+D2, ...
            cumsum[1:] = np.cumsum(self.query[:dl] * self.series[diag: diag + dl])
            q_range = slice(0, dlr)
            s_range = slice(diag, diag + dlr)
        else:
            # Eg: for diag = -2:
            # D = (y2 * x0), (y3 * x1), (y4 * x2)...
            # cumsum = 0, D0, D0+D1, D0+D1+D2, ...
            cumsum[1:] = np.cumsum(self.query[-diag: -diag + dl] * self.series[:dl])
            s_range = slice(0, dlr)
            q_range = slice(-diag, -diag + dlr)

        mean_q = self.mu_q[q_range]
        mean_s = self.mu_s[s_range]
        std_q = self.std_q[q_range]
        std_s = self.std_s[s_range]

        dot_prod = cumsum[self.m:] - cumsum[:dlr]

        dist_sq = np.zeros(dlr, dtype=float)
        non_zero_std_q = self.std_q_nonzero[q_range]
        non_zero_std_s = self.std_s_nonzero[s_range]

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
        dist_sq = np.zeros(len(self.query.view) - self.m + 1, dtype=float)
        series_subseq = self.series[column: column + self.m]

        if self.prev_calc_column_index != column - 1 or column == 0:
            # Previous column not cached or data for incremental calculation not available: full calculation
            dot_prod = convolve(self.query.view, series_subseq[::-1], 'valid')
        else:
            # Previous column cached, reuse it
            if self.first_row is None:
                first_query = self.query[0:self.m]
                self.first_row = RingBuffer(convolve(self.series.view, first_query[::-1], 'valid'),
                                            shape=(self.series.max_shape[0] - self.m + 1,))
                self.first_row_backlog = 0
            elif self.first_row_backlog > 0:
                # Series has been updated since last calculation of first_row
                elems_to_recalc = self.first_row_backlog + self.m - 1
                first_query = self.query[0:self.m]
                self.first_row.push(convolve(self.series[-elems_to_recalc:], first_query[::-1], 'valid'))
                self.first_row_backlog = 0

            dot_prod = self.prev_calc_column_dot_prod  # work in same array
            dot_prod[1:] = (self.prev_calc_column_dot_prod[:-1]
                            - self.series[column - 1] * self.query[:len(self.query.view) - self.m]
                            + self.series[column + self.m - 1] * self.query[self.m:])
            dot_prod[0] = self.first_row[column]

        self.prev_calc_column_dot_prod = dot_prod
        self.prev_calc_column_index = column

        if self.std_s[column] != 0:
            q_valid = self.std_q.view != 0

            # Series subsequence is not stable, if query subsequence is stable, the distance is sqrt(m) by definition.
            dist_sq[~q_valid] = self.m

            dist_sq[q_valid] = 2 * (self.m - (dot_prod[q_valid] - self.m * self.mu_q[q_valid] * self.mu_s[column]) /
                                    (self.std_q[q_valid] * self.std_s[column]))
        else:
            # Series subsequence is stable, results are either sqrt(m) or 0, depending on whether or not
            # query subsequences are stable as well.

            dist_sq[self.std_q.view != 0] = self.m
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

    def calc_single(self, row, column, dot_prod=None):
        """
        Calculates a single point of the distance matrix.

        :param row: index of the subsequence in the query series
        :param column: index of the subsequence in the data series
        :param dot_prod: the dotproduct of the subsequences, if provided, this method can run in constant time
        :return: z-normalised distance of the 2 subsequences
        """
        std_q = self.std_q[row]
        std_s = self.std_s[column]

        if std_q == 0. and std_s == 0.:
            return 0.

        if std_q == 0. or std_s == 0.:
            return self.m

        if not dot_prod:
            dot_prod = np.sum(self.query[row: row+self.m] * self.series[column: column+self.m])
        mean_q = self.mu_q[row]
        mean_s = self.mu_s[column]

        dist_sq = 2 * (self.m - (dot_prod - self.m * mean_q * mean_s) / (std_q * std_s))

        if self.noise_std != 0.:
            dist_sq -= (2 * (self.m + 1) * np.square(self.noise_std) / np.square(np.maximum(std_s, std_q)))

        if dist_sq < _EPS:
            return 0.
        else:
            return np.sqrt(dist_sq)
