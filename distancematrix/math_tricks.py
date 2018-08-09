import numpy as np
from distancematrix.ringbuffer import RingBuffer

_EPS = 1e-12


def sliding_mean_std(series, m):
    """
    Calculates the sliding mean and standard deviation over the series using a window of size m.

    :param series: 1D array
    :param m: sliding window size
    :return: tuple of 2 arrays, each of size (len(series) - m + 1)
    """
    mean, var = sliding_mean_var(series, m)
    return mean, np.sqrt(var)


def sliding_mean_var(series, m):
    if m <= 0 or not isinstance(m, int):
        raise RuntimeError('m should be an integer > 0.')

    if not np.isfinite(series).all():
        raise RuntimeError('Provided series contains nan or infinite values.')

    n = len(series)

    cum_sum = np.cumsum(series / m)
    sliding_avg = cum_sum[m - 1:n].copy()  # x(0..m-1) x(0..m) x(0..m+1) ... x(0..n)
    sliding_avg[1:] -= cum_sum[0:n - m]  # x(0..m-1) x(1..m) x(2..m+1) ... x(n-m+1..n)

    cum_sum_sq = np.cumsum(np.square(series) / m)
    series_sum_sq = cum_sum_sq[m - 1:n].copy()  # x²(0..m-1) x²(0..m) x²(0..m+1) ... x²(0..n)
    series_sum_sq[1:] -= cum_sum_sq[0:n - m]  # x²(0..m-1) x²(1..m) x²(2..m+1) ... x²(n-m+1..n)

    sliding_var = series_sum_sq - np.square(sliding_avg)  # std^2 = E[X²] - E[X]²
    sliding_var[sliding_var < _EPS] = 0  # Due to rounding errors, zero values can have very small non-zero values

    return sliding_avg, sliding_var


class StreamingStats(object):
    """
    Class that tracks a data stream and corresponding mean and variance of a window over this data.

    The data stream has to be updated by the user, after which the mean/variance stream will be updated automatically.

    This class uses RingBuffers internally, so any old view (data, mean, variance) should be considered unreliable
    after new data was pushed to this class.
    """

    def __init__(self, series, m) -> None:
        """
        Creates a new instance. This instance will keep track of a data stream (with dimensions matching those of
        series) and a stream of moving mean and variances using a window of length m.

        :param series: Starting data of the data stream
        :param m: window size for mean and variance
        """
        if m > series.shape[-1]:
            raise RuntimeError("M should be <= series.shape[-1].")

        self._data_buffer = RingBuffer(series)
        self._m = m

        sliding_avg, sliding_var = sliding_mean_var(series, m)
        self._mean_buffer = RingBuffer(sliding_avg)
        self._var_buffer = RingBuffer(sliding_var)

    def append(self, data):
        buffer_length = self._data_buffer.view.shape[-1]
        data_length = data.shape[-1]
        if data_length >= buffer_length:
            sliding_avg, sliding_var = sliding_mean_var(data[..., -buffer_length:], self._m)
            self._mean_buffer.push(sliding_avg)
            self._var_buffer.push(sliding_var)
        else:
            # Sliding variance formula: http://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
            # First steps of derivation: http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/

            old_mean = self._mean_buffer.view[..., -1]
            old_var = self._var_buffer.view[..., -1]
            values_to_remove = self._data_buffer.view[..., -self._m: min(-1, -self._m + data_length)]
            values_to_add = data[..., :values_to_remove.shape[-1]]
            new_means = old_mean + np.cumsum(- values_to_remove + values_to_add) / self._m
            old_means = np.concatenate((np.atleast_1d(old_mean), new_means[..., :-1]))
            new_vars = old_var + np.cumsum((values_to_add - values_to_remove) * (values_to_add - new_means + values_to_remove - old_means) / self._m)
            new_vars[new_vars < _EPS] = 0.

            self._mean_buffer.push(new_means)
            self._var_buffer.push(new_vars)

            if data_length >= self._m:
                sliding_avg, sliding_var = sliding_mean_var(data, self._m)
                self._mean_buffer.push(sliding_avg)
                self._var_buffer.push(sliding_var)

        self._data_buffer.push(data)

    @property
    def data(self):
        return self._data_buffer.view

    @property
    def mean(self):
        return self._mean_buffer.view

    @property
    def var(self):
        return self._var_buffer.view
