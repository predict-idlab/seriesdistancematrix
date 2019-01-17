import numpy as np
from distancematrix.ringbuffer import RingBuffer
from distancematrix.util import sliding_window_view


def sliding_mean_std(series, m):
    """
    Calculates the sliding mean and standard deviation over the series using a window of size m.
    The series should only contain finite values.

    :param series: 1D numpy array
    :param m: sliding window size
    :return: tuple of 2 arrays, each of size (len(series) - m + 1)
    """
    if m <= 0 or not isinstance(m, int):
        raise RuntimeError('m should be an integer > 0.')

    if series.ndim != 1:
        raise RuntimeError('series should be one dimensional')

    if not np.isfinite(series).all():
        raise RuntimeError('Provided series contains nan or infinite values.')

    sliding_view = sliding_window_view(series, [m])
    return np.mean(sliding_view, axis=1), np.std(sliding_view, axis=1)


def sliding_mean_var(series, m):
    """
    Calculates the sliding mean and variance over the series using a window of size m.
    The series should only contain finite values.

    :param series: 1D numpy array
    :param m: sliding window size
    :return: tuple of 2 arrays, each of size (len(series) - m + 1)
    """
    if m <= 0 or not isinstance(m, int):
        raise RuntimeError('m should be an integer > 0.')

    if series.ndim != 1:
        raise RuntimeError('series should be one dimensional')

    if not np.isfinite(series).all():
        raise RuntimeError('Provided series contains nan or infinite values.')

    sliding_view = sliding_window_view(series, [m])
    return np.mean(sliding_view, axis=1), np.var(sliding_view, axis=1)


class StreamingStats(object):
    """
    Class that tracks a data stream and corresponding mean and standard deviation of a window over this data.

    The data stream has to be updated by the user, after which the mean/std stream will be updated automatically.

    This class uses RingBuffers internally, so any old view (data, mean, std) should be considered unreliable
    after new data was pushed to this class.
    """

    def __init__(self, series, m) -> None:
        """
        Creates a new instance. This instance will keep track of a data stream (with dimensions matching those of
        series) and a stream of moving mean and standard deviation using a window of length m.

        :param series: Starting data of the data stream
        :param m: window size for mean and variance
        """
        if m > series.shape[-1]:
            raise RuntimeError("M should be <= series.shape[-1].")

        self._data_buffer = RingBuffer(series)
        self._m = m

        sliding_avg, sliding_std = sliding_mean_std(series, m)
        self._mean_buffer = RingBuffer(sliding_avg)
        self._std_buffer = RingBuffer(sliding_std)

    def append(self, data):
        data_length = data.shape[-1]

        if data_length == 0:
            return

        self._data_buffer.push(data)
        new_means, new_stds = sliding_mean_std(self._data_buffer[max(-self._m - 1 - data_length, 0):], self._m)
        self._mean_buffer.push(new_means)
        self._std_buffer.push(new_stds)

        # Original implementation below, this approach might still be interesting if the current approach proves to be
        # too slow in practice. One issue that remains to be solved (why this method was replaced) is that
        # a mid-signal constant window will not result in variance of 0. One approach might be to simply check
        # for constant signals. A starting point might be:
        # https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi?rq=1
        # The numerical stability test gives a use case where this method fails.
        #
        # buffer_length = self._data_buffer.view.shape[-1]
        # if data_length >= buffer_length:
        #     sliding_avg, sliding_var = sliding_mean_var(data[..., -buffer_length:], self._m)
        #     self._mean_buffer.push(sliding_avg)
        #     self._var_buffer.push(sliding_var)
        # else:
        #     # Sliding variance formula: http://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
        #     # First steps of derivation: http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
        #     # (For non-online calculation, the formula used in sliding_mean_var is faster)
        #
        #     old_mean = self._mean_buffer.view[..., -1]
        #     old_var = self._var_buffer.view[..., -1]
        #     values_to_remove = self._data_buffer.view[..., -self._m: min(-1, -self._m + data_length)]
        #     values_to_add = data[..., :values_to_remove.shape[-1]]
        #     new_means = old_mean + np.cumsum(- values_to_remove + values_to_add) / self._m
        #     old_means = np.concatenate((np.atleast_1d(old_mean), new_means[..., :-1]))
        #     new_vars = old_var + np.cumsum((values_to_add - values_to_remove) * (
        #        values_to_add - new_means + values_to_remove - old_means) / self._m)
        #     new_vars[new_vars < 1e-12] = 0.  # Unreliable!
        #
        #     self._mean_buffer.push(new_means)
        #     self._var_buffer.push(new_vars)
        #
        #     if data_length >= self._m:
        #         sliding_avg, sliding_var = sliding_mean_var(data, self._m)
        #         self._mean_buffer.push(sliding_avg)
        #         self._var_buffer.push(sliding_var)
        #
        # self._data_buffer.push(data)

    @property
    def data(self):
        return self._data_buffer.view

    @property
    def mean(self):
        return self._mean_buffer.view

    @property
    def std(self):
        return self._std_buffer.view
