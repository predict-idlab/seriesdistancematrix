import numpy as np

_EPS = 1e-12


def sliding_mean_std(series, m):
    """
    Calculates the sliding mean and standard deviation over the series using a window of size m.

    :param series: 1D array
    :param m: sliding window size
    :return: tuple of 2 arrays, each of size (len(series) - m + 1)
    """

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

    sliding_std_sq = series_sum_sq - np.square(sliding_avg)  # std^2 = E[X²] - E[X]²
    sliding_std_sq[sliding_std_sq < _EPS] = 0  # Due to rounding errors, zero values can have very small non-zero values
    sliding_std = np.sqrt(sliding_std_sq)

    return sliding_avg, sliding_std
