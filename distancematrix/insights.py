import numpy as np


def lowest_value_idxs(array, exclude_distance):
    """
    Creates a generator that returns the indices of the lowest elements, where each index differs by at least
    exclude_distance from every previously returned index. Non-finite values are ignored.

    :param array: 1D array
    :param exclude_distance: a positive integer
    :return: a generator
    """
    if not array.ndim == 1:
        raise RuntimeError("Array should be 1-dimensional.")
    if type(exclude_distance) is not int or exclude_distance < 0:
        raise RuntimeError('Exclude distance should be positive integer.')

    array = array.astype(np.float, copy=True)
    array[~np.isfinite(array)] = np.inf

    min_idx = np.argmin(array)

    while array[min_idx] != np.inf:
        yield min_idx

        array[max(0, min_idx - exclude_distance): min_idx + exclude_distance + 1] = np.inf
        min_idx = np.argmin(array)

    return


def highest_value_idxs(array, exclude_distance):
    """
    Creates a generator that returns the indices of the highest elements, where each index differs by at least
    exclude_distance from every previously returned index. Non-finite values are ignored.

    :param array: 1D array
    :param exclude_distance: a positive integer
    :return: a generator
    """
    return lowest_value_idxs(-array, exclude_distance)
