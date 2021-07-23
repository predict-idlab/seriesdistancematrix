import numpy as np
import pandas as pd


def diag_length(h, w, diagonal=0):
    """
    Returns the number of elements on the specified diagonal of a matrix with dimensions (h, w).

    :param h: int, height of the matrix
    :param w: int, width of the matrix
    :param diagonal: int, diagonal index of the matrix
    :return: a positive integer, zero if diagonal fall completely outside the matrix
    """
    if diagonal >= 0:
        return max(min(h, w - diagonal), 0)
    else:
        return max(min(w, h + diagonal), 0)


def diag_indices(h, w, diagonal=0):
    """
    Returns the indices of the elements on the specified diagonal of a matrix with dimensions (h, w).

    :param h: int, height of the matrix
    :param w: int, width of the matrix
    :param diagonal: int, diagonal index of the matrix
    :return: a tuple of ranges, serving as indices of the elements
    """
    dl = diag_length(h, w, diagonal)

    if diagonal >= 0:
        return range(0, dl), range(diagonal, diagonal + dl)
    else:
        return range(-diagonal, -diagonal + dl), range(0, dl)


def diag_indices_of(array, diagonal=0):
    """
    Returns the indices of the elements on the specified diagonal of the given matrix.

    :param array: 2D array
    :param diagonal: int, diagonal index of the matrix
    :return: a tuple of ranges, serving as indices of the elements
    """
    if array.ndim != 2:
        raise RuntimeError("array should be 2D")

    return diag_indices(array.shape[0], array.shape[1], diagonal)


def cut_indices_of(array, cut):
    """
    Calculates the indices of the elements on the given cut for the given matrix.
    Where a diagonal runs from top left to bottom right, a cut runs from bottom left to top right.

    :param array: 2D array
    :param cut: index of the cut (cut 0 is the single element of the top left)
    :return: the indices to retrieve the cut
    """
    if array.ndim != 2:
        raise RuntimeError("array should be 2D")

    h, w = array.shape

    if cut < 0 or cut >= w + h - 1:
        return range(0, 0), range(0, 0)

    cut_length = cut + 1 - max(0, cut - h + 1) - max(0, cut - w + 1)

    if cut < h:
        return range(cut, cut - cut_length, -1), range(0, cut_length)
    else:
        return range(h-1, h-cut_length-1, -1), range(cut - h + 1, cut - h + 1 + cut_length)


def shortest_path_distances(cost_array):
    """
    Creates a new array of the same shape, where each entry contains the lowest sum of elements on the path
    from (0, 0) to that entry. Steps in the path can go horizontal, vertical and diagonal.

    :param cost_array: 2D array containing only positives
    :return: a new array
    """
    if cost_array.ndim != 2:
        raise RuntimeError("array should be 2D")

    dist = np.empty_like(cost_array, dtype=float)

    # Borders can only come from previous step
    dist[0, :] = np.cumsum(cost_array[0, :])
    dist[:, 0] = np.cumsum(cost_array[:, 0])

    # This operation could be vectorised by calculating one cut at a time, but the index juggling becomes quite
    # complex for rectangular arrays.
    for c in range(1, dist.shape[0]):
        for r in range(1, dist.shape[1]):
            dist[c, r] = min(dist[c-1, r], dist[c, r-1], dist[c-1, r-1]) + cost_array[c, r]

    return dist


def shortest_path(cost_array):
    """
    Finds the shortest (= least summed cost) path from the top left of the array to the bottom right.

    :param cost_array: 2D array containing only positives
    :return: array of indices, starting from the top left (index: [0, 0])
    """
    if cost_array.ndim != 2:
        raise RuntimeError("array should be 2D")

    row = cost_array.shape[0] - 1
    col = cost_array.shape[1] - 1

    walk_dist_matrix = shortest_path_distances(cost_array)

    path = [(row, col)]
    while row != 0 or col != 0:
        best_cost = np.inf
        if row != 0 and col != 0:
            delta_step = (-1, -1)
            best_cost = walk_dist_matrix[row - 1, col - 1]
        if row != 0 and walk_dist_matrix[row - 1, col] < best_cost:
            delta_step = (-1, 0)
            best_cost = walk_dist_matrix[row - 1, col]
        if col != 0 and walk_dist_matrix[row, col -1] < best_cost:
            delta_step = (0, -1)

        row += delta_step[0]
        col += delta_step[1]
        path.append((row, col))

    return path[::-1] # TODO: other indices order


def sliding_min(array, window_size):
    #result = np.empty(array.shape[0] - window_size + 1, array.dtype)
    #d = collections.deque()  # d is always sorted
    #
    #for i in range(array.shape[0]):
    #    while len(d) > 0 and d[-1][0] >= array[i]:
    #        d.pop()
    #    d.append((array[i], i))
    #
    #    if d[0][1] <= i - window_size:
    #        d.popleft()
    #
    #    if i >= window_size - 1:
    #        result[i - window_size + 1] = d[0][0]
    #
    #return result

    # Pandas has implemented this in native code, speedup of about 10 times
    return pd.Series(array).rolling(window_size).min().values[window_size - 1:]


def sliding_max(array, window_size):
    return pd.Series(array).rolling(window_size).max().values[window_size - 1:]


def sliding_window_view(x, shape, step=None, subok=False, writeable=False):
    """
    Create sliding window views of the N dimensions array with the given window
    shape. Window slides across each dimension of `x` and provides subsets of `x`
    at any window position.

    ``sliding_window_view`` create sliding window views of the N dimensions array
    with the given window shape and its implementation based on ``as_strided``.
    Please note that if writeable set to False, the return is views, not copies
    of array. In this case, write operations could be unpredictable, so the return
    views is readonly. Bear in mind, return copies (writeable=True), could possibly
    take memory multiple amount of origin array, due to overlapping windows.

    For some cases, there may be more efficient approaches

    :param x: ndarray
        Array to create sliding window views.
    :param shape: sequence of int
        The shape of the window. Must have same length as number of input array dimensions.
    :param step: sequence of int, optional
        The steps of window shifts for each dimension on input array at a time.
        If given, must have same length as number of input array dimensions.
        Defaults to 1 on all dimensions.
    :param subok: bool, optional
        If True, then sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    :param writeable: bool, optional
        If set to False, the returned array will always be readonly view.
        Otherwise it will return writable copies(see Notes).
    :return: ndarray
        Sliding window views (or copies) of `x`. view.shape = (x.shape - shape) // step + 1
    """

    # MIT License
    # Copyright (c) 2018 Fanjin Zeng
    # This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.
    # https://gist.github.com/Fnjn/b061b28c05b5b0e768c60964d2cafa8d

    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    try:
        shape = np.array(shape, int)
    except:
        raise TypeError('`shape` must be a sequence of integer')
    else:
        if shape.ndim > 1:
            raise ValueError('`shape` must be one-dimensional sequence of integer')
        if len(x.shape) != len(shape):
            raise ValueError("`shape` length doesn't match with input array dimensions")
        if np.any(shape <= 0):
            raise ValueError('`shape` cannot contain non-positive value')

    if step is None:
        step = np.ones(len(x.shape), np.intp)
    else:
        try:
            step = np.array(step, np.intp)
        except:
            raise TypeError('`step` must be a sequence of integer')
        else:
            if step.ndim > 1:
                raise ValueError('`step` must be one-dimensional sequence of integer')
            if len(x.shape) != len(step):
                raise ValueError("`step` length doesn't match with input array dimensions")
            if np.any(step <= 0):
                raise ValueError('`step` cannot contain non-positive value')

    o = (np.array(x.shape) - shape) // step + 1  # output shape
    if np.any(o <= 0):
        raise ValueError('window shape cannot larger than input array shape')

    strides = x.strides
    view_strides = strides * step

    view_shape = np.concatenate((o, shape), axis=0)
    view_strides = np.concatenate((view_strides, strides), axis=0)
    view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, subok=subok, writeable=writeable)

    if writeable:
        return view.copy()
    else:
        return view
