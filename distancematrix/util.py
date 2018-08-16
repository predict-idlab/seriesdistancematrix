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


def cross_count(array):
    """
    Given the Matrix Profile Index, it returns a cross count
    :param array: 1D array containing the mpindex
    :return: cross count of the mpindex, which indicates how many arcs over each index exist
    """
    l = array.shape[0]
    nnmark = np.zeros(l)
    for i in range(l):
        small = min(i, array[i])
        large = max(i, array[i])
        nnmark[small] += 1
        nnmark[large] -= 1

    return np.cumsum(nnmark)


def norm_cross_count(array, window):
    """
    Corrects cross_count considering the fact that edges have less expected crosses.

    :param array: 1D array cross_count (see function)
    :param window: int, size of the window used to generate the Mpindex
    :return: normalized crosscount, as actual crosscount divided by expected crosscount
    """
    ncc = np.ones_like(array)

    l = cross_count.shape[0] + 1
    zone = window * 5  # 5 is used in the matlab code. Keeping it as such.

    for i in range(zone, l-zone):
        ac = cross_count[i]
        ic = 2 * (i + 1) * (l - i + 1) / l

        ncc[i] = min(ac/ic, 1)

    return ncc
