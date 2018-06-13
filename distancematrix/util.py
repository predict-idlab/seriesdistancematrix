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
