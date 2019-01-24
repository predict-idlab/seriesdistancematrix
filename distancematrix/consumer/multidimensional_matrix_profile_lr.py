import numpy as np
from .abstract_consumer import AbstractConsumer


class MultidimensionalMatrixProfileLR(AbstractConsumer):
    """
    A consumer that builds the multidimensional matrix profile. This consumer takes in distance measures from
    multiple channels (dimensions) at the same time and tracks the best distance, the index of this match and
    the dimensions used in this match.
    More specifically, if the input has N data channels, this consumer will select for each number of channels
    (1, 2, ..., N), the channels containing the best match, index and dimensions. It will not track matches for
    any possible combination of channels.

    This consumer keeps track of the left and right multidimensional profile, and can be used to create the
    (normal) multidimensional profile from it. The left profile, index and dimensions
    at index i contain information about a match whose index is less than or equal to i, while the right
    profile, index and dimensions track information about a match whose index is larger than i.

    The profile is an array with shape (num_dimensions, num_distances). The value at row i, j contains the best averaged
    distances encountered at index j for any i+1 dimensions. The index is similar, but tracks the index of the query
    series that had the best match.

    The dimensions being tracked is a list of length num_dimensions. Entry i of this list contains an
    (i+1, num_distances) array that lists the indices of the dimensions that contained the best match.
    """

    def __init__(self):
        """
        Creates a new consumer that calculates the left and right matrix profile, the corresponding
        indices and the used dimensions over multiple dimensions (data channels).
        """

        self._num_subseq = None
        self._range = None
        self._n_dim = None

        self.md_matrix_profile_left = None
        self.md_profile_index_left = None
        self.md_profile_dimension_left = None

        self.md_matrix_profile_right = None
        self.md_profile_index_right = None
        self.md_profile_dimension_right = None

    def initialise(self, dims, query_subseq, series_subseq):
        self._n_dim = dims
        self._num_subseq = series_subseq
        self._range = np.arange(0, self._num_subseq, dtype=np.int)

        self.md_matrix_profile_left = np.full((dims, self._num_subseq), np.inf, dtype=np.float)
        self.md_profile_index_left = np.full((dims, self._num_subseq), -1, dtype=np.int)
        self.md_profile_dimension_left = \
            [np.full((i + 1, self._num_subseq), -1, dtype=np.int) for i in range(dims)]

        self.md_matrix_profile_right = np.full((dims, self._num_subseq), np.inf, dtype=np.float)
        self.md_profile_index_right = np.full((dims, self._num_subseq), -1, dtype=np.int)
        self.md_profile_dimension_right = \
            [np.full((i + 1, self._num_subseq), -1, dtype=np.int) for i in range(dims)]

    def process_diagonal(self, diag, values):
        n_dim, num_values = values.shape

        values_sort_order = np.argsort(values, axis=0)
        values_sorted = np.sort(values, axis=0)
        values_cumsum = np.zeros(num_values)

        if diag >= 0:
            for dim in range(n_dim):
                values_cumsum += values_sorted[dim, :]
                values_mean_over_dim = values_cumsum / (dim + 1)

                self._update_matrix_profile(values_mean_over_dim,
                                            self._range[:num_values],
                                            values_sort_order[:dim + 1, :],
                                            self.md_matrix_profile_left[dim, diag:diag + num_values],
                                            self.md_profile_index_left[dim, diag:diag + num_values],
                                            self.md_profile_dimension_left[dim][:, diag:diag + num_values])

        else:
            for dim in range(n_dim):
                values_cumsum += values_sorted[dim, :]
                values_mean_over_dim = values_cumsum / (dim + 1)

                self._update_matrix_profile(values_mean_over_dim,
                                            self._range[-diag:-diag + num_values],
                                            values_sort_order[:dim + 1, :],
                                            self.md_matrix_profile_right[dim, :num_values],
                                            self.md_profile_index_right[dim, :num_values],
                                            self.md_profile_dimension_right[dim][:, :num_values])

    def _update_matrix_profile(self, new_distances, new_distance_indices, new_distance_dimensions,
                               matrix_profile, matrix_profile_index, matrix_profile_dims):
        update_pos = new_distances < matrix_profile
        matrix_profile[update_pos] = new_distances[update_pos]
        matrix_profile_index[update_pos] = new_distance_indices[update_pos]
        matrix_profile_dims[:, update_pos] = new_distance_dimensions[:, update_pos]

    def process_column(self, column_index, values):
        n_dim, num_values = values.shape

        values_sorted = np.sort(values, axis=0)
        values_cumsum = np.zeros(num_values)

        for dim in range(n_dim):
            values_cumsum += values_sorted[dim, :]
            min_position_l = np.argmin(values_cumsum[:column_index + 1])
            self.md_matrix_profile_left[dim, column_index] = values_cumsum[min_position_l] / (dim + 1)
            self.md_profile_index_left[dim, column_index] = min_position_l
            self.md_profile_dimension_left[dim][:, column_index] = np.argsort(values[:, min_position_l])[:dim + 1]

            # Check if column crosses into the lower triangle of the distance matrix
            if num_values >= column_index + 2:
                min_position_r = np.argmin(values_cumsum[column_index + 1:]) + column_index + 1
                self.md_matrix_profile_right[dim, column_index] = values_cumsum[min_position_r] / (dim + 1)
                self.md_profile_index_right[dim, column_index] = min_position_r
                self.md_profile_dimension_right[dim][:, column_index] = np.argsort(values[:, min_position_r])[:dim + 1]

    def md_matrix_profile(self):
        """
        Merges the left and right multidimensional matrix profile, to create the multidimensional matrix profile.
        :return: ndarray of shape (num_dimensions, num_subsequences)
        """
        left_best = self.md_matrix_profile_left < self.md_matrix_profile_right
        return np.where(
            left_best,
            self.md_matrix_profile_left,
            self.md_matrix_profile_right
        )

    def md_profile_index(self):
        """
        Merges the left and right multidimensional matrix profile index, to create the multidimensional matrix profile
        index.
        :return: ndarray of shape (num_dimensions, num_subsequences)
        """
        left_best = self.md_matrix_profile_left < self.md_matrix_profile_right
        return np.where(
            left_best,
            self.md_profile_index_left,
            self.md_profile_index_right
        )

    def md_profile_dimensions(self):
        """
        Merges the left and right dimensions, to create the dimensions for the multidimensional matrix profile.
        :return: list of length num_dimensions, where the entry at index i is an ndarray of shape
        (i+1, num_subsequences).
        """
        profile_dimension = [np.full((i + 1, self._num_subseq), -1, dtype=np.int) for i in range(self._n_dim)]

        for dim in range(self._n_dim):
            left_best = self.md_matrix_profile_left[dim, :] < self.md_matrix_profile_right[dim, :]
            profile_dimension[dim] = np.where(
                left_best,
                self.md_profile_dimension_left[dim],
                self.md_profile_dimension_right[dim]
            )

        return profile_dimension
