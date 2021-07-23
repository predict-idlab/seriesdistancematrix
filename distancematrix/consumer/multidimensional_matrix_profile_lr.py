import numpy as np
from distancematrix.ringbuffer import RingBuffer

from .abstract_consumer import AbstractStreamingConsumer


class MultidimensionalMatrixProfileLR(AbstractStreamingConsumer):
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

    This consumer supports streaming.
    """

    def __init__(self, rb_scale_factor=2.):
        """
        Creates a new consumer that calculates the left and right matrix profile, the corresponding
        indices and the used dimensions over multiple dimensions (data channels).

        :param rb_scale_factor: scaling factor used for RingBuffers in case of streaming data (should be >= 1),
            this allows choosing a balance between less memory (low values) and reduced data copying (higher values)
        """

        if rb_scale_factor < 1.:
            raise ValueError("rb_scale_factor should be >= 1, it was: " + str(rb_scale_factor))

        self._num_subseq = None
        self._range = None
        self._n_dim = None

        self._md_matrix_profile_left = None
        self._md_profile_index_left = None
        self._md_profile_dimension_left = None

        self._md_matrix_profile_right = None
        self._md_profile_index_right = None
        self._md_profile_dimension_right = None

        self._series_shift = 0
        self._query_shift = 0

        self._rb_scale_factor = rb_scale_factor

    def initialise(self, dims, query_subseq, series_subseq):
        self._n_dim = dims
        self._num_subseq = series_subseq
        self._range = RingBuffer(np.arange(0, self._num_subseq, dtype=int),
                                 scaling_factor=self._rb_scale_factor)

        self._md_matrix_profile_left = RingBuffer(np.full((dims, self._num_subseq), np.inf, dtype=float),
                                                  scaling_factor=self._rb_scale_factor)
        self._md_profile_index_left = RingBuffer(np.full((dims, self._num_subseq), -1, dtype=int),
                                                 scaling_factor=self._rb_scale_factor)
        self._md_profile_dimension_left = \
            [RingBuffer(np.full((i + 1, self._num_subseq), -1, dtype=int),
                        scaling_factor=self._rb_scale_factor) for i in range(dims)]

        self._md_matrix_profile_right = RingBuffer(np.full((dims, self._num_subseq), np.inf, dtype=float),
                                                   scaling_factor=self._rb_scale_factor)
        self._md_profile_index_right = RingBuffer(np.full((dims, self._num_subseq), -1, dtype=int),
                                                  scaling_factor=self._rb_scale_factor)
        self._md_profile_dimension_right = \
            [RingBuffer(np.full((i + 1, self._num_subseq), -1, dtype=int),
                        scaling_factor=self._rb_scale_factor) for i in range(dims)]

    def process_diagonal(self, diag, values):
        n_dim, num_values = values.shape
        shift_diff = self._series_shift - self._query_shift

        values_sort_order = np.argsort(values, axis=0)
        values_sorted = np.sort(values, axis=0)
        values_cumsum = np.zeros(num_values)

        if diag + shift_diff >= 0:
            # left MP
            if diag >= 0:
                for dim in range(n_dim):
                    values_cumsum += values_sorted[dim, :]
                    values_mean_over_dim = values_cumsum / (dim + 1)

                    self._update_matrix_profile(values_mean_over_dim,
                                                self._range[:num_values],
                                                values_sort_order[:dim + 1, :],
                                                self._md_matrix_profile_left[dim, diag:diag + num_values],
                                                self._md_profile_index_left[dim, diag:diag + num_values],
                                                self._md_profile_dimension_left[dim][:, diag:diag + num_values])
            else:
                for dim in range(n_dim):
                    values_cumsum += values_sorted[dim, :]
                    values_mean_over_dim = values_cumsum / (dim + 1)

                    self._update_matrix_profile(values_mean_over_dim,
                                                self._range[-diag:-diag + num_values],
                                                values_sort_order[:dim + 1, :],
                                                self._md_matrix_profile_left[dim, :num_values],
                                                self._md_profile_index_left[dim, :num_values],
                                                self._md_profile_dimension_left[dim][:, :num_values])
        else:
            # right MP
            if diag >= 0:
                for dim in range(n_dim):
                    values_cumsum += values_sorted[dim, :]
                    values_mean_over_dim = values_cumsum / (dim + 1)

                    self._update_matrix_profile(values_mean_over_dim,
                                                self._range[num_values],
                                                values_sort_order[:dim + 1, :],
                                                self._md_matrix_profile_right[dim, diag:diag + num_values],
                                                self._md_profile_index_right[dim, diag:diag + num_values],
                                                self._md_profile_dimension_right[dim][:, diag:diag + num_values])
            else:
                for dim in range(n_dim):
                    values_cumsum += values_sorted[dim, :]
                    values_mean_over_dim = values_cumsum / (dim + 1)

                    self._update_matrix_profile(values_mean_over_dim,
                                                self._range[-diag:-diag + num_values],
                                                values_sort_order[:dim + 1, :],
                                                self._md_matrix_profile_right[dim, :num_values],
                                                self._md_profile_index_right[dim, :num_values],
                                                self._md_profile_dimension_right[dim][:, :num_values])

        if diag >= 0:
            for dim in range(n_dim):
                values_cumsum += values_sorted[dim, :]
                values_mean_over_dim = values_cumsum / (dim + 1)

                self._update_matrix_profile(values_mean_over_dim,
                                            self._range[:num_values],
                                            values_sort_order[:dim + 1, :],
                                            self._md_matrix_profile_left[dim, diag:diag + num_values],
                                            self._md_profile_index_left[dim, diag:diag + num_values],
                                            self._md_profile_dimension_left[dim][:, diag:diag + num_values])

        else:
            for dim in range(n_dim):
                values_cumsum += values_sorted[dim, :]
                values_mean_over_dim = values_cumsum / (dim + 1)

                self._update_matrix_profile(values_mean_over_dim,
                                            self._range[-diag:-diag + num_values],
                                            values_sort_order[:dim + 1, :],
                                            self._md_matrix_profile_right[dim, :num_values],
                                            self._md_profile_index_right[dim, :num_values],
                                            self._md_profile_dimension_right[dim][:, :num_values])

    def _update_matrix_profile(self, new_distances, new_distance_indices, new_distance_dimensions,
                               matrix_profile, matrix_profile_index, matrix_profile_dims):
        update_pos = new_distances < matrix_profile
        matrix_profile[update_pos] = new_distances[update_pos]
        matrix_profile_index[update_pos] = new_distance_indices[update_pos]
        matrix_profile_dims[:, update_pos] = new_distance_dimensions[:, update_pos]

    def process_column(self, column_index, values):
        n_dim, num_values = values.shape
        shift_diff = self._series_shift - self._query_shift

        border = max(0, column_index + 1 + shift_diff)

        values_sorted = np.sort(values, axis=0)
        values_cumsum = np.zeros(num_values)

        for dim in range(n_dim):
            values_cumsum += values_sorted[dim, :]

            if border > 0:
                min_position_l = np.argmin(values_cumsum[:border])
                new_min_value = values_cumsum[min_position_l] / (dim + 1)

                if new_min_value < self._md_matrix_profile_left[dim, column_index]:
                    self._md_matrix_profile_left[dim, column_index] = new_min_value
                    self._md_profile_index_left[dim, column_index] = min_position_l + self._query_shift
                    self._md_profile_dimension_left[dim][:, column_index] =\
                        np.argsort(values[:, min_position_l])[:dim + 1]

            # Check if column crosses into the lower triangle of the distance matrix
            if num_values > border:
                min_position_r = np.argmin(values_cumsum[border:]) + border
                new_min_value = values_cumsum[min_position_r] / (dim + 1)

                # In case of shifting, a lower value could already be present
                if new_min_value < self._md_matrix_profile_right[dim, column_index]:
                    self._md_matrix_profile_right[dim, column_index] = new_min_value
                    self._md_profile_index_right[dim, column_index] = min_position_r + self._query_shift
                    self._md_profile_dimension_right[dim][:, column_index] =\
                        np.argsort(values[:, min_position_r])[:dim + 1]

    def shift_query(self, amount):
        if amount == 0:
            return

        self._query_shift += amount
        self._range.push(np.arange(self._range[-1] + 1, self._range[-1] + 1 + amount))

    def shift_series(self, amount):
        if amount == 0:
            return

        self._series_shift += amount

        push_values = np.full((self._n_dim, amount), np.inf)
        self._md_matrix_profile_left.push(push_values)
        self._md_matrix_profile_right.push(push_values)

        push_values[:] = -1
        self._md_profile_index_left.push(push_values)
        self._md_profile_index_right.push(push_values)

        for dim in range(self._n_dim):
            self._md_profile_dimension_left[dim].push(push_values[:dim + 1, :])
            self._md_profile_dimension_right[dim].push(push_values[:dim + 1, :])

    def md_matrix_profile(self):
        """
        Merges the left and right multidimensional matrix profile, to create the multidimensional matrix profile.
        :return: ndarray of shape (num_dimensions, num_subsequences)
        """
        left_best = self._md_matrix_profile_left.view < self._md_matrix_profile_right.view
        return np.where(
            left_best,
            self._md_matrix_profile_left.view,
            self._md_matrix_profile_right.view
        )

    def md_profile_index(self):
        """
        Merges the left and right multidimensional matrix profile index, to create the multidimensional matrix profile
        index.
        :return: ndarray of shape (num_dimensions, num_subsequences)
        """
        left_best = self._md_matrix_profile_left.view < self._md_matrix_profile_right.view
        return np.where(
            left_best,
            self._md_profile_index_left.view,
            self._md_profile_index_right.view
        )

    def md_profile_dimensions(self):
        """
        Merges the left and right dimensions, to create the dimensions for the multidimensional matrix profile.
        :return: list of length num_dimensions, where the entry at index i is an ndarray of shape
        (i+1, num_subsequences).
        """
        profile_dimension = [np.full((i + 1, self._num_subseq), -1, dtype=int) for i in range(self._n_dim)]

        for dim in range(self._n_dim):
            left_best = self._md_matrix_profile_left[dim, :] < self._md_matrix_profile_right[dim, :]
            profile_dimension[dim] = np.where(
                left_best,
                self._md_profile_dimension_left[dim].view,
                self._md_profile_dimension_right[dim].view
            )

        return profile_dimension

    @property
    def md_matrix_profile_left(self):
        return self._md_matrix_profile_left.view

    @property
    def md_matrix_profile_right(self):
        return self._md_matrix_profile_right.view

    @property
    def md_profile_index_left(self):
        return self._md_profile_index_left.view

    @property
    def md_profile_index_right(self):
        return self._md_profile_index_right.view

    @property
    def md_profile_dimension_left(self):
        return [buffer.view for buffer in self._md_profile_dimension_left]

    @property
    def md_profile_dimension_right(self):
        return [buffer.view for buffer in self._md_profile_dimension_right]
