import numpy as np
from .abstract_consumer import AbstractConsumer


class MatrixProfileLR(AbstractConsumer):
    """
    Consumer of distance matrix information to form the left and right matrix profile and their corresponding
    indices. The left matrix profile and index is a one dimensional series
    where each each value at index i contains the distance and index of the best match whose index is equal to or less
    than i. The right matrix profile and index contain the best match whose index is larger than i.
    """

    def __init__(self):
        """
        Creates a new consumer that calculates the left and right matrix profile, and corresponding
        indices over a single dimension.
        """
        self._num_subseq = None
        self._range = None

        self.matrix_profile_left = None
        self.profile_index_left = None
        self.matrix_profile_right = None
        self.profile_index_right = None

    def initialise(self, dims, query_subseq, series_subseq):
        self._num_subseq = series_subseq
        self._range = np.arange(0, max(series_subseq, query_subseq), dtype=np.int)

        self.matrix_profile_left = np.full(self._num_subseq, np.inf, dtype=np.float)
        self.profile_index_left = np.full(self._num_subseq, -1, dtype=int)
        self.matrix_profile_right = np.full(self._num_subseq, np.inf, dtype=np.float)
        self.profile_index_right = np.full(self._num_subseq, -1, dtype=int)

    def process_diagonal(self, diag, values):
        num_values = len(values)

        if diag >= 0:
            self._update_matrix_profile(
                values,
                self._range[:num_values],
                self.matrix_profile_left[diag:diag + num_values],
                self.profile_index_left[diag:diag + num_values])
        else:
            self._update_matrix_profile(
                values,
                self._range[-diag:-diag + num_values],
                self.matrix_profile_right[:num_values],
                self.profile_index_right[:num_values])

    def process_column(self, column_index, values):
        self.matrix_profile_left[column_index] = np.min(values[:column_index + 1])
        self.profile_index_left[column_index] = np.argmin(values[:column_index + 1])

        if len(values) >= column_index + 2:
            self.matrix_profile_right[column_index] = np.min(values[column_index + 1:])
            self.profile_index_right[column_index] = np.argmin(values[column_index + 1:]) + column_index + 1

    def _update_matrix_profile(self, dist_profile, dist_profile_idx,
                               matrix_profile, matrix_profile_index):
        update_pos = dist_profile < matrix_profile
        matrix_profile[update_pos] = dist_profile[update_pos]
        matrix_profile_index[update_pos] = dist_profile_idx[update_pos]

    def matrix_profile(self):
        """
        Creates the matrix profile based on the left and right matrix profile.

        :return: 1D array
        """
        matrix_profile = np.full(self.matrix_profile_left.shape, np.inf, dtype=np.float)

        left_best = self.matrix_profile_left < self.matrix_profile_right
        matrix_profile[left_best] = self.matrix_profile_left[left_best]
        matrix_profile[~left_best] = self.matrix_profile_right[~left_best]

        return matrix_profile

    def profile_index(self):
        """
        Creates the matrix profile index based on the left and right indices.

        :return: 1D array
        """
        profile_index = np.full(self.matrix_profile_left.shape, -1, dtype=int)

        left_best = self.matrix_profile_left < self.matrix_profile_right
        profile_index[left_best] = self.profile_index_left[left_best]
        profile_index[~left_best] = self.profile_index_right[~left_best]

        return profile_index


class MatrixProfileLRReservoir(AbstractConsumer):
    """
    Consumer of distance matrix information to form the left and right matrix profile and their corresponding
    indices. The left matrix profile and index is a one dimensional series
    where each each value at index i contains the distance and index of the best match whose index is less
    than i. The right matrix profile and index contain the best match whose index is equal to or larger than i.

    This consumer differs from the regular MatrixProfileLR consumer in that it uses reservoir sampling to determine
    the matrix profile indices. This means that if multiple values in the distance matrix column have the exact
    same distance, a random index will be stored in the matrix profile index.
    """

    def __init__(self, random_seed=None):
        """
        Creates a new consumer that calculates the left and right matrix profile, and corresponding
        indices over a single dimension.

        :param random_seed: seed to make behavior deterministic
        """
        self._num_subseq = None
        self._range = None
        self._rand = np.random.RandomState(seed=random_seed)

        self.matrix_profile_left = None
        self.profile_index_left = None
        self.num_matches_left = None
        self.matrix_profile_right = None
        self.profile_index_right = None
        self.num_matches_right = None

    def initialise(self, dims, query_subseq, series_subseq):
        self._num_subseq = series_subseq
        self._range = np.arange(0, max(series_subseq, query_subseq), dtype=np.int)

        self.matrix_profile_left = np.full(self._num_subseq, np.inf, dtype=np.float)
        self.profile_index_left = np.full(self._num_subseq, -1, dtype=np.int)
        self.num_matches_left = np.full(self._num_subseq, 0, dtype=np.int)
        self.matrix_profile_right = np.full(self._num_subseq, np.inf, dtype=np.float)
        self.profile_index_right = np.full(self._num_subseq, -1, dtype=np.int)
        self.num_matches_right = np.full(self._num_subseq, 0, dtype=np.int)

    def process_diagonal(self, diag, values):
        num_values = len(values)

        if diag >= 0:
            self._update_matrix_profile(
                values,
                self._range[:num_values],
                self.matrix_profile_left[diag:diag + num_values],
                self.profile_index_left[diag:diag + num_values],
                self.num_matches_left[diag:diag + num_values])
        else:
            self._update_matrix_profile(
                values,
                self._range[-diag:-diag + num_values],
                self.matrix_profile_right[:num_values],
                self.profile_index_right[:num_values],
                self.num_matches_right[:num_values])

    def process_column(self, column_index, values):
        min_dist = np.min(values[:column_index + 1])
        self.matrix_profile_left[column_index] = min_dist
        self.profile_index_left[column_index] = self._random_pick(np.nonzero(values[:column_index + 1] == min_dist)[0])

        if len(values) >= column_index + 2:
            min_dist = np.min(values[column_index + 1:])
            self.matrix_profile_right[column_index] = min_dist
            self.profile_index_right[column_index] = \
                self._random_pick(np.nonzero(values[column_index + 1:] == min_dist)[0] + column_index + 1)

    def _update_matrix_profile(self, dist_profile, dist_profile_idx,
                               matrix_profile, matrix_profile_index, num_matches):
        better = dist_profile < matrix_profile
        equal = np.logical_and(dist_profile == matrix_profile, np.isfinite(dist_profile))

        matrix_profile[better] = dist_profile[better]
        matrix_profile_index[better] = dist_profile_idx[better]
        num_matches[better] = 1

        # Reservoir sampling
        num_matches[equal] += 1
        matrix_profile_index[equal] = np.where(
            self._rand.rand(np.count_nonzero(equal)) < 1 / num_matches[equal],
            dist_profile_idx[equal],
            matrix_profile_index[equal]
        )

    def _random_pick(self, a):
        if len(a) == 1:
            return a[0]
        else:
            return a[self._rand.randint(0, len(a))]

    def matrix_profile(self):
        """
        Creates the matrix profile based on the left and right matrix profile.

        :return: 1D array
        """
        matrix_profile = np.full(self.matrix_profile_left.shape, np.inf, dtype=np.float)

        left_best = self.matrix_profile_left < self.matrix_profile_right
        matrix_profile[left_best] = self.matrix_profile_left[left_best]
        matrix_profile[~left_best] = self.matrix_profile_right[~left_best]

        return matrix_profile

    def profile_index(self):
        """
        Creates the matrix profile index based on the left and right indices.

        :return: 1D array
        """
        profile_index = np.full(self.matrix_profile_left.shape, -1, dtype=int)

        left_best = self.matrix_profile_left < self.matrix_profile_right
        profile_index[left_best] = self.profile_index_left[left_best]
        profile_index[~left_best] = self.profile_index_right[~left_best]

        value_equal = np.logical_and(self.matrix_profile_left == self.matrix_profile_right, self.num_matches_left)
        odds_left = self.num_matches_left[value_equal] / (
            self.num_matches_left[value_equal] + self.num_matches_right[value_equal])
        profile_index[value_equal] = np.where(
            self._rand.rand(np.count_nonzero(value_equal)) < odds_left,
            self.profile_index_left[value_equal],
            self.profile_index_right[value_equal]
        )

        return profile_index
