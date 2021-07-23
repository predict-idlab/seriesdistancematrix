from typing import Union, List

import numpy as np

from distancematrix.consumer.abstract_consumer import AbstractConsumer
from distancematrix.insights import lowest_value_idxs


class RadiusProfile0(AbstractConsumer):
    """
    Consumer that calculates (common-k) radius profiles.

    The (common-k) radius profile tracks the distance between each subsequence and its k-th best match.
    It can be used to find subsequences with at least k repetitions (so called common motifs).

    This class has been optimised for finding matches without ignoring trivial matches.
    In other words, it is not possible to define an exclusion zone for the matches.
    """
    def __init__(self, track_indices):
        """
        Creates a new radius profile consumer that tracks the distance between each subsequence and its
        k-th best matches.

        Note that the resulting radius profile will contain distances as if the given track_indices were sorted.

        :param track_indices: values of k to track
        """
        self.track_indices = np.array(track_indices, ndmin=1, dtype=int)

        if self.track_indices.ndim != 1:
            raise ValueError('Track_indices should be scalar or one-dimensional.')
        if len(self.track_indices) == 0:
            raise ValueError('At least one track index needed.')
        if np.any(self.track_indices < 0):
            raise ValueError('Only positive track_indices allowed.')

        self.track_indices.sort()
        self.values = None

    def initialise(self, dims, query_subseq, series_subseq):
        self.values = np.full((len(self.track_indices), series_subseq), np.nan, dtype=float)

    def process_diagonal(self, diag, values):
        raise NotImplementedError

    def process_column(self, column_index, values):
        values = values[0]

        sorted_values = np.empty(len(values) + 1, dtype=float)
        sorted_values[:-1] = np.sort(values)
        sorted_values[-1] = np.nan

        self.values[:, column_index] = np.take(sorted_values, self.track_indices, mode="clip")


class RadiusProfile(AbstractConsumer):
    """
    Consumer that calculates (common-k) radius profiles.

    The (common-k) radius profile tracks the distance between each subsequence and its k-th best match.
    It can be used to find subsequences with at least k repetitions (so called common motifs).
    """
    def __init__(self, track_indices: Union[int, List[int]], exclude_distance: int):
        """
        Creates a new radius profile consumer that tracks the distance between each subsequence and its
        k-th best matches.

        Note that the resulting radius profile will contain distances as if the given track_indices were sorted.

        .. seealso:: If excludedistance is zero,
         consider using :class:`distancematrix.consumer.radius_profile.RadiusProfile0`

        :param track_indices: values of k to track
        :param exclude_distance: trivial match exclusion distance, typical subsequence length / 2.
        """
        self.track_indices = np.array(track_indices, ndmin=1, dtype=int)

        if self.track_indices.ndim != 1:
            raise ValueError('Track_indices should be scalar or one-dimensional.')
        if len(self.track_indices) == 0:
            raise ValueError('At least one track index needed.')
        if np.any(self.track_indices < 0):
            raise ValueError('Only positive track_indices allowed.')
        if type(exclude_distance) is not int or exclude_distance < 0:
            raise RuntimeError('Exclude distance should be positive integer.')

        self.track_indices.sort()
        self.exclusion = exclude_distance
        self.values = None

    def initialise(self, dims, query_subseq, series_subseq):
        self.values = np.full((len(self.track_indices), series_subseq), np.nan, dtype=float)

    def process_diagonal(self, diag, values):
        raise NotImplementedError

    def process_column(self, column_index, values):
        values = values[0]

        iterator = lowest_value_idxs(values, self.exclusion)
        tracker_idx = 0

        # Iterate from best match to worst, ignoring trivial matches
        for i, low_value_idx in enumerate(iterator):
            # If we are interested in the i-th match
            if i == self.track_indices[tracker_idx]:
                self.values[tracker_idx, column_index] = values[low_value_idx]
                tracker_idx += 1

                # Abort if we found all matches we are tracking
                if tracker_idx >= len(self.track_indices):
                    return

        return
