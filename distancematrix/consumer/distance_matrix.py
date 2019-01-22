import numpy as np
from ..util import diag_indices_of
from .abstract_consumer import AbstractStreamingConsumer


class DistanceMatrix(AbstractStreamingConsumer):
    def __init__(self):
        """
        Creates a new consumer that will store the complete distance matrix.
        This consumer supports streaming.

        Note that the distance matrix requires quadratic memory, so it is unsuited for long time series.
        """

        self.distance_matrix = None

    def initialise(self, dims, query_subseq, series_subseq):
        if dims != 1:
            raise RuntimeError("Input should be 1D")

        self.distance_matrix = np.full((query_subseq, series_subseq), np.nan, dtype=np.float)

    def process_diagonal(self, diagonal_index, values):
        indices = diag_indices_of(self.distance_matrix, diagonal_index)
        self.distance_matrix[indices] = values

    def process_column(self, column_index, values):
        self.distance_matrix[:, column_index] = values

    def shift_series(self, amount):
        if amount == 0:
            return

        self.distance_matrix = np.roll(self.distance_matrix, -amount, axis=1)
        self.distance_matrix[:, -amount:] = np.nan

    def shift_query(self, amount):
        if amount == 0:
            return

        self.distance_matrix = np.roll(self.distance_matrix, -amount, axis=0)
        self.distance_matrix[-amount:, :] = np.nan
