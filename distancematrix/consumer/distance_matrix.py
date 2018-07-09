import numpy as np
from ..util import diag_indices_of
from .abstract_consumer import AbstractConsumer


class DistanceMatrix(AbstractConsumer):
    def __init__(self):
        """
        Creates a new consumer that will store the complete distance matrix.

        Note that the distance matrix requires quadratic memory, so it is unsuited for long time series.
        """

        self.distance_matrix = None

    def initialise(self, series, query, m):
        if series.ndim != 1:
            raise RuntimeError("Series should be 1D")
        if query.ndim != 1:
            raise RuntimeError("Query should be 1D")

        n = len(series)
        q = len(query)

        self.distance_matrix = np.full((q-m+1, n-m+1), np.nan, dtype=np.float)

    def process_diagonal(self, diagonal_index, values):
        indices = diag_indices_of(self.distance_matrix, diagonal_index)
        self.distance_matrix[indices] = values

    def process_column(self, column_index, values):
        self.distance_matrix[:, column_index] = values
