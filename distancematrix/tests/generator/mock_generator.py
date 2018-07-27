from distancematrix.util import diag_indices_of


class MockGenerator(object):
    """
    Mock generator for testing purposes. Simply returns distances from a given distance matrix.
    """

    def __init__(self, dist_matrix):
        """
        Creates a new mock generator that will return distances from the provided distance matrix.

        :param dist_matrix: distances to return.
        """
        self.series = None
        self.query = None
        self.m = None

        self.dist_matrix = dist_matrix

    def prepare(self, series, query, m):
        self.series = series
        self.query = query
        self.m = m

        num_series_subseq = len(series) - m + 1
        num_query_subseq = len(query) - m + 1

        if self.dist_matrix.shape[0] != num_query_subseq:
            raise RuntimeError("Mismatch for query dimensions.")
        if self.dist_matrix.shape[1] != num_series_subseq:
            raise RuntimeError("Mismatch for series dimensions.")

    def calc_diagonal(self, diag):
        return self.dist_matrix[diag_indices_of(self.dist_matrix, diag)]

    def calc_column(self, column):
        return self.dist_matrix[:, column]
