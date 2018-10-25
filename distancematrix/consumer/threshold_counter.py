import numpy as np

from .abstract_consumer import AbstractConsumer


class ThresholdCounter(AbstractConsumer):
    """
    Consumer that counts the number of values in each column of the distancematrix that are below
    or equal to specified thresholds.

    This consumer counts values as they are passed and does not extrapolate or keep information about which
    values were already counted. Specifically: partial calculations will result in counts of the produced values,
    and passing the same diagonals multiple time could result in double counts.
    """

    def __init__(self, thresholds):
        """
        Creates a new counter.

        :param thresholds: scalar or 1D array of threshold values
        """
        self.thresholds = np.array(thresholds, ndmin=1, dtype=np.float)
        if self.thresholds.ndim != 1:
            raise ValueError('Thresholds should be scalar or one-dimensional.')
        self.counts = None

    def initialise(self, dims, query_subseq, series_subseq):
        self.counts = np.full((len(self.thresholds), series_subseq), 0, dtype=np.int)

    def process_diagonal(self, diag, values):
        values = values[0]
        num_values = len(values)

        if diag >= 0:
            self.counts[:, diag:diag + num_values] += values <= self.thresholds[:, None]
        else:
            self.counts[:, :num_values] += values <= self.thresholds[:, None]

    def process_column(self, column_index, values):
        values = values[0]

        self.counts[:, column_index] = np.count_nonzero(values <= self.thresholds[:, None], axis=1)
