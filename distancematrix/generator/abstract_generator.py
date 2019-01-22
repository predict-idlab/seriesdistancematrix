from abc import ABC, abstractmethod


class AbstractGenerator(ABC):
    @abstractmethod
    def prepare(self, m, series, query=None):
        """
        Create a bound non-streaming generator for the given series and query sequences.

        :param m: the size of the subsequences used to calculate distances between series and query
        :param series: 1D array, used as the horizontal axis of a distance matrix
        :param query: 1D array, used as the vertical axis of a distance matrix, or None to indicate a self-join
        :return: a bound generator
        """
        pass

    @abstractmethod
    def prepare_streaming(self, m, series_window, query_window=None):
        """
        Create a bound generator that supports streaming data.
        The generator will need to receive data before any distances can be calculated.

        :param m: the size of the subsequences used to calculate distances between series and query
        :param series_window: number of values to keep in memory for series, the length of the
          horizontal axis of the distance matrix will be equal to (series_window - m + 1)
        :param query_window: number of values to keep in memory for query, the length of the
          vertical axis of the distance matrix will be equal to (query_window - m + 1),
          or None to indicate a self-join.
        :return: a bound generator that supports streaming
        """
        pass


class AbstractBoundGenerator(ABC):
    @abstractmethod
    def calc_diagonal(self, diag):
        """
        Calculates all distances of the distance matrix diagonal with the given index.

        If diag is zero, this calculates the main diagonal, running from the top left to the bottom right.
        Any positive value represents a diagonal above the main diagonal, and a negative value represents
        a diagonal below the main diagonal.

        :param diag: the diagonal index
        :return: 1D array, containing all values
        """
        pass

    @abstractmethod
    def calc_column(self, column):
        """
        Calculates all distances of the distance matrix on the specified column.

        :param column: the column index (starting at 0)
        :return: 1D array, containing all values
        """
        pass


class AbstractBoundStreamingGenerator(ABC):
    @abstractmethod
    def append_series(self, values):
        """
        Adds more data points to the series sequence (and the query in case of a self-join).
        Older data points will be dropped if the series would become larger than the foreseen capacity.

        :param values: 1D array, the new values to append to the series
        :return: None
        """

    @abstractmethod
    def append_query(self, values):
        """
        Adds more data points to the query sequence.
        Older data points will be dropped if the query would become larger than the foreseen capacity.

        :param values: 1D array, the new values to append to the query
        :return: None
        """