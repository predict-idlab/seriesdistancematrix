from abc import ABC, abstractmethod


class AbstractConsumer(ABC):
    @abstractmethod
    def initialise(self, series, query, m):
        """
        Initialise this consumer using the given series, query and subsequence length.

        :param series: series to be processed, dimension (num_dimensions, num_data_points)
        :param m: length of subsequences that will be considered
        :return: None
        """
        pass

    @abstractmethod
    def process_diagonal(self, diagonal_index, values):
        """
        Method called when a diagonal of the distance matrix is calculated.

        :param diagonal_index: index of the diagonal, main diagonal has index 0
        :param values: array of shape (num_dimensions, num_values_on_diagonal) containing the distances
        :return: None
        """
        pass

    @abstractmethod
    def process_column(self, column_index, values):
        """
        Method called when a column of the distance matrix is calculated.

        :param column_index: index of the column
        :param values: array of shape (num_dimensions, num_values_on_column) containing the distances
        :return: None
        """
        pass
