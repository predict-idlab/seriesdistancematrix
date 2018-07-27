from abc import ABC, abstractmethod


class AbstractConsumer(ABC):
    @abstractmethod
    def initialise(self, dims, query_subseq, series_subseq):
        """
        Initialise this consumer.

        :param dims: the number of dimensions (data channels) this consumer will receive
        :param query_subseq: the number of query subsequences (rows in the distance matrix)
        :param series_subseq: the number of series subsequences (column in the distance matrix)
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
