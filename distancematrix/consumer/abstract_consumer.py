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

        The number of values on the diagonal might be less than the diagonal of the full matrix profile,
        this can occur when not enough data is available yet to calculate the entire distance matrix
        (typically for streaming when not enough data is available to fill the entire foreseen space).

        :param diagonal_index: index of the diagonal in range ]-num_query_subseq, num_series_subseq[,
            the main diagonal has index 0
        :param values: array of shape (num_dimensions, num_values_on_diagonal) containing the distances
        :return: None
        """
        pass

    @abstractmethod
    def process_column(self, column_index, values):
        """
        Method called when a column of the distance matrix is calculated.

        The number of values on the column might be less than the column of the full matrix profile,
        this can occur when not enough data is available yet to calculate the entire distance matrix
        (typically for streaming when not enough data is available to fill the entire foreseen space).

        :param column_index: index of the column, in range [0, series_subseq[
        :param values: array of shape (num_dimensions, num_values_on_column) containing the distances
        :return: None
        """
        pass


class AbstractStreamingConsumer(AbstractConsumer):
    @abstractmethod
    def shift_query(self, amount):
        """
        Inform the consumer that the distance matrix has shifted in the query direction.

        :param amount: amount of subsequences that were shifted
        :return: None
        """
        pass

    @abstractmethod
    def shift_series(self, amount):
        """
        Inform the consumer that the distance matrix has shifted in the series direction.

        :param amount: amount of subsequences that were shifted
        :return: None
        """
        pass
