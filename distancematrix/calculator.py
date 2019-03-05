import numpy as np
import time
import random
from collections import OrderedDict
from math import ceil
from abc import ABC, abstractmethod

from distancematrix.interrupt_util import interrupt_catcher
from distancematrix.util import diag_length


class AbstractCalculator(ABC):
    """
    Base class for calculators. A calculator is repsonsible for managing
    consumers and generators for a distance matrix calculation. It provides
    a single point of interaction for the user.

    In order to do useful work, generators and consumers need to be added.
    Generators will use the input query and series to form a distance matrix.
    Consumers process these values in a way that is useful.
    """
    def __init__(self, self_join, m, num_series_subseq, num_query_subseq, n_dim, trivial_match_buffer=None):
        """
        Initialises a new calculator (without any generators/consumers)

        :param self_join: indicates whether a self-join of the series is being performed,
          this can lead to faster results
        :param m: subsequence length to consider
        :param num_series_subseq: number of subsequences on the series axis (second dim of distance matrix)
        :param num_query_subseq: number of subsequences on the query axis (first dim of distance matrix)
        :param n_dim: number of data channels (dimensions) in series/query
        :param trivial_match_buffer: used only in case of a self-join, the number of values next to the main diagonal
        (of the distance matrix) to skip. If None, defaults to m/2. Any consumers will either not receive values
        (in case of diagonal calculation) or Infinity values (in case of column calculation).
        """
        self._self_join = self_join
        self.m = m
        self.n_dim = n_dim
        self.num_series_subseq = num_series_subseq

        if self_join:
            self.num_query_subseq = num_series_subseq

            if trivial_match_buffer is None:
                trivial_match_buffer = m // 2
            if trivial_match_buffer not in range(-1, self.num_series_subseq):
                raise RuntimeError("Invalid value for trivial_match_buffer: " + str(trivial_match_buffer))
            self.trivial_match_buffer = trivial_match_buffer
        else:
            self.num_query_subseq = num_query_subseq
            self.trivial_match_buffer = -1

        # Generators calculate distance values from the series and query
        self._generators = OrderedDict()
        # Consumers process the calculated distance values
        self._consumers = OrderedDict()

        # Tracking column calculations
        self._last_column_calculated = -1

    def add_consumer(self, generator_ids, consumer):
        """
        Adds a consumer that uses the distances calculated by the provided generators.

        :param generator_ids: list containing ids of the generators
        :param consumer: the consumer to add
        :return: the bound consumer
        """
        gen_dims = len(generator_ids)
        consumer.initialise(gen_dims, self.num_query_subseq, self.num_series_subseq)
        self._consumers[consumer] = generator_ids
        return consumer

    @abstractmethod
    def add_generator(self, input_dim, generator):
        """
        Adds a generator that will use the data from the specified channel (from series/query).

        :param input_dim: index of the data channel
        :param generator: the generator to add
        :return: the bound generator
        """
        pass

    def calculate_columns(self, start=None, upto=1., print_progress=False):
        """
        Calculates columns of the distance matrix. The calculator keeps track of the rightmost column that was
        already calculated and will use it as starting position unless the start position is provided.

        Note that the generators are optimised for calculating consecutive columns from left to right.

        :param start: int for absolute position, float for relative position. The first column to calculate.
        If None, continues from the rightmost column that was calculated so far.
        :param upto: int for absolute position, float for relative position. The last column (exclusive) to calculate.
        :param print_progress: print progress to console?
        :return: None
        """
        if start is None:
            start = self._last_column_calculated + 1
        column_limit = self._max_column()
        start = _ratio_to_int(start, self.num_series_subseq, column_limit)
        current_column = start
        upto = _ratio_to_int(upto, self.num_series_subseq, column_limit)

        generators = list(self._generators.keys())
        generators_needed_ids = list(set(id for id_list in self._consumers.values() for id in id_list))
        column_dists = np.full((len(self._generators), self.num_query_subseq), np.nan, dtype=np.float)

        start_time = time.time()

        with interrupt_catcher() as is_interrupted:
            while current_column < upto and not is_interrupted():
                for generator_id in generators_needed_ids:  # todo: parallel
                    generator = generators[generator_id]
                    column_dists[generator_id, :] = generator.calc_column(current_column)

                if self.trivial_match_buffer >= 0:
                    trivial_match_start = max(0, current_column - self.trivial_match_buffer)
                    trivial_match_end = current_column + self.trivial_match_buffer + 1
                    column_dists[:, trivial_match_start : trivial_match_end] = np.inf

                for consumer, generator_ids in self._consumers.items():  # todo: parallel
                    consumer.process_column(current_column, column_dists[generator_ids, :])

                self._last_column_calculated = max(current_column, self._last_column_calculated)
                current_column += 1

                if print_progress:
                    columns_calculated = current_column - start
                    columns_remaining = upto + 1 - current_column
                    print("\r{0:5.3f}% {1:10.1f} sec".format(
                        columns_calculated / (upto + 1 - start) * 100,
                        (time.time() - start_time) / columns_calculated * columns_remaining
                    ), end="")

    @property
    def num_dist_matrix_values(self):
        return self.num_query_subseq * self.num_series_subseq

    @property
    def generators(self):
        return list(self._generators.keys())

    @property
    def consumers(self):
        return list(self._consumers.keys())

    def _max_column(self):
        return self.num_series_subseq


class AnytimeCalculator(AbstractCalculator):
    """
    Calculator that allows approximate calculations in a fraction of the time, but does not support
    data streaming.

    A calculator is repsonsible for managing
    consumers and generators for a distance matrix calculation. It provides
    a single point of interaction for the user.
    """
    def __init__(self, m, series, query=None, trivial_match_buffer=None):
        """
        Creates a new calculator.

        :param m: subsequence length to consider
        :param series: 1D or 2D (dimensions x datapoints) array
        :param query: 1D or 2D (dimensions x datapoints) array, if None, a self-join on series is performed
        :param trivial_match_buffer: used only in case of a self-join, the number of values next to the main diagonal
        (of the distance matrix) to skip. If None, defaults to m/2. Any consumers will either not receive values
        (in case of diagonal calculation) or Infinity values (in case of column calculation).
        """
        self_join = query is None

        self.series = np.atleast_2d(series).astype(np.float, copy=True)
        if not self_join:
            self.query = np.atleast_2d(query).astype(np.float, copy=True)
        else:
            self.query = self.series

        if self.series.ndim != 2:
            raise RuntimeError("Series should be 1D or 2D ndarray.")
        if self.query.ndim != 2:
            raise RuntimeError("Query should be 1D or 2D ndarray.")

        n_dim = self.series.shape[0]
        if n_dim != self.query.shape[0]:
            raise RuntimeError("Dimensions of series and query do not match.")

        num_series_subseq = self.series.shape[1] - m + 1
        num_query_subseq = self.query.shape[1] - m + 1

        super().__init__(self_join, m, num_series_subseq, num_query_subseq, n_dim, trivial_match_buffer)
        # Code below depends on trivial match buffer being set.

        # Tracking diagonal calculations
        if not self_join:
            self._diagonal_calc_order = np.arange(-self.num_query_subseq + 1, self.num_series_subseq)
            self._diagonal_values_total = self.num_query_subseq * self.num_series_subseq
        else:
            self._diagonal_calc_order = np.arange(self.trivial_match_buffer + 1, self.num_series_subseq)
            # Upper half of a square with size a = a * (a+1) / 2
            temp = self.num_series_subseq - self.trivial_match_buffer - 1
            self._diagonal_values_total = temp * (temp + 1) // 2
        random.shuffle(self._diagonal_calc_order, random.Random(0).random)
        self._diagonal_calc_list_next_index = 0
        self._diagonal_values_calculated = 0
        self._diagonal_calc_time = 0

    def add_generator(self, input_dim, generator):
        if input_dim < 0 or input_dim >= self.n_dim:
            raise ValueError("Invalid input_dim, should be in range [0, %s]" % self.n_dim)

        if not self._self_join:
            bound_gen = generator.prepare(self.m, self.series[input_dim, :], self.query[input_dim, :])
        else:
            bound_gen = generator.prepare(self.m, self.series[input_dim, :])

        self._generators[bound_gen] = input_dim
        return bound_gen

    def calculate_diagonals(self, partial=1., print_progress=False):
        """
        Calculates diagonals of the distance matrix. The advantage of calculating diagonals is that values are spread
        over the entire distance matrix, which can provide a quick approximation for any consumer.

        :param partial: int for a number of values, float for relative number of values. The number of distance
        matrix values that should be calculated (including the counts of previous diagonals calulated).
        :param print_progress: print progress to the console
        :return: None
        """
        generators = list(self._generators.keys())
        generators_needed_ids = list(set(id for id_list in self._consumers.values() for id in id_list))
        max_diagonal_length = min(self.num_query_subseq, self.num_series_subseq)
        diag_dists = np.full((len(self._generators), max_diagonal_length), np.nan, dtype=np.float)

        values_needed = _ratio_to_int(partial, self._diagonal_values_total, self._diagonal_values_total)

        with interrupt_catcher() as is_interrupted:
            while self._diagonal_values_calculated < values_needed and not is_interrupted():
                start_time = time.time()

                # Diagonal: 0 is the main diagonal, 1 is one above the main diagonal, etc...
                diagonal = self._diagonal_calc_order[self._diagonal_calc_list_next_index]
                diagonal_length = diag_length(self.num_query_subseq, self.num_series_subseq, diagonal)
                diagonal_values = diag_dists[:, :diagonal_length]

                for generator_id in generators_needed_ids:  # todo: parallel
                    generator = generators[generator_id]
                    diagonal_values[generator_id, :] = generator.calc_diagonal(diagonal)

                for consumer, generator_ids in self._consumers.items():  # todo: parallel
                    values_to_consume = diagonal_values[generator_ids, :]
                    consumer.process_diagonal(diagonal, values_to_consume)
                    if self._self_join:
                        consumer.process_diagonal(-diagonal, values_to_consume)

                self._diagonal_values_calculated += int(diagonal_length)  # numpy.int32 to int
                self._diagonal_calc_list_next_index += 1

                self._diagonal_calc_time += time.time() - start_time
                if print_progress:
                    local_progress = self._diagonal_values_calculated / values_needed
                    global_progress = self._diagonal_values_calculated / self._diagonal_values_total
                    avg_time_per_value = self._diagonal_calc_time / self._diagonal_values_calculated
                    time_left = avg_time_per_value * (values_needed - self._diagonal_values_calculated)
                    print("\r{0:5.3f}% {1:10.1f} sec ({2:5.3f}% total)".
                          format(local_progress * 100, time_left, global_progress * 100), end="")


class StreamingCalculator(AbstractCalculator):
    """
    Calculator that allows streaming data, but does not support anytime calculations.

    A calculator is repsonsible for managing
    consumers and generators for a distance matrix calculation. It provides
    a single point of interaction for the user.
    """
    def __init__(self, m, series_window, query_window=None, n_dim=1, trivial_match_buffer=None):
        """
        Creates a new calculator.

        :param m: subsequence length to consider
        :param series_window: number of data points of series to keep in memory for calculations
        :param query_window: number of data points of series to keep in memory for calculations,
          or None to specify that a self-join over series should be performed
        :param n_dim: number of data channels that will be used
        :param trivial_match_buffer: used only in case of a self-join, the number of values next to the main diagonal
        (of the distance matrix) to skip. If None, defaults to m/2. Any consumers will either not receive values
        (in case of diagonal calculation) or Infinity values (in case of column calculation).
        """
        self.streamed_series_points = 0

        self_join = query_window is None

        num_series_subseq = series_window - m + 1
        if self_join:
            num_query_subseq = num_series_subseq
            self.streamed_query_points = -1
        else:
            num_query_subseq = query_window - m + 1
            self.streamed_query_points = 0

        super().__init__(self_join, m, num_series_subseq, num_query_subseq, n_dim, trivial_match_buffer)

    def add_generator(self, input_dim, generator):
        if input_dim < 0 or input_dim >= self.n_dim:
            raise ValueError("Invalid input_dim, should be in range [0, %s]" % self.n_dim)

        if not self._self_join:
            bound_gen = generator.prepare_streaming(self.m,
                                                    self.num_series_subseq + self.m - 1,
                                                    self.num_query_subseq + self.m - 1)
        else:
            bound_gen = generator.generator.prepare_streaming(self.m,
                                                              self.num_series_subseq + self.m - 1,
                                                              self.num_query_subseq + self.m - 1)

        self._generators[bound_gen] = input_dim
        return bound_gen

    def append_series(self, values):
        """
        Add more data points to series.

        As a side effect, the last calculated column index is shifted along with the data.

        :param values: 1D array for one data point on each channel, or 2D array of shape (num_dim, num_points)
        :return: None
        """
        values = np.asarray(values)

        if values.ndim == 1:
            if len(values) == self.n_dim:
                values = values.reshape((self.n_dim, 1))
            else:
                raise RuntimeError("Expected {dim} values, but received {len}".format(dim=self.n_dim, len=len(values)))
        elif values.ndim != 2 or values.shape[0] != self.n_dim:
            raise RuntimeError("Provided values do not match shape ({dim}, x)".format(dim=self.n_dim))

        for gen in self.generators:
            input_dim = self._generators[gen]
            gen.append_series(values[input_dim, :])

        self.streamed_series_points += values.shape[1]
        column_shift = max(0, self.streamed_series_points - max(self.num_series_subseq + self.m - 1,
                                                                self.streamed_series_points - values.shape[1]))

        if column_shift:
            for cons in self.consumers:
                cons.shift_series(column_shift)
            self._last_column_calculated = min(-1, self._last_column_calculated - column_shift)

    def append_query(self, values):
        """
        Add more data points to query. Cannot be used if performing a self join.

        Note that appending query data does not adjust the last column calculated index.

        :param values: 1D array for one data point on each channel, or 2D array of shape (num_dim, num_points)
        :return: None
        """
        if self._self_join:
            raise RuntimeError("Cannot append to query when performing a self-join on series.")

        values = np.asarray(values)

        if values.ndim == 1:
            if len(values) == self.n_dim:
                values = values.reshape((self.n_dim, 1))
            else:
                raise RuntimeError("Expected {dim} values, but received {len}".format(dim=self.n_dim, len=len(values)))
        elif values.ndim != 2 or values.shape[0] != self.n_dim:
            raise RuntimeError("Provided values do not match shape ({dim}, x)".format(dim=self.n_dim))
        values = np.atleast_2d(values)

        for gen in self.generators:
            input_dim = self._generators[gen]
            gen.append_query(values[input_dim, :])

        for cons in self.consumers:
            cons.shift_query(values.shape[1])

    def _max_column(self):
        return min(max(0, self.streamed_series_points - self.m + 1), self.num_series_subseq)


def _ratio_to_int(ratio_or_result, full, max_value):
    if isinstance(ratio_or_result, (float, np.floating)):
        if ratio_or_result < 0 or ratio_or_result > 1:
            raise ValueError("Value should be in range [0, 1].")

        return min(max(0, ceil(ratio_or_result * full)), max_value)

    if isinstance(ratio_or_result, (int, np.integer)):
        return min(max(0, ratio_or_result), max_value)

    raise RuntimeError("Invalid type, should be int or float.")


def _find_invalid_subseq_idxs(invalid_data_idxs, dims, m, min_result_idx, max_result_idx):
    """
    Given indices of invalid data points, calculate the list of indices of affected subsequences.

    :param invalid_data_idxs: 2-element tuple of indices of invalid data points (ourput of np.nonzero)
    :param dims: number of data dimensions
    :param m: subsequence length
    :param min_result_idx: minimum allowed index in the result
    :param max_result_idx: maximum allowed index in the result
    :return: a list containg dims lists, each containing indices of subsequences affected by invalid data points
    """
    result_set = [set() for i in range(dims)]

    for dim, invalid_idx in np.transpose(invalid_data_idxs):
        affected_subsequences = range(max(min_result_idx, invalid_idx - m + 1, min(invalid_idx + m, max_result_idx)))
        result_set[dim].update(affected_subsequences)

    return [np.array(list(s), dtype=np.int64) for s in result_set]
