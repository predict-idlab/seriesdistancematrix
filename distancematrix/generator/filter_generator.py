import numpy as np

from distancematrix.util import sliding_window_view
from distancematrix.ringbuffer import RingBuffer
from distancematrix.generator.abstract_generator import AbstractGenerator
from distancematrix.generator.abstract_generator import AbstractBoundGenerator
from distancematrix.generator.abstract_generator import AbstractBoundStreamingGenerator


def is_not_finite(data, subseq_length):
    """
    Marks infinite or nan values as invalid.
    """
    return ~np.isfinite(data)


class FilterGenerator(AbstractGenerator):
    def __init__(self, generator, invalid_data_function=is_not_finite):
        """
        Creates a new generator by wrapping another generator.

        :param generator: the generator whose results and input data will be filtered
        :param invalid_data_function: a function that takes in the original data (series or query) and
           subsequence length and returns a boolean array of the same size that has a True value for any invalid values.
           These values will be replaced by zeros before reaching the wrapped generator. Any distance values
           that were calculated using invalid data points will be positive infinite values.
        :param invalid_subseq_function: optional - a function that takes in the original data (series or query) and
           subsequence length and returns a boolean array of size matching the number of subsequences that has
           a True value for any invalid subsequence. Invalid subsequences will have positive infinte values
           as distance.
        """
        self._generator = generator
        self._invalid_data_function = invalid_data_function

    def prepare_streaming(self, m, series_window, query_window=None):
        gen = self._generator.prepare_streaming(m, series_window, query_window)

        num_s_subseq = series_window - m + 1
        if query_window is None:
            num_q_subseq = None
        else:
            num_q_subseq = query_window - m + 1

        return BoundStreamingFilterGenerator(gen, m, num_s_subseq, num_q_subseq, self._invalid_data_function)

    def prepare(self, m, series, query=None):
        new_series, invalid_series_subseq = _correct_data_and_create_masks(series, m, self._invalid_data_function)

        if query is not None:
            new_query, invalid_query_subseq = _correct_data_and_create_masks(query, m, self._invalid_data_function)
            num_q_subseq = len(query) - m + 1
        else:
            new_query = None
            invalid_query_subseq = invalid_series_subseq
            num_q_subseq = len(series) - m + 1

        generator = self._generator.prepare(m, new_series, new_query)
        return BoundFilterGenerator(generator, m, num_q_subseq, invalid_series_subseq, invalid_query_subseq)


class BoundFilterGenerator(AbstractBoundGenerator):
    """
        Wrapper around other generators that will replace values in the distance matrix marked as invalid
        by positive infinity. It can also perform a data pre-processing step before data reaches the wrapped generator,
        by setting values marked as invalid to zero, this can be useful for example to remove nan values for a generator
        that does not support nan values.
        """

    def __init__(self, generator, m, num_q_subseq, invalid_series_subseq, invalid_query_subseq):
        """
        Creates a new generator by wrapping another generator.

        :param generator: the generator whose results and input data will be filtered
        :param invalid_data_function: optional - a function that takes in the original data (series or query) and
           subsequence length and returns a boolean array of the same size that has a True value for any invalid values.
           These values will be replaced by zeros before reaching the wrapped generator. Any distance values
           that were calculated using invalid data points will be positive infinite values.
        :param invalid_subseq_function: optional - a function that takes in the original data (series or query) and
           subsequence length and returns a boolean array of size matching the number of subsequences that has
           a True value for any invalid subsequence. Invalid subsequences will have positive infinte values
           as distance.
        """
        self.generator = generator

        self.m = m
        self.num_q_subseq = num_q_subseq

        self.invalid_series_subseq = invalid_series_subseq
        self.invalid_query_subseq = invalid_query_subseq

    def calc_diagonal(self, diag):
        distances = self.generator.calc_diagonal(diag)

        if diag >= 0:
            if self.invalid_series_subseq is not None:
                distances[self.invalid_series_subseq[diag: diag+len(distances)]] = np.Inf
            if self.invalid_query_subseq is not None:
                distances[self.invalid_query_subseq[:len(distances)]] = np.Inf
        else:
            if self.invalid_series_subseq is not None:
                distances[self.invalid_series_subseq[:len(distances)]] = np.Inf
            if self.invalid_query_subseq is not None:
                distances[self.invalid_query_subseq[-diag: -diag+len(distances)]] = np.Inf

        return distances

    def calc_column(self, column):
        if self.invalid_series_subseq is not None and self.invalid_series_subseq[column]:
            return np.full(self.num_q_subseq, np.Inf)

        distances = self.generator.calc_column(column)

        if self.invalid_query_subseq is not None:
            distances[self.invalid_query_subseq] = np.Inf

        return distances


class BoundStreamingFilterGenerator(BoundFilterGenerator, AbstractBoundStreamingGenerator):
    """
    Wrapper around other generators that will replace values in the distance matrix marked as invalid
    by positive infinity. It can also perform a data pre-processing step before data reaches the wrapped generator,
    by setting values marked as invalid to zero, this can be useful for example to remove nan values for a generator
    that does not support nan values.
    """

    def __init__(self, generator, m, num_s_subseq, num_q_subseq, invalid_data_function):
        """
        Creates a new generator by wrapping another generator.

        :param generator: the generator whose results and input data will be filtered
        :param invalid_data_function: optional - a function that takes in the original data (series or query) and
           subsequence length and returns a boolean array of the same size that has a True value for any invalid values.
           These values will be replaced by zeros before reaching the wrapped generator. Any distance values
           that were calculated using invalid data points will be positive infinite values.
        """

        self._invalid_data_function = invalid_data_function

        invalid_s_subseq_buffer = RingBuffer(None, shape=(num_s_subseq,), dtype=np.bool)

        self.invalid_series = RingBuffer(None, shape=(num_s_subseq + m - 1,), dtype=np.bool)

        if num_q_subseq is None:
            self.self_join = True
            invalid_q_subseq_buffer = invalid_s_subseq_buffer
            num_q_subseq = num_s_subseq
            self.invalid_query = self.invalid_series
        else:
            self.self_join = False

            invalid_q_subseq_buffer = RingBuffer(None, shape=(num_q_subseq,), dtype=np.bool)
            self.invalid_query = RingBuffer(None, shape=(num_q_subseq + m - 1,), dtype=np.bool)

        super().__init__(generator, m, num_q_subseq, invalid_s_subseq_buffer, invalid_q_subseq_buffer)

    def append_series(self, values):
        invalid_points = _apply_data_validation(values, self.m, self._invalid_data_function)
        self.invalid_series.push(invalid_points)

        if np.any(invalid_points):
            values = values.copy()
            values[invalid_points] = 0

        if len(self.invalid_series.view) >= self.m:
            rel_values = self.invalid_series[-(len(values) + self.m - 1):]
            self.invalid_series_subseq.push(np.any(sliding_window_view(rel_values, (self.m,)), axis=-1))

        self.generator.append_series(values)

    def append_query(self, values):
        if self.self_join:
            raise RuntimeError("Cannot append to query for a self-join.")

        invalid_points = _apply_data_validation(values, self.m, self._invalid_data_function)
        self.invalid_query.push(invalid_points)

        if np.any(invalid_points):
            values = values.copy()
            values[invalid_points] = 0

        if len(self.invalid_query.view) >= self.m:
            rel_values = self.invalid_query[-(len(values) + self.m - 1):]
            self.invalid_query_subseq.push(np.any(sliding_window_view(rel_values, (self.m,)), axis=-1))

        self.generator.append_query(values)

    def calc_column(self, column):
        if self.invalid_series_subseq[column]:
            return np.full(len(self.invalid_query_subseq.view), np.Inf)

        distances = self.generator.calc_column(column)
        distances[self.invalid_query_subseq.view] = np.Inf

        return distances


def _apply_data_validation(data, m, invalid_data_function):
    """
    Returns a boolean array of the same size as data.

    :param data:
    :param m:
    :param invalid_data_function:
    :return:
    """
    invalid_data = invalid_data_function(data, m)
    if invalid_data.shape != data.shape:
        raise RuntimeError("Invalid_data_function's output does not have expected dimension.")

    return invalid_data


def _correct_data_and_create_masks(data, m, invalid_data_function):
    """
    Runs invalid_data_function and invalid_subseq_function, if they are defined.
    Any invalid data points are set to zero value and returned in a copied array.
    A boolean array is created to mark all invalid subsequence indices (= True values).

    :param data: 1D-array
    :param m: subsequence length
    :return: tuple of: data or a modified copy of data; None or a boolean 1D array containing at least 1 True
      (= invalid subsequence) value
    """
    invalid_data = invalid_data_function(data, m)
    if invalid_data.shape != data.shape:
        raise RuntimeError("Invalid_data_function's output does not have expected dimension.")


    # invalid_data = invalid_data and np.any(invalid_data)
    # invalid_subseq = invalid_subseq and np.any(invalid_subseq)

    new_data = data
    invalid_mask = None
    if invalid_data is not None:
        new_data = data.copy()
        new_data[invalid_data] = 0
        invalid_mask = _invalid_data_to_invalid_subseq(invalid_data, m)

    return new_data, invalid_mask

def _invalid_data_to_invalid_subseq(invalid_data, subseq_length):
    """
    Converts a boolean array marking invalid data points to a boolean array marking invalid subsequences.
    (A subsequence is invalid if it contained any invalid data point.)

    :param invalid_data: 1D array of booleans, True indicating invalid data points
    :param subseq_length: subsequence length
    :return: 1D boolean array of length num-subsequences
    """
    data_length = invalid_data.shape[0]
    result = np.zeros(data_length - subseq_length + 1, dtype=np.bool)

    impacted = 0
    for i in range(0, subseq_length - 1):
        if invalid_data[i]:
            impacted = subseq_length
        if impacted:
            impacted -= 1

    for i in range(subseq_length-1, data_length):
        if invalid_data[i]:
            impacted = subseq_length
        if impacted:
            result[i - subseq_length + 1] = True
            impacted -= 1

    return result
