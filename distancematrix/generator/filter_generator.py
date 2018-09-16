import numpy as np


def is_not_finite(data, subseq_length):
    """
    Marks infinite or nan values as invalid.
    """
    return ~np.isfinite(data)


class FilterGenerator(object):
    """
    Wrapper around other generators that will replace values in the distance matrix marked as invalid
    by positive infinity. It can also perform a data pre-processing step before data reaches the wrapped generator,
    by setting values marked as invalid to zero, this can be useful for example to remove nan values for a generator
    that does not support nan values.
    """
    def __init__(self, generator, invalid_data_function=is_not_finite, invalid_subseq_function=None):
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
        self.invalid_data_function = invalid_data_function
        self.invalid_subseq_function = invalid_subseq_function

        self.num_q_subseq = None
        self.invalid_series_subseq = None
        self.invalid_query_subseq = None

    def prepare(self, series, query, m):
        if series.ndim != 1:
            raise RuntimeError("Series should be 1D")
        if query.ndim != 1:
            raise RuntimeError("Query should be 1D")

        self.num_q_subseq = query.shape[0] - m + 1

        new_series, self.invalid_series_subseq = self._correct_data_and_create_masks(series, m)

        if series is query:
            # Self-join: skip work and ensure new_query and new_series are the same array
            new_query = new_series
            self.invalid_query_subseq = self.invalid_series_subseq
        else:
            new_query, self.invalid_query_subseq = self._correct_data_and_create_masks(query, m)

        self.generator.prepare(new_series, new_query, m)

    def _correct_data_and_create_masks(self, data, m):
        """
        Runs invalid_data_function and invalid_subseq_function, if they are defined.
        Any invalid data points are set to zero value and returned in a copied array.
        A boolean array is created to mark all invalid subsequence indices (= True values).

        :param data: 1D-array
        :param m: subsequence length
        :return: tuple of: data or a modified copy of data; None or a boolean 1D array containing at least 1 True
          (= invalid subsequence) value
        """
        if self.invalid_data_function:
            invalid_data = self.invalid_data_function(data, m)
            if invalid_data.shape != data.shape:
                raise RuntimeError("Invalid_data_function's output does not have expected dimension.")
        else:
            invalid_data = None

        if self.invalid_subseq_function:
            invalid_subseq = self.invalid_subseq_function(data, m)
            if invalid_subseq.shape != (data.shape[0] - m + 1,):
                raise RuntimeError("Invalid_subseq_function's output does not have expected dimension.")
        else:
            invalid_subseq = None

        # invalid_data = invalid_data and np.any(invalid_data)
        # invalid_subseq = invalid_subseq and np.any(invalid_subseq)

        new_data = data
        invalid_mask = None
        if invalid_data is not None:
            new_data = data.copy()
            new_data[invalid_data] = 0
            invalid_mask = _invalid_data_to_invalid_subseq(invalid_data, m)
        if invalid_subseq is not None:
            if invalid_mask is None:
                invalid_mask = invalid_subseq
            else:
                invalid_mask = np.logical_or(invalid_mask, invalid_subseq)

        return new_data, invalid_mask

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
