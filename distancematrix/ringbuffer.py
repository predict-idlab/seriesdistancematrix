import numpy as np
from math import ceil


class RingBuffer(object):
    """
    A data structure that represents a sliding window over a data stream. Data can be pushed onto the buffer,
    thereby discarding the oldest data. The buffer is not resizable.

    Data is pushed onto the last dimension (in case of multidimensional data).

    Users should always reference the buffer instance, not the buffer view, as the view will be replaced
    as data is pushed onto the buffer. For user comfort, indexing and slicing on the buffer instance will
    immediately access the buffer view.
    """

    def __init__(self, data, shape=None, dtype=None, scaling_factor=2.) -> None:
        """
        Creates a new RingBuffer.

        :param data: data to initialize the buffer, data may be smaller or larger than shape, may be None to
            initialize an empty buffer
        :param shape: the shape of the buffer, if None, uses the shape of data
        :param dtype: the datatype for the buffer, if None, uses the dtype of data
        :param scaling_factor: determines internal buffer size (window size x scaling_factor)
        """
        super().__init__()

        if data is None and shape is None:
            raise RuntimeError("Data and shape may not both be None.")

        if data is None and dtype is None:
            raise RuntimeError("Data and dtype may not both be None.")

        if data is not None:
            data = np.asarray(data)

        if not shape:
            shape = list(data.shape)
        if not dtype:
            dtype = data.dtype

        self.max_shape = tuple(shape)
        self._view_start = 0  # Where view of the buffer starts
        self._view_max_length = shape[-1]  # Max length (last dimension) of the exposed view
        self._view_length = 0  # Current length of the exposed view

        buffer_shape = list(shape)
        buffer_shape[-1] = ceil(scaling_factor * shape[-1])
        self._buffer = np.empty(buffer_shape, dtype)

        self.view = self._buffer[..., self._view_start: self._view_start + self._view_length]
        if data is not None:
            self.push(data)

    def push(self, data):
        """
        Appends the given data to the buffer, discarding the oldest values.
        Data is appended to the last dimension of the data window.

        :param data: the data to append, all dimensions except the last should match those of the window
        :return: True if any data point was removed by this operation
        """
        data = np.atleast_1d(data)
        if not data.shape[:-1] == self._buffer.shape[:-1]:
            raise RuntimeError("Data shape does not match buffer size.")

        data_len = data.shape[-1]

        if data_len == 0:
            return False

        # If the view does not has its target capacity, first fill until it does
        if self._view_length < self._view_max_length:
            delta = min(data_len, self._view_max_length - self._view_length)
            self._buffer[..., self._view_length: self._view_length+delta] = data[..., :delta]
            self._view_length += delta
            self.view = self._buffer[..., :self._view_length]

            if data_len == delta:
                return False

            # The buffer (its view) is now filled, continue the normal flow to process the remaining data.
            data = data[..., delta:]
            data_len = data.shape[-1]

        # The view is at target capacity at this point, we will start "dropping" data.

        # The data fits in the remaining pre-allocated memory
        if self._view_start + self._view_max_length + data_len <= self._buffer.shape[-1]:
            self._view_start += data_len
            self.view = self._buffer[..., self._view_start:self._view_start + self._view_max_length]
            self.view[..., -data_len:] = data

        # The data does not fit in the remaining memory, but is less than the view capacity:
        # we reset the view, copy enough old data to fill to capacity, and append the new data
        elif data_len < self._view_max_length:
            mem_len = self._view_max_length - data_len
            self._buffer[..., :mem_len] = \
                self._buffer[..., self._view_start+data_len:self._view_start+self._view_max_length]
            self._buffer[..., mem_len:self._view_max_length] = data
            self._view_start = 0
            self.view = self._buffer[..., self._view_start:self._view_start + self._view_max_length]

        # The data does not fit in the remaining memory, and can (over)fill the view capacity:
        # we reset the view and copy a part of the new data equal to the view capacity.
        else:
            self._buffer[..., :self._view_max_length] = data[..., -self._view_max_length:]
            self._view_start = 0
            self.view = self._buffer[..., self._view_start:self._view_start + self._view_max_length]

        return True

    def __setitem__(self, key, value):
        self.view.__setitem__(key, value)

    def __getitem__(self, key):
        return self.view.__getitem__(key)

    def __delitem__(self, key):
        self.view.__delitem__(key)