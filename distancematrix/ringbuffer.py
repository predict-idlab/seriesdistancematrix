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

    def __init__(self, data, scaling_factor=2.) -> None:
        """
        Creates a new RingBuffer.

        :param data: the initial window of the buffer
        :param scaling_factor: determines internal buffer size (window size x scaling_factor)
        """
        super().__init__()

        data = np.asarray(data)
        new_shape = list(data.shape)
        new_shape[-1] = ceil(scaling_factor * new_shape[-1])

        self._buffer = np.empty(new_shape, data.dtype)
        self._window_size = data.shape[-1]
        self._view_start = 0

        self.view = self._buffer[..., self._view_start:self._view_start+self._window_size]
        self.view[:] = data

    def push(self, data):
        """
        Appends the given data to the buffer, discarding the oldest values.
        Data is appended to the last dimension of the data window.

        :param data: the data to append, all dimensions except the last should match those of the window
        :return: None
        """
        data = np.asarray(data)
        if not data.shape[:-1] == self._buffer.shape[:-1]:
            raise RuntimeError("Data shape does not match buffer size.")

        data_len = data.shape[-1]

        if data_len == 0:
            return

        if self._view_start + self._window_size + data_len <= self._buffer.shape[-1]:
            self._view_start += data_len
            self.view = self._buffer[..., self._view_start:self._view_start + self._window_size]
            self.view[..., -data_len:] = data

        elif data_len < self._window_size:
            mem_len = self._window_size - data_len
            self._buffer[..., :mem_len] = self._buffer[..., self._view_start+data_len:self._view_start+self._window_size]
            self._buffer[..., mem_len:self._window_size] = data
            self._view_start = 0
            self.view = self._buffer[..., self._view_start:self._view_start + self._window_size]

        else:
            self._buffer[..., :self._window_size] = data[..., -self._window_size:]
            self._view_start = 0
            self.view = self._buffer[..., self._view_start:self._view_start + self._window_size]

    def __setitem__(self, key, value):
        self.view.__setitem__(key, value)

    def __getitem__(self, key):
        return self.view.__getitem__(key)

    def __delitem__(self, key):
        self.view.__delitem__(key)