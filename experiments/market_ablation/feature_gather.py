"""Reuse one full-channel window read while preserving the existing batch logic."""
import numpy as np

from .config import FEATURE_PACKS


class _Column:
    """Let NumPy recover a complete gathered window from its ordered columns."""
    def __init__(self, values, channel):
        self._values, self._channel = values, channel

    def __array__(self, dtype=None, copy=None):
        column = np.asarray(self._values[..., self._channel], dtype=dtype)
        return column.copy() if copy else column

    def __getattr__(self, name):
        return getattr(self._values[..., self._channel], name)

    def __array_function__(self, function, types, arguments, keywords):
        if function is not np.stack:
            return NotImplemented
        columns = arguments[0]
        axis = keywords.get("axis", arguments[1] if len(arguments) > 1 else 0)
        if (axis in (-1, self._values.ndim - 1)
                and keywords.get("out") is None and keywords.get("dtype") is None
                and len(columns) == self._values.shape[-1]
                and all(isinstance(column, _Column) and column._values is self._values
                        and column._channel == index for index, column in enumerate(columns))):
            return self._values
        return np.stack([np.asarray(column) for column in columns], *arguments[1:], **keywords)


class _WindowGather:
    def __init__(self, array):
        self._array = array
        self._positions = None
        self._values = None

    def __getitem__(self, key):
        if (isinstance(key, tuple) and len(key) == 3
                and isinstance(key[0], np.ndarray)
                and isinstance(key[1], slice)
                and key[1].start is None and key[1].stop is None and key[1].step is None
                and isinstance(key[2], (int, np.integer))):
            positions, _, channel = key
            if positions is not self._positions:
                self._values = self._array[positions]
                self._positions = positions
            return _Column(self._values, channel)
        return self._array[key]


class _BatchView:
    def __init__(self, dataset):
        self._dataset = dataset
        self.features = _WindowGather(dataset.features)
        self.feature_mask = _WindowGather(dataset.feature_mask)

    def __getattr__(self, name):
        return getattr(self._dataset, name)


def feature_batch(dataset, origins, variant, moments):
    """Prefetch all channels only when the requested pack uses every channel."""
    channels = tuple(FEATURE_PACKS[variant.features])
    if channels != tuple(range(dataset.features.shape[-1])):
        return dataset.batch(origins, variant, moments)
    # Run the same batch method through a read-only view. Normalization, missing
    # masks, history counts, labels, and eligibility retain their existing code.
    return type(dataset).batch(_BatchView(dataset), origins, variant, moments)
