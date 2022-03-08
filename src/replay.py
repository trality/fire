import numpy as np


class Replay:

    def __init__(self, max_length: int):
        self.max_length = max_length
        self._empty = True
        self._queue = {}

    def append(self, d: dict):
        if self._empty:
            self._initialize(d)
            self._empty = False
        self._append(d)

    def _initialize(self, d: dict):
        self.fields = list(d.keys())
        self._index = 0
        self.size = 0
        for field in self.fields:
            dtype = d[field].dtype if hasattr(
                d[field], 'dtype') else type(d[field])
            shape = [self.max_length] + \
                (list(d[field].shape) if hasattr(d[field], 'shape') else [1])
            self._queue[field] = np.empty(shape, dtype=dtype)

    def _append(self, d: dict):
        for field in self.fields:
            self._queue[field][self._index] = d[field]
        self._update_index()

    def _update_index(self):
        self.size = max(self.size, self._index + 1)
        self._index = (self._index + 1) % self.max_length

    def __getitem__(self, field=None):
        if field is None:
            return {k: self._queue[k][:self.size] for k in self._queue}
        return self._queue[field][:self.size]

    def __repr__(self) -> str:
        return repr(self.__getitem__())

    def random_sample(self, n: int, replace=False):
        i = np.random.choice(np.arange(self.size), n, replace=replace)
        return {k: self._queue[k][i] for k in self._queue}
