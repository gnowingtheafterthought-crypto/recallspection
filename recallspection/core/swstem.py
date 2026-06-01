import numpy as np

class SWSTM:
    """Sparse-Write Self-Token Memory for bit-perfect retrieval."""
    def __init__(self, dim: int, max_size: int = 10000):
        self.dim = dim
        self.max_size = max_size
        self.keys = np.zeros((max_size, dim), dtype='float32')
        self.values = [None] * max_size
        self.ptr = 0

    def add(self, key: np.ndarray, value: any):
        idx = self.ptr % self.max_size
        self.keys[idx] = key.flatten()
        self.values[idx] = value
        self.ptr += 1
