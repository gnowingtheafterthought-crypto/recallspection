import numpy as np

class SWSTM:
    """Sparse-Write Self-Token Memory for bit‑perfect retrieval."""
    
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

    def get(self, key: np.ndarray, top_k: int = 1):
        """Retrieve the closest stored value(s) by L2 distance."""
        key = key.flatten()
        diff = self.keys[:self.ptr] - key
        dist = np.linalg.norm(diff, axis=1)
        if len(dist) == 0:
            return []
        idxs = np.argsort(dist)[:top_k]
        return [self.values[i] for i in idxs if self.values[i] is not None]