"""
Simple tests for SWSTM (real).
Run with: pytest tests/test_swstm.py -v
"""

import pytest
import numpy as np
from recallspection.core.swstem import SWSTM

def test_swstm_add_and_get():
    sw = SWSTM(dim=4, max_size=10)
    key1 = np.array([1.0, 2.0, 3.0, 4.0])
    val1 = "hello"
    sw.add(key1, val1)
    result = sw.get(key1, top_k=1)
    assert result == [val1]

def test_swstm_multiple():
    sw = SWSTM(dim=2, max_size=5)
    data = [
        (np.array([1, 1]), "one"),
        (np.array([2, 2]), "two"),
        (np.array([3, 3]), "three"),
    ]
    for key, val in data:
        sw.add(key, val)
    for key, val in data:
        result = sw.get(key, top_k=1)
        assert result == [val]
    near_key = np.array([1.1, 1.1])
    result = sw.get(near_key, top_k=1)
    assert result == ["one"]

def test_swstm_empty():
    sw = SWSTM(dim=3, max_size=5)
    result = sw.get(np.array([1,2,3]))
    assert result == []

def test_swstm_overflow():
    sw = SWSTM(dim=2, max_size=3)
    for i in range(5):
        sw.add(np.array([i, i]), f"val_{i}")
    result = sw.get(np.array([4,4]), top_k=1)
    assert result == ["val_4"]
    result = sw.get(np.array([0,0]), top_k=1)
    assert result != ["val_0"]