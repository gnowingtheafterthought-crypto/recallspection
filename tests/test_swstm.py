"""
Simple tests for SWSTM (real).
Run with: pytest tests/test_swstm.py -v
"""

import pytest
import numpy as np
from recallspection.core.swstem import SWSTM

def test_swstm_add_and_get():
    """Test basic add and retrieve."""
    sw = SWSTM(dim=4, max_size=10)
    key1 = np.array([1.0, 2.0, 3.0, 4.0])
    val1 = "hello"
    sw.add(key1, val1)

    result = sw.get(key1, top_k=1)
    assert result == [val1], f"Expected ['hello'], got {result}"

def test_swstm_multiple():
    """Test adding and retrieving multiple items."""
    sw = SWSTM(dim=2, max_size=5)
    data = [
        (np.array([1, 1]), "one"),
        (np.array([2, 2]), "two"),
        (np.array([3, 3]), "three"),
    ]
    for key, val in data:
        sw.add(key, val)

    # Exact key should return correct value
    for key, val in data:
        result = sw.get(key, top_k=1)
        assert result == [val], f"Expected [{val}], got {result}"

    # A key close to an existing one should still find it
    near_key = np.array([1.1, 1.1])
    result = sw.get(near_key, top_k=1)
    assert result == ["one"], f"Expected ['one'], got {result}"

def test_swstm_empty():
    """Test get on empty memory returns empty list."""
    sw = SWSTM(dim=3, max_size=5)
    result = sw.get(np.array([1,2,3]))
    assert result == [], f"Expected [], got {result}"

def test_swstm_overflow():
    """Test that circular buffer works correctly."""
    sw = SWSTM(dim=2, max_size=3)
    for i in range(5):
        key = np.array([i, i])
        sw.add(key, f"val_{i}")

    # After 5 adds, only last 3 should remain
    result = sw.get(np.array([4,4]), top_k=1)
    assert result == ["val_4"], f"Expected ['val_4'], got {result}"
    # Older ones should be overwritten
    result = sw.get(np.array([0,0]), top_k=1)
    assert result != ["val_0"], "Old value should have been overwritten"