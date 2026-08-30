"""
Test suite for SWSTM v7.0 engine.
Run with: pytest tests/test_swstm.py -v
"""

import pytest
import torch
from recallspection.swstm import (
    FlatSWSTM,
    HierarchicalSWSTM,
    SWSTMEngine,
)
from recallspection.exact import ExactMemory
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Helper: generate synthetic facts
# -----------------------------------------------------------------------------

def generate_facts(n: int, prefix: str = "fact"):
    """Return (keys, values) as lists of strings."""
    keys = [f"{prefix}_{i}" for i in range(n)]
    values = [f"value_{i}" for i in range(n)]
    return keys, values

# -----------------------------------------------------------------------------
# Test 1: Flat SWSTM – 100 facts
# -----------------------------------------------------------------------------

def test_flat_swtm_100_facts():
    """Reproduce 97%+ exact match on 100 synthetic facts."""
    keys, values = generate_facts(100)
    engine = SWSTMEngine(mode="flat", flat_num_slots=200)

    # Add facts
    for k, v in zip(keys, values):
        engine.add(k, v)

    # Measure accuracy (no training needed when slots = 2x facts)
    acc = engine.exact_match_accuracy(keys, values)
    print(f"Flat 100 facts accuracy: {acc*100:.2f}%")
    assert acc >= 0.97, f"Flat accuracy {acc} < 0.97"

# -----------------------------------------------------------------------------
# Test 2: Flat SWSTM with training (optional)
# -----------------------------------------------------------------------------

def test_flat_swtm_training():
    """Test training loop improves accuracy."""
    keys, values = generate_facts(100)
    model = FlatSWSTM(num_slots=200, key_dim=384, val_dim=384)
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode keys and values
    key_vecs = torch.stack([encoder.encode(k, convert_to_tensor=True) for k in keys])
    val_vecs = key_vecs  # we use key embedding as value

    # Add facts to model
    for k, v in zip(key_vecs, values):
        model.add(k, v)

    # Train for 50 epochs
    model.train_epoch(key_vecs, val_vecs, epochs=50, lr=0.001)

    # Evaluate
    acc = model.exact_match_accuracy(key_vecs, values)
    assert acc >= 0.97, f"Trained flat accuracy {acc} < 0.97"

# -----------------------------------------------------------------------------
# Test 3: Hierarchical SWSTM – 5,000 facts
# -----------------------------------------------------------------------------

def test_hierarchical_swtm_5000_facts():
    """Test hierarchical with 5,000 facts, 10 clusters, 2x slots each."""
    keys, values = generate_facts(5000, prefix="big")
    engine = SWSTMEngine(
        mode="hierarchical",
        hierarchical_num_clusters=10,
        hierarchical_slots_per_expert=1000,   # 2x average (500 per cluster)
    )

    # Add facts (they will be stored pending)
    for k, v in zip(keys, values):
        engine.add(k, v)

    # Fit router and flush pending
    engine.fit_router()

    # Check accuracy on a subset (full 5000 might be slow; we test 1000)
    sample_keys = keys[:1000]
    sample_values = values[:1000]
    acc = engine.exact_match_accuracy(sample_keys, sample_values)
    print(f"Hierarchical 5k facts (sample 1k) accuracy: {acc*100:.2f}%")
    assert acc >= 0.99, f"Hierarchical accuracy {acc} < 0.99"

# -----------------------------------------------------------------------------
# Test 4: Engine API compatibility (drop‑in replacement)
# -----------------------------------------------------------------------------

def test_engine_api_compatibility():
    """Ensure SWSTMEngine mimics ExactMemory API."""
    engine = SWSTMEngine(mode="flat", flat_num_slots=200)

    # Add
    engine.add("test_key", "test_value")
    # Get
    result = engine.get("test_key", top_k=1)
    assert result == ["test_value"], f"Expected ['test_value'], got {result}"

    # Add more
    engine.add("capital of France", "Paris")
    result = engine.get("France's capital", top_k=1)
    assert result == ["Paris"], f"Expected ['Paris'], got {result}"

    # Check fact count
    assert engine.fact_count == 3

# -----------------------------------------------------------------------------
# Test 5: Auto‑mode switching (flat -> hierarchical when threshold crossed)
# -----------------------------------------------------------------------------

def test_auto_mode_switch():
    """If mode='auto', engine switches to hierarchical after auto_threshold_flat."""
    engine = SWSTMEngine(
        mode="auto",
        auto_threshold_flat=10,   # small threshold for testing
        hierarchical_num_clusters=5,
        hierarchical_slots_per_expert=20,
    )
    # Add 15 facts – should trigger switch to hierarchical
    keys, values = generate_facts(15)
    for k, v in zip(keys, values):
        engine.add(k, v)

    # After adding, the engine should have switched to hierarchical
    assert isinstance(engine.memory, HierarchicalSWSTM), \
        f"Expected HierarchicalSWSTM, got {type(engine.memory)}"

    # It should still retrieve correctly
    acc = engine.exact_match_accuracy(keys, values)
    assert acc >= 0.9, f"Auto‑switch accuracy {acc} < 0.9"

# -----------------------------------------------------------------------------
# Test 6: ExactMemory basic test
# -----------------------------------------------------------------------------

def test_exact_memory():
    exact = ExactMemory()
    exact.add("test_key", "test_value")
    result = exact.get("test_key")
    assert result == "test_value"
    # Tamper test
    exact._storage["test_key"] = b"TAMPERED"
    result = exact.get("test_key")
    assert result is None

# -----------------------------------------------------------------------------
# Run all tests if executed directly
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])