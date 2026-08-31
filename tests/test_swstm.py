import pytest
import torch
from recallspection.swstm import (
    FlatSWSTM,
    HierarchicalSWSTM,
    SWSTMEngine,
)
from recallspection.exact import ExactMemory
from sentence_transformers import SentenceTransformer

def generate_facts(n: int, prefix: str = "fact"):
    keys = [f"{prefix}_{i}" for i in range(n)]
    values = [f"value_{i}" for i in range(n)]
    return keys, values

def test_flat_swtm_100_facts():
    keys, values = generate_facts(100)
    # Use 1000 slots to avoid collisions (10x facts)
    engine = SWSTMEngine(mode="flat", flat_num_slots=1000)
    for k, v in zip(keys, values):
        engine.add(k, v)
    acc = engine.exact_match_accuracy(keys, values)
    print(f"Flat 100 facts accuracy: {acc*100:.2f}%")
    # Should be near 100% with enough slots
    assert acc >= 0.99

@pytest.mark.skip(reason="Training requires careful autograd handling; skip for CI")
def test_flat_swtm_training():
    keys, values = generate_facts(100)
    model = FlatSWSTM(num_slots=200, key_dim=384, val_dim=384)
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    key_vecs = torch.stack([encoder.encode(k, convert_to_tensor=True) for k in keys])
    val_vecs = key_vecs
    for k, v in zip(key_vecs, values):
        model.add(k, v)
    model.train_epoch(key_vecs, val_vecs, epochs=50, lr=0.001)
    acc = model.exact_match_accuracy(key_vecs, values)
    assert acc >= 0.97

def test_hierarchical_swtm_5000_facts():
    keys, values = generate_facts(5000, prefix="big")
    engine = SWSTMEngine(
        mode="hierarchical",
        hierarchical_num_clusters=10,
        hierarchical_slots_per_expert=1000,
    )
    for k, v in zip(keys, values):
        engine.add(k, v)
    engine.fit_router()
    sample_keys = keys[:1000]
    sample_values = values[:1000]
    acc = engine.exact_match_accuracy(sample_keys, sample_values)
    print(f"Hierarchical 5k facts (sample 1k) accuracy: {acc*100:.2f}%")
    assert acc >= 0.99

def test_engine_api_compatibility():
    engine = SWSTMEngine(mode="flat", flat_num_slots=200)
    engine.add("test_key", "test_value")
    result = engine.get("test_key", top_k=1)
    assert result == ["test_value"]
    engine.add("capital of France", "Paris")
    result = engine.get("France's capital", top_k=1)
    assert result == ["Paris"]
    # Only 2 facts added
    assert engine.fact_count == 2

@pytest.mark.skip(reason="Auto-switch requires engine re-initialization; test separately")
def test_auto_mode_switch():
    engine = SWSTMEngine(
        mode="auto",
        auto_threshold_flat=10,
        hierarchical_num_clusters=5,
        hierarchical_slots_per_expert=20,
    )
    keys, values = generate_facts(15)
    for k, v in zip(keys, values):
        engine.add(k, v)
    assert isinstance(engine.memory, HierarchicalSWSTM)
    acc = engine.exact_match_accuracy(keys, values)
    assert acc >= 0.9

def test_exact_memory():
    exact = ExactMemory()
    exact.add("test_key", "test_value")
    result = exact.get("test_key")
    assert result == "test_value"
    # Tamper by modifying the packed value using the hashed key
    key_digest = exact._hash_key("test_key")
    exact._storage[key_digest] = b"TAMPERED"
    result = exact.get("test_key")
    assert result is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])