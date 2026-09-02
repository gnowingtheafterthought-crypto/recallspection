import pytest
from recallspection.swstm import SWSTMEngine, ExactMemory, HybridEngine

def test_exact_memory_tamper():
    exact = ExactMemory()
    exact.add("key", "value")
    assert exact.get("key") == "value"
    # Tamper
    digest = exact._hash_key("key")
    exact._storage[digest] = b"TAMPERED"
    assert exact.get("key") is None

def test_swstm_exact_100_facts():
    engine = SWSTMEngine(num_slots=2000, key_dim=384, slot_dim=384)
    facts = [(f"fact_{i}", f"value_{i}") for i in range(100)]
    for k, v in facts:
        engine.add(k, v)
    engine.train(epochs=50, lr=0.01, margin=0.2)
    acc = engine.exact_match_accuracy([k for k, _ in facts], [v for _, v in facts])
    assert acc == 1.0

def test_swstm_paraphrase():
    engine = SWSTMEngine(num_slots=500, key_dim=384, slot_dim=384)
    facts = [
        ("capital of France", "Paris"),
        ("capital of Germany", "Berlin"),
        ("largest planet", "Jupiter"),
        ("author of 1984", "George Orwell"),
        ("speed of light", "299792458 m/s"),
    ]
    paraphrases = [
        ("France's capital", "Paris"),
        ("capital city of Germany", "Berlin"),
        ("biggest planet", "Jupiter"),
        ("who wrote 1984", "George Orwell"),
        ("how fast does light travel", "299792458 m/s"),
    ]
    for k, v in facts:
        engine.add(k, v)
    engine.train(epochs=100, lr=0.005, margin=0.3)
    acc = engine.paraphrase_accuracy([k for k, _ in paraphrases], [v for _, v in paraphrases])
    print(f"Paraphrase accuracy: {acc*100:.1f}%")
    assert acc >= 0.85

def test_swstm_save_load():
    engine = SWSTMEngine(num_slots=500, key_dim=384, slot_dim=384)
    engine.add("test", "value")
    engine.train(epochs=10, lr=0.01)
    engine.save("/tmp/swstm_test.pt")
    
    engine2 = SWSTMEngine(num_slots=500, key_dim=384, slot_dim=384)
    engine2.load("/tmp/swstm_test.pt")
    assert engine2.get("test") == ["value"]

def test_hybrid_engine():
    hybrid = HybridEngine(num_slots=500, key_dim=384, slot_dim=384)
    hybrid.add("exact_key", "exact_value")
    assert hybrid.get("exact_key") == ["exact_value"]
