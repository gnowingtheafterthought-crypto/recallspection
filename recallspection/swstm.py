"""
swstm.py — SWSTM v8.0 (Pure Neural Exact Memory)
================================================
No FAISS. No dict fallback in neural retrieval.
Slot-based competitive addressing with STE.
ExactMemory is separate — audit trail only.
"""

import json
import hashlib
import zlib
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer


# ================================================================
# 1. EXACTMEMORY — Audit/Compliance Core (unchanged, hardened)
# ================================================================

class ExactMemory:
    """Tamper-evident cryptographic key-value store."""
    def __init__(self):
        self._storage: Dict[bytes, bytes] = {}
        self._fact_count = 0

    def _hash_key(self, key: Union[str, bytes]) -> bytes:
        if isinstance(key, str):
            key = key.encode('utf-8')
        return hashlib.sha3_256(key).digest()

    def _pack(self, value: Any) -> bytes:
        json_str = json.dumps(value, sort_keys=True)
        compressed = zlib.compress(json_str.encode('utf-8'), level=6)
        checksum = hashlib.sha256(compressed).digest()  # full 32-byte
        return checksum + compressed

    def _unpack(self, packed: bytes) -> Optional[Any]:
        if len(packed) < 32:
            return None
        checksum, compressed = packed[:32], packed[32:]
        if hashlib.sha256(compressed).digest() != checksum:
            return None
        try:
            return json.loads(zlib.decompress(compressed).decode('utf-8'))
        except Exception:
            return None

    def add(self, key: Union[str, bytes], value: Any) -> None:
        self._storage[self._hash_key(key)] = self._pack(value)
        self._fact_count += 1

    def get(self, key: Union[str, bytes]) -> Optional[Any]:
        packed = self._storage.get(self._hash_key(key))
        return self._unpack(packed) if packed else None

    def __len__(self) -> int:
        return self._fact_count


# ================================================================
# 2. SWSTM CORE — Slot-Based Competitive Memory
# ================================================================

class SWSTMCore(nn.Module):
    """
    Flat SWSTM with STE-based hard slot selection.

    Architecture:
        keys (N, key_dim) 
            -> normalize 
            -> dot with prototypes (num_slots, key_dim) 
            -> add self_token bias 
            -> softmax / argmax (STE) 
            -> read from memory (num_slots, slot_dim) 
            -> return values (N, slot_dim)
    """
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        key_dim: int,
        temperature: float = 0.01,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.key_dim = key_dim
        self.temperature = temperature

        # Competitive addressing
        self.prototype = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)
        self.self_token = nn.Parameter(torch.zeros(num_slots))

        # Memory bank (values live here)
        self.register_buffer("memory", torch.zeros(num_slots, slot_dim))
        self.register_buffer("slot_occupied", torch.zeros(num_slots, dtype=torch.bool))
        self.register_buffer("write_count", torch.zeros(num_slots, dtype=torch.long))

    def forward(self, keys: torch.Tensor, values: Optional[torch.Tensor] = None, op: str = "read"):
        keys_norm = F.normalize(keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)
        soft_w = F.softmax(sims / self.temperature, dim=-1)
        hard_idx = torch.argmax(soft_w, dim=-1)
        one_hot = F.one_hot(hard_idx, num_classes=self.num_slots).float()
        weights = one_hot.detach() + (soft_w - soft_w.detach())  # STE

        if op == "write":
            if values is None:
                raise ValueError("values required for write")
            delta = torch.einsum("bn,bd->nd", weights, values)
            self.memory.add_(delta)
            self.slot_occupied[hard_idx] = True
            self.write_count[hard_idx] += 1
            return hard_idx

        return torch.matmul(weights, self.memory)

    def read_exact(self, keys: torch.Tensor) -> torch.Tensor:
        """Hard assignment read (no STE, no gradients)."""
        keys_norm = F.normalize(keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)
        hard_idx = torch.argmax(sims, dim=-1)
        one_hot = F.one_hot(hard_idx, num_classes=self.num_slots).float()
        return torch.matmul(one_hot, self.memory)

    def margin_loss(self, keys: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
        """Repulsive loss: push top-2 prototypes apart."""
        keys_norm = F.normalize(keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)
        top2, _ = sims.topk(2, dim=-1)
        return torch.mean(F.relu(margin - (top2[:, 0] - top2[:, 1])))

    def init_prototypes_from_keys(self, keys: torch.Tensor):
        """Initialize prototypes via k-means on key embeddings."""
        from sklearn.cluster import KMeans
        keys_np = keys.detach().cpu().numpy()
        n_clusters = min(self.num_slots, keys_np.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(keys_np)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=keys.dtype, device=keys.device)
        self.prototype.data[:n_clusters] = centroids[:n_clusters]
        # Remaining prototypes stay random (will be used as expansion slots)

    def get_state(self) -> Dict[str, Any]:
        return {
            "prototype": self.prototype.cpu(),
            "self_token": self.self_token.cpu(),
            "memory": self.memory.cpu(),
            "slot_occupied": self.slot_occupied.cpu(),
            "write_count": self.write_count.cpu(),
            "num_slots": self.num_slots,
            "slot_dim": self.slot_dim,
            "key_dim": self.key_dim,
            "temperature": self.temperature,
        }

    def set_state(self, state: Dict[str, Any]):
        self.prototype.data.copy_(state["prototype"].to(self.prototype.device))
        self.self_token.data.copy_(state["self_token"].to(self.self_token.device))
        self.memory.copy_(state["memory"].to(self.memory.device))
        self.slot_occupied.copy_(state["slot_occupied"].to(self.slot_occupied.device))
        self.write_count.copy_(state["write_count"].to(self.write_count.device))


# ================================================================
# 3. SWSTM ENGINE — High-Level Interface
# ================================================================

class SWSTMEngine:
    """
    SWSTM-only memory engine.

    - Neural retrieval ONLY (no dict fallback)
    - Values stored as strings, indexed by slot
    - Training loop included
    - Save/load included
    """
    def# In __init__, replace:
self._train_keys: List[torch.Tensor] = []
self._train_slots: List[int] = []

# With:
self._train_buffer: List[Tuple[torch.Tensor, torch.Tensor, str]] = []

# In add(), replace:
self._train_keys.append(key_vec.squeeze(0))
self._train_slots.append(slot_idx)

# With:
self._train_buffer.append((key_vec.squeeze(0), val_vec.squeeze(0), value))

# In train(), replace the entire method body with:
def train(self, epochs: int = 50, lr: float = 0.01, margin: float = 0.2):
    if len(self._train_buffer) < 2:
        print("[SWSTM] Not enough data to train.")
        return
    
    keys = torch.stack([k for k, _, _ in self._train_buffer])
    
    # Initialize prototypes once, then rebuild memory
    if not hasattr(self.model, '_prototypes_initialized'):
        print(f"[SWSTM] Initializing {self.model.num_slots} prototypes from {len(keys)} keys...")
        self.model.init_prototypes_from_keys(keys)
        self.model._prototypes_initialized = True
        
        # CRITICAL: Rebuild memory with new prototypes
        self.model.memory.zero_()
        self.model.slot_occupied.zero_()
        self.model.write_count.zero_()
        self.slot_to_value.clear()
        self.value_to_slot.clear()
        
        for key_vec, val_vec, value_str in self._train_buffer:
            slot_idx = self.model.forward(
                key_vec.unsqueeze(0), val_vec.unsqueeze(0), op="write"
            ).item()
            self.slot_to_value[slot_idx] = value_str
            self.value_to_slot[value_str] = slot_idx
    
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    print(f"[SWSTM] Training on {len(keys)} keys for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = self.model.margin_loss(keys, margin=margin)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, margin_loss={loss.item():.4f}")
    
    print("[SWSTM] Training complete.")


# ================================================================
# 4. HYBRID ENGINE — Dual Core (SWSTM + ExactMemory)
# ================================================================

class HybridEngine:
    """
    Dual-core memory:
    - ExactMemory: deterministic, tamper-evident, audit trail
    - SWSTMEngine: neural, paraphrase-capable

    Writes go to BOTH. Reads try Exact first, then SWSTM.
    """
    def __init__(self, **swstm_kwargs):
        self.exact = ExactMemory()
        self.swstm = SWSTMEngine(**swstm_kwargs)

    def add(self, key: str, value: str):
        self.exact.add(key, value)
        self.swstm.add(key, value)

    def get(self, key: str, top_k: int = 1) -> List[str]:
        # 1. Exact match
        result = self.exact.get(key)
        if result is not None:
            return [result]
        # 2. Neural fallback
        return self.swstm.get(key, top_k=top_k)

    def train(self, epochs: int = 50, lr: float = 0.01):
        self.swstm.train(epochs=epochs, lr=lr)

    def save(self, path: Union[str, Path]):
        self.swstm.save(path)

    def load(self, path: Union[str, Path]):
        self.swstm.load(path)


# ================================================================
# 5. SELF-TEST
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SWSTM v8.0 — Self-Test")
    print("=" * 60)

    # --- Test 1: Basic exact retrieval ---
    print("\n[TEST 1] Exact key retrieval (100 facts, untrained)")
    engine = SWSTMEngine(num_slots=1000, key_dim=384, slot_dim=384)
    facts = [(f"fact_{i}", f"value_{i}") for i in range(100)]
    for k, v in facts:
        engine.add(k, v)

    acc = engine.exact_match_accuracy([k for k, _ in facts], [v for _, v in facts])
    print(f"  Exact accuracy (untrained): {acc*100:.1f}%")

    # --- Test 2: Paraphrase retrieval (untrained) ---
    print("\n[TEST 2] Paraphrase retrieval (untrained)")
    para_facts = [
        ("capital of France", "Paris"),
        ("capital of Germany", "Berlin"),
        ("largest planet", "Jupiter"),
    ]
    para_queries = [
        ("France's capital", "Paris"),
        ("capital city of Germany", "Berlin"),
        ("biggest planet", "Jupiter"),
    ]

    engine2 = SWSTMEngine(num_slots=500, key_dim=384, slot_dim=384)
    for k, v in para_facts:
        engine2.add(k, v)

    for q, expected in para_queries:
        result = engine2.get(q)
        print(f"  '{q}' -> {result} (expected: {expected})")

    para_acc = engine2.paraphrase_accuracy([q for q, _ in para_queries], [e for _, e in para_queries])
    print(f"  Paraphrase accuracy (untrained): {para_acc*100:.1f}%")

    # --- Test 3: Train and re-test ---
    print("\n[TEST 3] Training and re-testing")
    engine2.train(epochs=30, lr=0.01)

    para_acc_trained = engine2.paraphrase_accuracy([q for q, _ in para_queries], [e for _, e in para_queries])
    print(f"  Paraphrase accuracy (trained): {para_acc_trained*100:.1f}%")

    # --- Test 4: Save / Load ---
    print("\n[TEST 4] Save and load")
    engine2.save("/tmp/swstm_test.pt")
    engine3 = SWSTMEngine(num_slots=500, key_dim=384, slot_dim=384)
    engine3.load("/tmp/swstm_test.pt")
    result = engine3.get("France's capital")
    print(f"  After load: 'France\'s capital' -> {result}")

    # --- Test 5: Hybrid Engine ---
    print("\n[TEST 5] Hybrid Engine (Exact + SWSTM)")
    hybrid = HybridEngine(num_slots=500, key_dim=384, slot_dim=384)
    hybrid.add("secret_key_42", "classified_data")
    print(f"  Exact lookup: {hybrid.get('secret_key_42')}")

    print("\n" + "=" * 60)
    print("Self-test complete.")
    print("=" * 60)
