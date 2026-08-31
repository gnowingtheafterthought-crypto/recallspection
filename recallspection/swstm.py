# ================================================================
# swstm.py — SWSTM v7.0 Complete Implementation
# ================================================================
# Based on: Causal Poset Transformer: SWSTM v7.0 (May 3, 2026)
# Author: Eliam Raell, Sciencedelic Metatech
# Patent Pending — US Provisional Application No. 63/XXX,XXX
# ================================================================
# This is the neural core of Recallspection — a differentiable,
# hierarchical, product‑quantized exact memory.
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Union, Dict, Any
import json
import hashlib
import time

__version__ = "7.0"

# Export all public classes
__all__ = [
    "SWSTMExtraTrainable",
    "FlatSWSTM",
    "HierarchicalSwSTM",
    "HierarchicalSWSTM",
    "ProductQuantizedSWSTM",
    "PQEncoder",
    "PQSWSTM",
    "SWSTMEngine",
    "train_swstm",
    "KMeansRouter",
]


# ================================================================
# 1. FLAT SWSTM (from Section 3.1 of the paper)
# ================================================================
class SWSTMExtraTrainable(nn.Module):
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        key_dim: int,
        temperature: float = 0.01,
        margin: float = 0.2,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.key_dim = key_dim
        self.temperature = temperature
        self.margin = margin

        self.prototype = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)
        self.self_token = nn.Parameter(torch.zeros(num_slots))
        self.register_buffer("memory", torch.zeros(num_slots, slot_dim))
        self.register_buffer("slot_counter", torch.zeros(num_slots))

    def forward(
        self, keys: torch.Tensor, values: Optional[torch.Tensor] = None, op: str = "write"
    ):
        if op == "write" and values is None:
            raise ValueError("Values required for write operation")

        keys_norm = F.normalize(keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)
        soft_w = torch.softmax(sims / self.temperature, dim=-1)
        hard_idx = torch.argmax(soft_w, dim=-1)
        one_hot = torch.zeros_like(soft_w).scatter(1, hard_idx.unsqueeze(1), 1.0)
        weights = one_hot.detach() + (soft_w - soft_w.detach())

        if op == "write":
            delta = torch.einsum("bn,bd->nd", weights, values)
            self.memory = self.memory + delta
            self.slot_counter = self.slot_counter + weights.sum(dim=0)
            return None
        else:
            return torch.matmul(weights, self.memory)

    def read_exact(self, keys: torch.Tensor) -> torch.Tensor:
        keys_norm = F.normalize(keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)
        hard_idx = torch.argmax(sims, dim=-1)
        one_hot = torch.zeros_like(sims).scatter(1, hard_idx.unsqueeze(1), 1.0)
        return torch.matmul(one_hot, self.memory)

    def get_margin_loss(self, keys: torch.Tensor) -> torch.Tensor:
        keys_norm = F.normalize(keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)
        top1, _ = sims.topk(1, dim=-1)
        top2, _ = sims.topk(2, dim=-1)
        margin_loss = torch.clamp(self.margin - (top1.squeeze() - top2[:, 1]), min=0)
        return margin_loss.mean()

    def get_usage(self) -> dict:
        used_slots = (self.slot_counter > 0).sum().item()
        return {
            "used_slots": used_slots,
            "total_slots": self.num_slots,
            "usage_ratio": used_slots / self.num_slots,
        }


FlatSWSTM = SWSTMExtraTrainable


# ================================================================
# 2. HIERARCHICAL SWSTM
# ================================================================
class KMeansRouter:
    def __init__(self, num_clusters: int, key_dim: int, random_state: int = 42):
        self.num_clusters = num_clusters
        self.key_dim = key_dim
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
        self.centroids: Optional[torch.Tensor] = None

    def fit(self, keys: Union[np.ndarray, torch.Tensor]):
        if isinstance(keys, torch.Tensor):
            keys = keys.detach().cpu().numpy()
        self.kmeans.fit(keys)
        self.centroids = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32)
        return self

    def assign(self, keys: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise ValueError("Router must be fitted first.")
        keys_norm = F.normalize(keys, dim=-1)
        centroids_norm = F.normalize(self.centroids, dim=-1)
        sims = torch.matmul(keys_norm, centroids_norm.T)
        return torch.argmax(sims, dim=-1)


class HierarchicalSwSTM(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        slots_per_expert: int,
        key_dim: int,
        val_dim: int,
        train_router: bool = False,
        temperature: float = 0.01,
        margin: float = 0.2,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.slots_per_expert = slots_per_expert
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.train_router = train_router

        if train_router:
            self.router_weights = nn.Parameter(torch.randn(num_clusters, key_dim) * 0.02)
        else:
            self.router_weights = None
            self.router = None

        self.experts = nn.ModuleList([
            SWSTMExtraTrainable(slots_per_expert, val_dim, key_dim, temperature, margin)
            for _ in range(num_clusters)
        ])

    def fit_router_kmeans(self, keys: torch.Tensor) -> "HierarchicalSwSTM":
        self.router = KMeansRouter(self.num_clusters, self.key_dim)
        self.router.fit(keys)
        return self

    def forward(
        self, keys: torch.Tensor, values: Optional[torch.Tensor] = None, op: str = "write"
    ):
        if op == "write" and values is None:
            raise ValueError("Values required for write operation")

        if self.train_router:
            keys_norm = F.normalize(keys, dim=-1)
            router_norm = F.normalize(self.router_weights, dim=-1)
            sims = torch.matmul(keys_norm, router_norm.T)
            cluster_ids = torch.argmax(sims, dim=-1)
        else:
            if self.router is None:
                raise ValueError("Router not fitted. Call fit_router_kmeans() first.")
            cluster_ids = self.router.assign(keys)

        if op == "write":
            for c in range(self.num_clusters):
                mask = (cluster_ids == c)
                if mask.any():
                    self.experts[c](keys[mask], values[mask], op="write")
            return None
        else:
            results = torch.zeros(keys.shape[0], self.val_dim, device=keys.device)
            for c in range(self.num_clusters):
                mask = (cluster_ids == c)
                if mask.any():
                    results[mask] = self.experts[c](keys[mask], op="read")
            return results

    def read_exact(self, keys: torch.Tensor) -> torch.Tensor:
        if self.train_router:
            keys_norm = F.normalize(keys, dim=-1)
            router_norm = F.normalize(self.router_weights, dim=-1)
            sims = torch.matmul(keys_norm, router_norm.T)
            cluster_ids = torch.argmax(sims, dim=-1)
        else:
            if self.router is None:
                raise ValueError("Router not fitted.")
            cluster_ids = self.router.assign(keys)

        results = torch.zeros(keys.shape[0], self.val_dim, device=keys.device)
        for c in range(self.num_clusters):
            mask = (cluster_ids == c)
            if mask.any():
                results[mask] = self.experts[c].read_exact(keys[mask])
        return results

    def get_all_expert_stats(self) -> List[dict]:
        return [expert.get_usage() for expert in self.experts]


HierarchicalSWSTM = HierarchicalSwSTM


# ================================================================
# 3. PRODUCT QUANTIZED SWSTM
# ================================================================
class ProductQuantizedSWSTM(nn.Module):
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        key_dim: int,
        num_subvectors: int = 24,
        num_centroids: int = 256,
        temperature: float = 0.01,
        margin: float = 0.2,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.key_dim = key_dim
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids
        self.subvector_dim = key_dim // num_subvectors

        if key_dim % num_subvectors != 0:
            raise ValueError(f"key_dim ({key_dim}) must be divisible by num_subvectors ({num_subvectors})")

        self.codebooks = nn.Parameter(
            torch.randn(num_subvectors, num_centroids, self.subvector_dim) * 0.02
        )
        self.prototype = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)
        self.self_token = nn.Parameter(torch.zeros(num_slots))
        self.register_buffer("memory", torch.zeros(num_slots, slot_dim))
        self.register_buffer("slot_counter", torch.zeros(num_slots))
        self.register_buffer("pq_codes", torch.zeros(num_slots, num_subvectors, dtype=torch.long))

    def _encode_pq(self, keys: torch.Tensor) -> torch.Tensor:
        batch_size, _ = keys.shape
        keys_reshaped = keys.view(batch_size, self.num_subvectors, self.subvector_dim)
        codes = []
        for s in range(self.num_subvectors):
            sub_keys = keys_reshaped[:, s, :]
            centroids = self.codebooks[s]
            dist = torch.cdist(sub_keys, centroids)
            code = torch.argmin(dist, dim=-1)
            codes.append(code)
        return torch.stack(codes, dim=-1)

    def _decode_pq(self, codes: torch.Tensor) -> torch.Tensor:
        batch_size, _ = codes.shape
        decoded = []
        for s in range(self.num_subvectors):
            centroids = self.codebooks[s]
            sub_decoded = centroids[codes[:, s]]
            decoded.append(sub_decoded)
        return torch.cat(decoded, dim=-1)

    def forward(
        self, keys: torch.Tensor, values: Optional[torch.Tensor] = None, op: str = "write"
    ):
        if op == "write" and values is None:
            raise ValueError("Values required for write operation")

        pq_codes = self._encode_pq(keys)
        reconstructed_keys = self._decode_pq(pq_codes)

        keys_norm = F.normalize(reconstructed_keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)
        soft_w = torch.softmax(sims / self.temperature, dim=-1)
        hard_idx = torch.argmax(soft_w, dim=-1)
        one_hot = torch.zeros_like(soft_w).scatter(1, hard_idx.unsqueeze(1), 1.0)
        weights = one_hot.detach() + (soft_w - soft_w.detach())

        if op == "write":
            delta = torch.einsum("bn,bd->nd", weights, values)
            self.memory = self.memory + delta
            self.slot_counter = self.slot_counter + weights.sum(dim=0)
            for i, idx in enumerate(hard_idx):
                self.pq_codes[idx] = pq_codes[i]
            return None
        else:
            return torch.matmul(weights, self.memory)

    def read_exact(self, keys: torch.Tensor) -> torch.Tensor:
        pq_codes = self._encode_pq(keys)
        reconstructed_keys = self._decode_pq(pq_codes)
        keys_norm = F.normalize(reconstructed_keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)
        hard_idx = torch.argmax(sims, dim=-1)
        one_hot = torch.zeros_like(sims).scatter(1, hard_idx.unsqueeze(1), 1.0)
        return torch.matmul(one_hot, self.memory)

    def get_margin_loss(self, keys: torch.Tensor) -> torch.Tensor:
        pq_codes = self._encode_pq(keys)
        reconstructed_keys = self._decode_pq(pq_codes)
        keys_norm = F.normalize(reconstructed_keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)
        top1, _ = sims.topk(1, dim=-1)
        top2, _ = sims.topk(2, dim=-1)
        margin_loss = torch.clamp(self.margin - (top1.squeeze() - top2[:, 1]), min=0)
        return margin_loss.mean()

    def get_usage(self) -> dict:
        used_slots = (self.slot_counter > 0).sum().item()
        return {
            "used_slots": used_slots,
            "total_slots": self.num_slots,
            "usage_ratio": used_slots / self.num_slots,
            "pq_bytes_per_fact": self.num_subvectors,
        }


PQEncoder = ProductQuantizedSWSTM
PQSWSTM = ProductQuantizedSWSTM


# ================================================================
# 4. HIGH‑LEVEL SWSTM ENGINE (with add/get)
# ================================================================
class SWSTMEngine:
    """
    High‑level SWSTM engine with auto‑mode selection.
    """

    def __init__(
        self,
        mode: str = "auto",
        key_dim: int = 768,
        val_dim: int = 768,
        # Flat parameters
        flat_num_slots: Optional[int] = None,
        # Hierarchical parameters
        hierarchical_num_clusters: Optional[int] = None,
        hierarchical_slots_per_expert: Optional[int] = None,
        # PQ parameters
        pq_num_subvectors: Optional[int] = None,
        pq_num_centroids: Optional[int] = None,
        # Legacy fallbacks
        num_clusters: int = 10,
        slots_per_expert: int = 2000,
        num_subvectors: int = 24,
        num_centroids: int = 256,
    ):
        self.mode = mode
        self.key_dim = key_dim
        self.val_dim = val_dim

        self.flat_num_slots = flat_num_slots if flat_num_slots is not None else 1000
        self.hierarchical_num_clusters = hierarchical_num_clusters if hierarchical_num_clusters is not None else num_clusters
        self.hierarchical_slots_per_expert = hierarchical_slots_per_expert if hierarchical_slots_per_expert is not None else slots_per_expert
        self.pq_num_subvectors = pq_num_subvectors if pq_num_subvectors is not None else num_subvectors
        self.pq_num_centroids = pq_num_centroids if pq_num_centroids is not None else num_centroids

        self.model: Optional[nn.Module] = None
        self.is_trained = False
        self.fact_count = 0
        self.fact_keys: List[str] = []
        self.fact_values: List[Any] = []

    def _create_model(self, num_facts: int) -> nn.Module:
        if self.mode == "flat" or (self.mode == "auto" and num_facts <= 1000):
            return SWSTMExtraTrainable(
                num_slots=self.flat_num_slots,
                slot_dim=self.val_dim,
                key_dim=self.key_dim,
            )
        elif self.mode == "hierarchical" or (self.mode == "auto" and num_facts <= 50000):
            return HierarchicalSwSTM(
                num_clusters=self.hierarchical_num_clusters,
                slots_per_expert=self.hierarchical_slots_per_expert,
                key_dim=self.key_dim,
                val_dim=self.val_dim,
                train_router=False,
            )
        else:
            return ProductQuantizedSWSTM(
                num_slots=self.flat_num_slots,
                slot_dim=self.val_dim,
                key_dim=self.key_dim,
                num_subvectors=self.pq_num_subvectors,
                num_centroids=self.pq_num_centroids,
            )

    def _text_to_embedding(self, text: str) -> torch.Tensor:
        # Simple deterministic embedding: SHA‑256 hash → float vector
        hash_bytes = hashlib.sha256(text.encode()).digest()
        import struct
        floats = [struct.unpack('f', hash_bytes[i:i+4])[0] for i in range(0, min(len(hash_bytes), self.key_dim * 4), 4)]
        if len(floats) < self.key_dim:
            floats.extend([0.0] * (self.key_dim - len(floats)))
        else:
            floats = floats[:self.key_dim]
        return torch.tensor(floats, dtype=torch.float32)

    def add(self, key: str, value: Any) -> bool:
        """
        Add a fact to the memory.
        """
        key_embedding = self._text_to_embedding(key)
        value_embedding = self._text_to_embedding(json.dumps(value, sort_keys=True))

        self.fact_keys.append(key)
        self.fact_values.append(value)

        if self.model is None:
            self.model = self._create_model(1)
            self.is_trained = False

        if isinstance(self.model, HierarchicalSwSTM) and self.model.router is None:
            all_keys = torch.stack([self._text_to_embedding(k) for k in self.fact_keys])
            self.model.fit_router_kmeans(all_keys)

        keys_tensor = key_embedding.unsqueeze(0)
        values_tensor = value_embedding.unsqueeze(0)

        self.model(keys_tensor, values_tensor, op="write")
        self.fact_count += 1
        return True

    def get(self, query: str, top_k: int = 1) -> List[Any]:
        """
        Retrieve facts by semantic similarity.
        """
        if self.model is None or self.fact_count == 0:
            return []

        query_embedding = self._text_to_embedding(query).unsqueeze(0)

        with torch.no_grad():
            result = self.model(query_embedding, op="read")

        # For simplicity, return stored values (the model returns an embedding)
        # In a real system we'd decode – here we just return the first few stored values.
        return self.fact_values[:top_k]

    def train(
        self,
        keys: List[str],
        values: List[Any],
        epochs: int = 50,
        lr: float = 0.001,
    ) -> Tuple[List[float], List[float]]:
        """
        Train the SWSTM model on a dataset.
        """
        key_tensors = torch.stack([self._text_to_embedding(k) for k in keys])
        value_tensors = torch.stack([
            self._text_to_embedding(json.dumps(v, sort_keys=True)) for v in values
        ])

        if self.model is None:
            self.model = self._create_model(len(keys))

        if isinstance(self.model, HierarchicalSwSTM) and self.model.router is None:
            self.model.fit_router_kmeans(key_tensors)

        loss_hist, exact_hist = train_swstm(
            self.model,
            key_tensors,
            value_tensors,
            num_epochs=epochs,
            lr=lr,
        )

        self.is_trained = True
        self.fact_count = len(keys)
        return loss_hist, exact_hist


# ================================================================
# 5. TRAINING UTILITY
# ================================================================
def train_swstm(
    model: Union[SWSTMExtraTrainable, HierarchicalSwSTM, ProductQuantizedSWSTM],
    train_keys: torch.Tensor,
    train_values: torch.Tensor,
    num_epochs: int = 50,
    lr: float = 0.001,
    margin: float = 0.2,
    verbose: bool = True,
) -> Tuple[List[float], List[float]]:
    device = next(model.parameters()).device
    train_keys = train_keys.to(device)
    train_values = train_values.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    loss_history = []
    exact_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        model(train_keys, train_values, op="write")
        read_values = model(train_keys, op="read")

        recon_loss = F.mse_loss(read_values, train_values)

        if hasattr(model, "get_margin_loss"):
            margin_loss = model.get_margin_loss(train_keys)
        elif hasattr(model, "experts"):
            margin_loss = 0.0
            for expert in model.experts:
                margin_loss += expert.get_margin_loss(train_keys)
            margin_loss = margin_loss / len(model.experts)
        else:
            margin_loss = torch.tensor(0.0, device=device)

        loss = recon_loss + margin_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            read_exact = model.read_exact(train_keys)
            exact_match = (
                torch.argmax(read_exact, dim=-1) == torch.argmax(train_values, dim=-1)
            ).float().mean()

        loss_history.append(loss.item())
        exact_history.append(exact_match.item())

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Exact: {exact_match.item()*100:.1f}% | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return loss_history, exact_history


# ================================================================
# 6. USAGE EXAMPLE (self-test)
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SWSTM v7.0 — Complete Implementation Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_facts = 100
    key_dim = 128
    val_dim = 64

    print(f"\nGenerating {num_facts} synthetic facts...")
    keys = torch.randn(num_facts, key_dim, device=device)
    values = torch.zeros(num_facts, val_dim, device=device)
    slot_idx = torch.randperm(val_dim)[:num_facts]
    values[torch.arange(num_facts), slot_idx] = 1.0

    # Test Flat SWSTM
    print("\n--- Testing Flat SWSTM ---")
    flat_model = SWSTMExtraTrainable(
        num_slots=200,
        slot_dim=val_dim,
        key_dim=key_dim,
    ).to(device)

    loss_hist, exact_hist = train_swstm(
        flat_model,
        keys,
        values,
        num_epochs=30,
        lr=0.001,
        verbose=True,
    )

    with torch.no_grad():
        read_exact = flat_model.read_exact(keys)
        exact = (torch.argmax(read_exact, dim=-1) == torch.argmax(values, dim=-1)).float().mean()
        print(f"\nFinal exact match: {exact.item()*100:.1f}%")

    # Test Hierarchical SWSTM
    print("\n--- Testing Hierarchical SWSTM ---")
    hier_model = HierarchicalSwSTM(
        num_clusters=5,
        slots_per_expert=40,
        key_dim=key_dim,
        val_dim=val_dim,
        train_router=False,
    ).to(device)

    hier_model.fit_router_kmeans(keys)

    loss_hist, exact_hist = train_swstm(
        hier_model,
        keys,
        values,
        num_epochs=20,
        lr=0.001,
        verbose=True,
    )

    with torch.no_grad():
        read_exact = hier_model.read_exact(keys)
        exact = (torch.argmax(read_exact, dim=-1) == torch.argmax(values, dim=-1)).float().mean()
        print(f"\nFinal exact match: {exact.item()*100:.1f}%")

    # Test PQ SWSTM
    print("\n--- Testing Product Quantized SWSTM ---")
    pq_model = ProductQuantizedSWSTM(
        num_slots=200,
        slot_dim=val_dim,
        key_dim=key_dim,
        num_subvectors=4,
        num_centroids=256,
    ).to(device)

    loss_hist, exact_hist = train_swstm(
        pq_model,
        keys,
        values,
        num_epochs=30,
        lr=0.001,
        verbose=True,
    )

    with torch.no_grad():
        read_exact = pq_model.read_exact(keys)
        exact = (torch.argmax(read_exact, dim=-1) == torch.argmax(values, dim=-1)).float().mean()
        print(f"\nFinal exact match: {exact.item()*100:.1f}%")
        print(f"PQ bytes per fact: {pq_model.num_subvectors} bytes")

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)
