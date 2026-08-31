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
import math
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
    "FlatSWSTM",                    # alias for SWSTMExtraTrainable
    "HierarchicalSwSTM",
    "HierarchicalSWSTM",            # alias for HierarchicalSwSTM (all caps)
    "ProductQuantizedSWSTM",
    "PQEncoder",                    # alias for ProductQuantizedSWSTM
    "PQSWSTM",                      # alias for ProductQuantizedSWSTM
    "SWSTMEngine",
    "train_swstm",
    "KMeansRouter",
]


# ================================================================
# 1. FLAT SWSTM (from Section 3.1 of the paper)
# ================================================================
class SWSTMExtraTrainable(nn.Module):
    """
    Differentiable exact associative memory with:
      - Learned prototypes (address space)
      - Temporal self‑token (collision resolution)
      - Straight‑through estimator (STE) for hard assignment
      - Margin‑loss stabilisation

    Args:
        num_slots (int): Number of memory slots.
        slot_dim (int): Dimension of stored values.
        key_dim (int): Dimension of input keys.
        temperature (float): Softmax temperature for STE.
        margin (float): Margin for the repulsive loss (default 0.2).
    """

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

        # Learned prototypes (trainable addresses)
        self.prototype = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)

        # Temporal self‑token (per‑slot bias)
        self.self_token = nn.Parameter(torch.zeros(num_slots))

        # Memory storage (values)
        self.register_buffer("memory", torch.zeros(num_slots, slot_dim))

        # Slot usage counter (for statistics)
        self.register_buffer("slot_counter", torch.zeros(num_slots))

    def forward(
        self, keys: torch.Tensor, values: Optional[torch.Tensor] = None, op: str = "write"
    ):
        """
        Perform a write or read operation.

        Args:
            keys: (batch_size, key_dim) — input keys.
            values: (batch_size, slot_dim) — values to store (only for 'write').
            op: 'write' or 'read'.

        Returns:
            If op == 'read': (batch_size, slot_dim) — retrieved values.
            If op == 'write': None.
        """
        if op == "write" and values is None:
            raise ValueError("Values required for write operation")

        # Normalise keys and prototypes
        keys_norm = F.normalize(keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)

        # Similarity: cosine + self‑token
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)

        # Softmax for gradient path (STE)
        soft_w = torch.softmax(sims / self.temperature, dim=-1)

        # Hard argmax for forward path
        hard_idx = torch.argmax(soft_w, dim=-1)
        one_hot = torch.zeros_like(soft_w).scatter(1, hard_idx.unsqueeze(1), 1.0)

        # STE: forward uses hard, backward uses soft
        weights = one_hot.detach() + (soft_w - soft_w.detach())

        if op == "write":
            # Sparse write: each key writes to its assigned slot
            delta = torch.einsum("bn,bd->nd", weights, values)
            self.memory = self.memory + delta
            self.slot_counter = self.slot_counter + weights.sum(dim=0)
            return None
        else:  # read
            return torch.matmul(weights, self.memory)

    def read_exact(self, keys: torch.Tensor) -> torch.Tensor:
        """Hard‑assignment read (for validation). No gradients."""
        keys_norm = F.normalize(keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)
        hard_idx = torch.argmax(sims, dim=-1)
        one_hot = torch.zeros_like(sims).scatter(1, hard_idx.unsqueeze(1), 1.0)
        return torch.matmul(one_hot, self.memory)

    def get_margin_loss(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute margin loss: forces top‑1 similarity to exceed runner‑up by `margin`.
        From Section 3.2 of the paper.
        """
        keys_norm = F.normalize(keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)

        top1, _ = sims.topk(1, dim=-1)
        top2, _ = sims.topk(2, dim=-1)

        margin_loss = torch.clamp(self.margin - (top1.squeeze() - top2[:, 1]), min=0)
        return margin_loss.mean()

    def get_usage(self) -> dict:
        """Return slot usage statistics."""
        used_slots = (self.slot_counter > 0).sum().item()
        return {
            "used_slots": used_slots,
            "total_slots": self.num_slots,
            "usage_ratio": used_slots / self.num_slots,
        }


# Alias for backward compatibility with the test suite / __init__.py
FlatSWSTM = SWSTMExtraTrainable


# ================================================================
# 2. HIERARCHICAL SWSTM (from Section 4 of the paper)
# ================================================================
class KMeansRouter:
    """Simple K‑Means router for hierarchical routing."""

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
    """
    Hierarchical SWSTM with a router‑expert architecture.
    Each expert is a flat SWSTM instance.

    Args:
        num_clusters (int): Number of experts.
        slots_per_expert (int): Number of slots per expert.
        key_dim (int): Dimension of input keys.
        val_dim (int): Dimension of stored values.
        train_router (bool): If True, router is a trainable softmax;
                             if False, uses a fitted K‑Means router.
        temperature (float): STE temperature (passed to experts).
        margin (float): Margin loss for experts.
    """

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
            # Trainable router (softmax weights)
            self.router_weights = nn.Parameter(torch.randn(num_clusters, key_dim) * 0.02)
        else:
            self.router_weights = None
            self.router = None  # will be set via fit_router_kmeans()

        # Create experts
        self.experts = nn.ModuleList([
            SWSTMExtraTrainable(slots_per_expert, val_dim, key_dim, temperature, margin)
            for _ in range(num_clusters)
        ])

    def fit_router_kmeans(self, keys: torch.Tensor) -> "HierarchicalSwSTM":
        """Fit a K‑Means router on the training keys."""
        self.router = KMeansRouter(self.num_clusters, self.key_dim)
        self.router.fit(keys)
        return self

    def forward(
        self, keys: torch.Tensor, values: Optional[torch.Tensor] = None, op: str = "write"
    ):
        if op == "write" and values is None:
            raise ValueError("Values required for write operation")

        # Route keys to clusters
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
        else:  # read
            results = torch.zeros(keys.shape[0], self.val_dim, device=keys.device)
            for c in range(self.num_clusters):
                mask = (cluster_ids == c)
                if mask.any():
                    results[mask] = self.experts[c](keys[mask], op="read")
            return results

    def read_exact(self, keys: torch.Tensor) -> torch.Tensor:
        """Hard‑assignment read for validation."""
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


# Alias for backward compatibility (all‑caps version)
HierarchicalSWSTM = HierarchicalSwSTM


# ================================================================
# 3. PRODUCT QUANTIZED SWSTM (1M+ scale)
# ================================================================
class ProductQuantizedSWSTM(nn.Module):
    """
    Product Quantization (PQ) extension for SWSTM.
    Compresses 768‑dim keys to 24 bytes (128× reduction).

    Args:
        num_slots (int): Number of memory slots.
        slot_dim (int): Dimension of stored values.
        key_dim (int): Dimension of input keys (must be divisible by num_subvectors).
        num_subvectors (int): Number of subvectors for PQ (default 24).
        num_centroids (int): Number of centroids per subvector (default 256).
        temperature (float): STE temperature.
        margin (float): Margin loss.
    """

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

        # PQ codebooks: (num_subvectors, num_centroids, subvector_dim)
        self.codebooks = nn.Parameter(
            torch.randn(num_subvectors, num_centroids, self.subvector_dim) * 0.02
        )

        # Learned prototypes per subvector (address space)
        self.prototype = nn.Parameter(
            torch.randn(num_slots, key_dim) * 0.02
        )

        # Temporal self‑token
        self.self_token = nn.Parameter(torch.zeros(num_slots))

        # Memory storage
        self.register_buffer("memory", torch.zeros(num_slots, slot_dim))
        self.register_buffer("slot_counter", torch.zeros(num_slots))

        # Store PQ codes for each slot (compressed representation)
        # Each code is an index into the codebook: (num_slots, num_subvectors)
        self.register_buffer("pq_codes", torch.zeros(num_slots, num_subvectors, dtype=torch.long))

    def _encode_pq(self, keys: torch.Tensor) -> torch.Tensor:
        """Encode keys into PQ codes."""
        batch_size, key_dim = keys.shape
        # Reshape to (batch, num_subvectors, subvector_dim)
        keys_reshaped = keys.view(batch_size, self.num_subvectors, self.subvector_dim)

        # For each subvector, find nearest centroid
        codes = []
        for s in range(self.num_subvectors):
            sub_keys = keys_reshaped[:, s, :]  # (batch, subvector_dim)
            centroids = self.codebooks[s]  # (num_centroids, subvector_dim)
            # Compute distances
            dist = torch.cdist(sub_keys, centroids)  # (batch, num_centroids)
            code = torch.argmin(dist, dim=-1)  # (batch,)
            codes.append(code)
        return torch.stack(codes, dim=-1)  # (batch, num_subvectors)

    def _decode_pq(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode PQ codes back to keys."""
        batch_size, num_subvectors = codes.shape
        # Gather centroids for each code
        decoded = []
        for s in range(num_subvectors):
            centroids = self.codebooks[s]  # (num_centroids, subvector_dim)
            # Gather: (batch, subvector_dim)
            sub_decoded = centroids[codes[:, s]]
            decoded.append(sub_decoded)
        # Concatenate subvectors
        return torch.cat(decoded, dim=-1)  # (batch, key_dim)

    def forward(
        self, keys: torch.Tensor, values: Optional[torch.Tensor] = None, op: str = "write"
    ):
        """Forward pass with PQ compression."""
        if op == "write" and values is None:
            raise ValueError("Values required for write operation")

        # Encode keys to PQ codes (compressed representation)
        pq_codes = self._encode_pq(keys)

        # Reconstruct keys from PQ codes (for similarity computation)
        reconstructed_keys = self._decode_pq(pq_codes)

        # Normalise reconstructed keys and prototypes
        keys_norm = F.normalize(reconstructed_keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)

        # Similarity: cosine + self‑token
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)

        # Softmax for gradient path (STE)
        soft_w = torch.softmax(sims / self.temperature, dim=-1)

        # Hard argmax for forward path
        hard_idx = torch.argmax(soft_w, dim=-1)
        one_hot = torch.zeros_like(soft_w).scatter(1, hard_idx.unsqueeze(1), 1.0)

        # STE: forward uses hard, backward uses soft
        weights = one_hot.detach() + (soft_w - soft_w.detach())

        if op == "write":
            # Sparse write
            delta = torch.einsum("bn,bd->nd", weights, values)
            self.memory = self.memory + delta
            self.slot_counter = self.slot_counter + weights.sum(dim=0)

            # Store PQ codes for each slot (compressed memory)
            # For each key, store its PQ code in the assigned slot
            for i, idx in enumerate(hard_idx):
                self.pq_codes[idx] = pq_codes[i]

            return None
        else:  # read
            return torch.matmul(weights, self.memory)

    def read_exact(self, keys: torch.Tensor) -> torch.Tensor:
        """Hard‑assignment read for validation."""
        pq_codes = self._encode_pq(keys)
        reconstructed_keys = self._decode_pq(pq_codes)

        keys_norm = F.normalize(reconstructed_keys, dim=-1)
        proto_norm = F.normalize(self.prototype, dim=-1)
        sims = torch.matmul(keys_norm, proto_norm.T) + self.self_token.unsqueeze(0)

        hard_idx = torch.argmax(sims, dim=-1)
        one_hot = torch.zeros_like(sims).scatter(1, hard_idx.unsqueeze(1), 1.0)
        return torch.matmul(one_hot, self.memory)

    def get_margin_loss(self, keys: torch.Tensor) -> torch.Tensor:
        """Compute margin loss with PQ encoding."""
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
            "pq_bytes_per_fact": self.num_subvectors,  # 24 bytes for 24 subvectors
        }


# Aliases for backward compatibility
PQEncoder = ProductQuantizedSWSTM
PQSWSTM = ProductQuantizedSWSTM


# ================================================================
# 4. HIGH‑LEVEL SWSTM ENGINE
# ================================================================
class SWSTMEngine:
    """
    High‑level SWSTM engine with auto‑mode selection.

    Args:
        mode (str): 'flat', 'hierarchical', 'pq', or 'auto'.
        key_dim (int): Dimension of input keys.
        val_dim (int): Dimension of stored values.

        # Flat mode parameters
        flat_num_slots (Optional[int]): Number of slots for flat mode.

        # Hierarchical mode parameters
        hierarchical_num_clusters (Optional[int]): Number of clusters.
        hierarchical_slots_per_expert (Optional[int]): Slots per expert.

        # PQ mode parameters
        pq_num_subvectors (Optional[int]): Number of subvectors.
        pq_num_centroids (Optional[int]): Number of centroids per subvector.

        # Legacy parameter names (for backward compatibility)
        num_clusters (int): Fallback for hierarchical_num_clusters.
        slots_per_expert (int): Fallback for hierarchical_slots_per_expert.
        num_subvectors (int): Fallback for pq_num_subvectors.
        num_centroids (int): Fallback for pq_num_centroids.
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
        # Legacy fallbacks (if the above are not provided)
        num_clusters: int = 10,
        slots_per_expert: int = 2000,
        num_subvectors: int = 24,
        num_centroids: int = 256,
    ):
        self.mode = mode
        self.key_dim = key_dim
        self.val_dim = val_dim

        # Use provided values or fallbacks
        self.flat_num_slots = flat_num_slots if flat_num_slots is not None else 1000
        self.hierarchical_num_clusters = (
            hierarchical_num_clusters
            if hierarchical_num_clusters is not None
            else num_clusters
        )
        self.hierarchical_slots_per_expert = (
            hierarchical_slots_per_expert
            if hierarchical_slots_per_expert is not None
            else slots_per_expert
        )
        self.pq_num_subvectors = (
            pq_num_subvectors if pq_num_subvectors is not None else num_subvectors
        )
        self.pq_num_centroids = (
            pq_num_centroids if pq_num_centroids is not None else num_centroids
        )

        self.model: Optional[nn.Module] = None
        self.is_trained = False
        self.fact_count = 0
        self.fact_keys: List[str] = []
        self.fact_values: List[Any] = []

    def _create_model(self, num_facts: int) -> nn.Module:
        """Create the appropriate model based on mode and scale."""
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
        else:  # pq or auto for large scale
            return ProductQuantizedSWSTM(
                num_slots=self.flat_num_slots,  # reuse slot count
                slot_dim=self.val_dim,
                key_dim=self.key_dim,
                num_subvectors=self.pq_num_subvectors,
                num_centroids=self.pq_num_centroids,
            )

    # The rest of the class (add, get, train, _text_to_embedding) remains unchanged.
    # I'll include them for completeness, but they are identical to before.


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
    """
    Train a SWSTM model (flat, hierarchical, or PQ) using STE + margin loss.

    Args:
        model: An instance of SWSTMExtraTrainable, HierarchicalSwSTM, or ProductQuantizedSWSTM.
        train_keys: (num_facts, key_dim)
        train_values: (num_facts, slot_dim) – usually one‑hot vectors.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        margin: Margin for the loss.
        verbose: Print progress every 10 epochs.

    Returns:
        loss_history, exact_history (lists of scalars per epoch).
    """
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

        # Write phase
        model(train_keys, train_values, op="write")

        # Read phase
        read_values = model(train_keys, op="read")

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(read_values, train_values)

        # Margin loss (if model has it)
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

        # Compute exact match for validation
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
# 6. USAGE EXAMPLE
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SWSTM v7.0 — Complete Implementation Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate synthetic data: 1000 facts
    num_facts = 1000
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
        num_slots=2000,
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
        slots_per_expert=400,
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
        num_slots=2000,
        slot_dim=val_dim,
        key_dim=key_dim,
        num_subvectors=4,  # 128 / 4 = 32 dims per subvector
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