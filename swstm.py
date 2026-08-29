"""
SWSTM v7.0 – Hierarchical Exact Associative Memory with Product Quantization
Based on "Causal Poset Transformer: SWSTM v7.0" by Eliam Raell.
Patent Pending.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
import warnings

# -----------------------------------------------------------------------------
# Flat SWSTM (STE + Margin + Self‑token)
# -----------------------------------------------------------------------------

class FlatSWSTM(nn.Module):
    """
    Flat memory with STE training, self-token, and margin loss.
    """

    def __init__(
        self,
        num_slots: int,
        key_dim: int = 384,
        val_dim: int = 384,
        temperature: float = 0.01,
        margin: float = 0.2,
        token_beta: float = 0.9,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.temperature = temperature
        self.margin = margin
        self.token_beta = token_beta

        # Trainable parameters
        self.prototypes = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)
        self.self_token = nn.Parameter(torch.zeros(num_slots))

        # Buffers (non‑trainable)
        self.register_buffer("memory", torch.zeros(num_slots, val_dim))
        self.register_buffer("used", torch.zeros(num_slots, dtype=torch.bool))
        self.register_buffer("slot_count", torch.zeros(num_slots, dtype=torch.long))

        self.fact_count = 0
        self.value_map: Dict[int, str] = {}  # slot index → original string

    def forward(self, keys, values=None, op="write"):
        norm_keys = F.normalize(keys, dim=-1)
        norm_proto = F.normalize(self.prototypes, dim=-1)
        sims = torch.mm(norm_keys, norm_proto.T) + self.self_token.unsqueeze(0)

        soft_w = F.softmax(sims / self.temperature, dim=-1)
        hard_idx = torch.argmax(soft_w, dim=-1)
        one_hot = F.one_hot(hard_idx, num_classes=self.num_slots).float()
        weights = one_hot.detach() + soft_w - soft_w.detach()  # STE

        if op == "write":
            delta = torch.einsum("bn,bd->nd", weights, values)
            self.memory = self.memory + delta
            self.used[hard_idx] = True
            self.slot_count[hard_idx] += 1

            # Update self‑token with max similarity (EMA)
            with torch.no_grad():
                max_sim, _ = torch.max(sims, dim=-1)
                for i, idx in enumerate(hard_idx):
                    self.self_token[idx] = (
                        self.token_beta * self.self_token[idx]
                        + (1 - self.token_beta) * max_sim[i].item()
                    )
            self.fact_count += keys.size(0)
            return hard_idx
        else:  # read
            return torch.mm(weights, self.memory)

    def add(self, key_vec: torch.Tensor, value_str: str) -> int:
        """Add a single key‑value pair. Returns slot index."""
        val_vec = key_vec  # we store the key embedding as value for simplicity
        idx = self.forward(key_vec.unsqueeze(0), val_vec.unsqueeze(0), op="write").item()
        self.value_map[idx] = value_str
        return idx

    def get(self, query_vec: torch.Tensor, top_k: int = 1) -> List[str]:
        """Retrieve top‑k values for a query."""
        norm_q = F.normalize(query_vec.unsqueeze(0), dim=-1)
        norm_mem = F.normalize(self.memory, dim=-1)
        sims = torch.mm(norm_q, norm_mem.T)
        sims = sims.masked_fill(~self.used.unsqueeze(0), -1e9)
        top_scores, top_indices = torch.topk(sims, min(top_k, self.fact_count), dim=-1)
        results = []
        for idx in top_indices.squeeze(0).tolist():
            if idx in self.value_map:
                results.append(self.value_map[idx])
        return results

    def margin_loss(self, sims):
        top1, _ = torch.topk(sims, 2, dim=-1)
        return torch.mean(torch.relu(self.margin - (top1[:, 0] - top1[:, 1])))

    def train_step(self, keys, values, optimizer):
        # Write
        self.forward(keys, values, op="write")
        # Read
        read_values = self.forward(keys, op="read")
        recon_loss = F.mse_loss(read_values, values)

        # Margin on similarities
        norm_keys = F.normalize(keys, dim=-1)
        norm_proto = F.normalize(self.prototypes, dim=-1)
        sims = torch.mm(norm_keys, norm_proto.T) + self.self_token.unsqueeze(0)
        margin_loss = self.margin_loss(sims)

        total_loss = recon_loss + 0.1 * margin_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return total_loss.item()

    def train_epoch(self, keys, values, epochs=100, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        losses = []
        for epoch in range(epochs):
            perm = torch.randperm(len(keys))
            total_loss = 0.0
            for i in perm:
                loss = self.train_step(keys[i].unsqueeze(0), values[i].unsqueeze(0), optimizer)
                total_loss += loss
            scheduler.step()
            avg_loss = total_loss / len(keys)
            losses.append(avg_loss)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")
        return losses

    def exact_match_accuracy(self, test_keys, test_values):
        correct = 0
        for k, v in zip(test_keys, test_values):
            retrieved = self.get(k, top_k=1)
            if retrieved and retrieved[0] == v:
                correct += 1
        return correct / len(test_keys) if test_keys else 0.0


# -----------------------------------------------------------------------------
# Hierarchical SWSTM (Router‑Expert)
# -----------------------------------------------------------------------------

class HierarchicalSWSTM(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        slots_per_expert: int,
        key_dim: int = 384,
        val_dim: int = 384,
        temperature: float = 0.01,
        margin: float = 0.2,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.slots_per_expert = slots_per_expert
        self.key_dim = key_dim
        self.val_dim = val_dim

        self.register_buffer("router_centroids", None)  # set after fit_router()
        self.experts = nn.ModuleList([
            FlatSWSTM(slots_per_expert, key_dim, val_dim, temperature, margin)
            for _ in range(num_clusters)
        ])

        self.fact_count = 0
        self.global_value_map: Dict[int, str] = {}  # global slot index → string

    def fit_router(self, all_keys: torch.Tensor):
        """Fit K‑Means centroids on all key embeddings."""
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init=10)
        kmeans.fit(all_keys.numpy())
        self.router_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    def add(self, keys: torch.Tensor, values: torch.Tensor, value_strings: List[str]):
        """Add a batch of key‑value pairs, routing to experts."""
        if self.router_centroids is None:
            raise RuntimeError("Call fit_router() before adding facts.")

        dists = torch.cdist(keys, self.router_centroids)
        cluster_ids = torch.argmin(dists, dim=-1)

        for c in range(self.num_clusters):
            mask = (cluster_ids == c)
            if mask.any():
                idx = self.experts[c].forward(keys[mask], values[mask], op="write")
                # store global mapping
                for i, slot in enumerate(idx.tolist()):
                    global_idx = c * self.slots_per_expert + slot
                    self.global_value_map[global_idx] = value_strings[mask.nonzero()[i].item()]
        self.fact_count += keys.size(0)

    def get(self, query_vec: torch.Tensor, top_k: int = 1) -> List[str]:
        if self.router_centroids is None:
            return []
        dists = torch.cdist(query_vec.unsqueeze(0), self.router_centroids)
        c = torch.argmin(dists).item()
        results = self.experts[c].get(query_vec, top_k=top_k)
        return results


# -----------------------------------------------------------------------------
# Unified Engine (drop‑in replacement for ExactMemory)
# -----------------------------------------------------------------------------

class SWSTMEngine:
    """
    Unified interface for SWSTM. Auto‑selects flat / hierarchical / PQ based on
    fact count (configurable). Mirrors ExactMemory API.
    """

    def __init__(
        self,
        mode: str = "auto",
        flat_num_slots: int = 200,
        hierarchical_num_clusters: int = 50,
        hierarchical_slots_per_expert: int = 2000,
        key_dim: int = 384,
        temperature: float = 0.01,
        margin: float = 0.2,
        auto_threshold_flat: int = 1000,
        auto_threshold_hier: int = 50000,
        encoder_model: str = "all-MiniLM-L6-v2",
    ):
        self.mode = mode
        self.flat_num_slots = flat_num_slots
        self.hierarchical_num_clusters = hierarchical_num_clusters
        self.hierarchical_slots_per_expert = hierarchical_slots_per_expert
        self.key_dim = key_dim
        self.temperature = temperature
        self.margin = margin
        self.auto_threshold_flat = auto_threshold_flat
        self.auto_threshold_hier = auto_threshold_hier

        self.memory = None
        self.fact_count = 0
        self.encoder = SentenceTransformer(encoder_model)
        self.pending_keys = []      # for batch router fitting
        self.pending_values = []    # for batch router fitting

    def _initialize(self, num_facts: int):
        if self.mode == "flat" or (self.mode == "auto" and num_facts <= self.auto_threshold_flat):
            slots = max(self.flat_num_slots, num_facts * 2)
            self.memory = FlatSWSTM(
                num_slots=slots,
                key_dim=self.key_dim,
                val_dim=self.key_dim,
                temperature=self.temperature,
                margin=self.margin,
            )
            print(f"[SWSTM] Flat mode: {slots} slots.")

        elif self.mode == "hierarchical" or (self.mode == "auto" and num_facts <= self.auto_threshold_hier):
            self.memory = HierarchicalSWSTM(
                num_clusters=self.hierarchical_num_clusters,
                slots_per_expert=self.hierarchical_slots_per_expert,
                key_dim=self.key_dim,
                val_dim=self.key_dim,
                temperature=self.temperature,
                margin=self.margin,
            )
            print(f"[SWSTM] Hierarchical mode: {self.hierarchical_num_clusters} experts, "
                  f"{self.hierarchical_slots_per_expert} slots each.")

        else:
            # PQ mode – we'll implement later (v18.1)
            raise NotImplementedError("PQ mode not yet implemented in this version.")

    def _prepare_batch(self):
        """If pending facts exist, fit router and add them all."""
        if not self.pending_keys:
            return
        if isinstance(self.memory, HierarchicalSWSTM):
            all_keys = torch.stack(self.pending_keys)
            self.memory.fit_router(all_keys)
            # Add all facts
            keys_batch = torch.stack(self.pending_keys)
            vals_batch = keys_batch  # we use key embedding as value
            self.memory.add(keys_batch, vals_batch, self.pending_values)
            self.pending_keys.clear()
            self.pending_values.clear()

    def add(self, key: Union[str, dict, bytes, torch.Tensor], value: str) -> str:
        """Add a fact. Returns confirmation string."""
        # Encode key to vector
        if isinstance(key, str):
            vec = self.encoder.encode(key, convert_to_tensor=True)
        elif isinstance(key, dict):
            vec = self.encoder.encode(json.dumps(key), convert_to_tensor=True)
        elif isinstance(key, bytes):
            vec = self.encoder.encode(key.decode("utf-8"), convert_to_tensor=True)
        elif isinstance(key, torch.Tensor):
            vec = key
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

        if self.memory is None:
            self._initialize(1)

        # If hierarchical, we need to collect facts before fitting router.
        if isinstance(self.memory, HierarchicalSWSTM):
            self.pending_keys.append(vec)
            self.pending_values.append(value)
            # If we've collected enough, we could auto‑fit, but we leave it to the user.
            # For simplicity, we fit when we hit 100 facts or on explicit call.
            if len(self.pending_keys) >= 100:
                self._prepare_batch()
        else:
            # Flat: add immediately
            self.memory.add(vec, value)

        self.fact_count += 1
        return f"Added fact #{self.fact_count}"

    def get(self, key: Union[str, torch.Tensor], top_k: int = 1) -> List[str]:
        """Retrieve value(s) for a query."""
        if self.memory is None:
            return []

        # Flush pending facts before reading
        self._prepare_batch()

        if isinstance(key, str):
            vec = self.encoder.encode(key, convert_to_tensor=True)
        elif isinstance(key, torch.Tensor):
            vec = key
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

        if isinstance(self.memory, FlatSWSTM):
            return self.memory.get(vec, top_k=top_k)
        elif isinstance(self.memory, HierarchicalSWSTM):
            return self.memory.get(vec, top_k=top_k)
        else:
            return []

    def train(self, epochs: int = 100, lr: float = 0.001):
        """Train the underlying flat memory (if applicable)."""
        if isinstance(self.memory, FlatSWSTM):
            # We need all keys and values that were added. For simplicity, we
            # assume the user will call this after adding facts.
            # This would require storing training data; we skip for now.
            warnings.warn("Training not fully implemented in Engine; use FlatSWSTM directly.")
        else:
            print("Training only supported for flat SWSTM.")

    def exact_match_accuracy(self, test_keys: List[str], test_values: List[str]) -> float:
        """Compute exact match accuracy on a test set."""
        correct = 0
        for k, v in zip(test_keys, test_values):
            retrieved = self.get(k, top_k=1)
            if retrieved and retrieved[0] == v:
                correct += 1
        return correct / len(test_keys) if test_keys else 0.0

    def fit_router(self):
        """Explicitly fit router for hierarchical mode (flush pending)."""
        if isinstance(self.memory, HierarchicalSWSTM):
            self._prepare_batch()
        else:
            print("fit_router() only needed for hierarchical mode.")
