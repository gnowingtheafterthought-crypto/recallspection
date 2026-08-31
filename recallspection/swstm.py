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
    def __init__(self, num_slots, key_dim=384, val_dim=384, temperature=0.01, margin=0.2, token_beta=0.9):
        super().__init__()
        self.num_slots = num_slots
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.temperature = temperature
        self.margin = margin
        self.token_beta = token_beta

        self.prototypes = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)
        self.self_token = nn.Parameter(torch.zeros(num_slots))
        self.register_buffer("memory", torch.zeros(num_slots, val_dim))
        self.register_buffer("used", torch.zeros(num_slots, dtype=torch.bool))
        self.register_buffer("slot_count", torch.zeros(num_slots, dtype=torch.long))
        self.fact_count = 0
        self.value_map: Dict[int, str] = {}

    def _get_weights(self, keys, with_grad=True):
        """Compute assignment weights. If with_grad=False, uses torch.no_grad."""
        if not with_grad:
            keys = keys.detach()
        device = keys.device
        norm_keys = F.normalize(keys, dim=-1)
        norm_proto = F.normalize(self.prototypes.to(device), dim=-1)
        self_token = self.self_token.to(device)
        sims = torch.mm(norm_keys, norm_proto.T) + self_token.unsqueeze(0)
        soft_w = F.softmax(sims / self.temperature, dim=-1)
        hard_idx = torch.argmax(soft_w, dim=-1)
        one_hot = F.one_hot(hard_idx, num_classes=self.num_slots).float()
        weights = one_hot.detach() + (soft_w - soft_w.detach())  # STE
        return weights, sims

    def write(self, keys, values):
        """Write phase: no gradients."""
        with torch.no_grad():
            # For inference, values are tensors, but we store them in memory.
            # Here we assume keys and values are already on the right device.
            weights, _ = self._get_weights(keys, with_grad=False)
            delta = torch.einsum('bn,bd->nd', weights, values)
            self.memory = self.memory + delta
            # Mark used slots
            hard_idx = torch.argmax(weights, dim=-1)
            self.used[hard_idx] = True
            self.slot_count[hard_idx] += 1
            self.fact_count += keys.size(0)

    def read(self, keys):
        """Read phase: gradients flow through this."""
        weights, _ = self._get_weights(keys, with_grad=True)
        return torch.matmul(weights, self.memory)

    def read_exact(self, keys):
        """Hard read for validation (no gradients)."""
        with torch.no_grad():
            weights, _ = self._get_weights(keys, with_grad=False)
            return torch.matmul(weights, self.memory)

    def margin_loss(self, keys):
        """Compute margin loss from similarities."""
        _, sims = self._get_weights(keys, with_grad=True)
        top1, _ = sims.topk(1, dim=-1)
        top2, _ = sims.topk(2, dim=-1)
        return torch.mean(torch.relu(self.margin - (top1.squeeze() - top2[:, 1])))

    def add(self, key_vec: torch.Tensor, value_str: str) -> int:
        """Add a single fact (for inference)."""
        if key_vec.dim() == 2 and key_vec.size(0) == 1:
            key_vec = key_vec.squeeze(0)
        if not key_vec.requires_grad:
            key_vec = key_vec.clone().detach().requires_grad_(True)
        val_vec = key_vec  # store key embedding as value
        # Use write (no grad) to store in memory
        self.write(key_vec.unsqueeze(0), val_vec.unsqueeze(0))
        # Get the slot index used (hard assignment)
        with torch.no_grad():
            weights, _ = self._get_weights(key_vec.unsqueeze(0), with_grad=False)
            idx = torch.argmax(weights, dim=-1).item()
        self.value_map[idx] = value_str
        # **Crucial:** Set self_token to 1.0 for this slot to guarantee retrieval.
        with torch.no_grad():
            self.self_token[idx] = 1.0
        return idx

    def get(self, query_vec: torch.Tensor, top_k: int = 1) -> List[str]:
        """Retrieve values (inference)."""
        if not query_vec.requires_grad:
            query_vec = query_vec.clone()
        device = query_vec.device
        # Use prototypes + self_token for similarity (same as write)
        norm_q = F.normalize(query_vec.unsqueeze(0), dim=-1)
        norm_proto = F.normalize(self.prototypes.to(device), dim=-1)
        self_token = self.self_token.to(device)
        sims = torch.mm(norm_q, norm_proto.T) + self_token.unsqueeze(0)
        sims = sims.masked_fill(~self.used.unsqueeze(0), -1e9)
        top_scores, top_indices = torch.topk(sims, min(top_k, self.fact_count), dim=-1)
        results = []
        for idx in top_indices.squeeze(0).tolist():
            if idx in self.value_map:
                results.append(self.value_map[idx])
        return results

    def train_step(self, keys, values, optimizer):
        # Write phase (no grad)
        self.write(keys, values)
        # Read phase (grad)
        read_values = self.read(keys)
        recon_loss = F.mse_loss(read_values, values)
        margin_loss = self.margin_loss(keys)
        total_loss = recon_loss + 0.1 * margin_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return total_loss.item()

    def train_epoch(self, keys, values, epochs=100, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            perm = torch.randperm(len(keys))
            total_loss = 0.0
            for i in perm:
                loss = self.train_step(keys[i].unsqueeze(0), values[i].unsqueeze(0), optimizer)
                total_loss += loss
            scheduler.step()
            avg_loss = total_loss / len(keys)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")

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

        self.register_buffer("router_centroids", None)
        self.experts = nn.ModuleList([
            FlatSWSTM(slots_per_expert, key_dim, val_dim, temperature, margin)
            for _ in range(num_clusters)
        ])

        self.fact_count = 0
        self.global_value_map: Dict[int, str] = {}

    def fit_router(self, all_keys: torch.Tensor):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init=10)
        kmeans.fit(all_keys.numpy())
        self.router_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    def add(self, keys: torch.Tensor, values: torch.Tensor, value_strings: List[str]):
        if self.router_centroids is None:
            raise RuntimeError("Call fit_router() before adding facts.")
        dists = torch.cdist(keys, self.router_centroids)
        cluster_ids = torch.argmin(dists, dim=-1)

        for c in range(self.num_clusters):
            mask = (cluster_ids == c)
            if mask.any():
                slot_indices = self.experts[c].forward(keys[mask], values[mask], op="write")
                for i, slot_idx in enumerate(slot_indices.tolist()):
                    global_idx = c * self.slots_per_expert + slot_idx
                    self.global_value_map[global_idx] = value_strings[mask.nonzero()[i].item()]
                    self.experts[c].value_map[slot_idx] = value_strings[mask.nonzero()[i].item()]
        self.fact_count += keys.size(0)

    def get(self, query_vec: torch.Tensor, top_k: int = 1) -> List[str]:
        if self.router_centroids is None:
            return []
        dists = torch.cdist(query_vec.unsqueeze(0), self.router_centroids)
        c = torch.argmin(dists).item()
        return self.experts[c].get(query_vec, top_k=top_k)

    def exact_match_accuracy(self, test_keys, test_values):
        correct = 0
        for k, v in zip(test_keys, test_values):
            retrieved = self.get(k, top_k=1)
            if retrieved and retrieved[0] == v:
                correct += 1
        return correct / len(test_keys) if test_keys else 0.0


# -----------------------------------------------------------------------------
# Product Quantization (PQ) for Million-Scale
# -----------------------------------------------------------------------------

class PQEncoder:
    def __init__(self, dim: int, num_subvectors: int = 24, num_centroids: int = 256):
        self.dim = dim
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids
        assert dim % num_subvectors == 0
        self.sub_dim = dim // num_subvectors
        self.codebooks = None

    def fit(self, vectors: torch.Tensor):
        from sklearn.cluster import KMeans
        vectors = vectors.reshape(-1, self.num_subvectors, self.sub_dim)
        codebooks = []
        for m in range(self.num_subvectors):
            subvecs = vectors[:, m, :].numpy()
            kmeans = KMeans(n_clusters=self.num_centroids, random_state=0, n_init=10)
            kmeans.fit(subvecs)
            codebooks.append(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
        self.codebooks = torch.stack(codebooks, dim=0)

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        vectors = vectors.reshape(-1, self.num_subvectors, self.sub_dim)
        codes = []
        for m in range(self.num_subvectors):
            dists = torch.cdist(vectors[:, m, :], self.codebooks[m])
            idx = torch.argmin(dists, dim=-1)
            codes.append(idx)
        return torch.stack(codes, dim=-1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        vectors = []
        for m in range(self.num_subvectors):
            centroids = self.codebooks[m][codes[:, m]]
            vectors.append(centroids)
        return torch.cat(vectors, dim=-1)


class PQSWSTM:
    def __init__(
        self,
        num_clusters: int = 1000,
        facts_per_cluster: int = 1000,
        dim: int = 384,
        num_subvectors: int = 24,
    ):
        self.num_clusters = num_clusters
        self.facts_per_cluster = facts_per_cluster
        self.dim = dim
        self.num_subvectors = num_subvectors
        self.router_centroids = None
        self.pq_encoder = PQEncoder(dim, num_subvectors)
        self.compressed_keys = None
        self.value_map = {}
        self.fact_count = 0

    def fit(self, all_keys: torch.Tensor, all_values: List[str]):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init=10)
        kmeans.fit(all_keys.numpy())
        self.router_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self.pq_encoder.fit(all_keys)
        codes = self.pq_encoder.encode(all_keys)
        self.compressed_keys = codes.to(torch.uint8)
        self.value_map = {i: v for i, v in enumerate(all_values)}
        self.fact_count = len(all_values)

    def retrieve(self, query_vec: torch.Tensor, top_k: int = 1) -> List[str]:
        if self.router_centroids is None or self.compressed_keys is None:
            return []
        norm_q = F.normalize(query_vec.unsqueeze(0), dim=-1)
        norm_router = F.normalize(self.router_centroids, dim=-1)
        sims = torch.mm(norm_q, norm_router.T)
        cluster_id = torch.argmax(sims, dim=-1).item()
        start = cluster_id * self.facts_per_cluster
        end = min(start + self.facts_per_cluster, self.compressed_keys.size(0))
        codes = self.compressed_keys[start:end].long()
        decoded = self.pq_encoder.decode(codes)
        norm_decoded = F.normalize(decoded, dim=-1)
        sims = torch.mm(norm_q, norm_decoded.T)
        top_scores, top_indices = torch.topk(sims, min(top_k, sims.size(1)), dim=-1)
        results = []
        for idx in top_indices.squeeze(0).tolist():
            global_idx = start + idx
            if global_idx in self.value_map:
                results.append(self.value_map[global_idx])
        return results


# -----------------------------------------------------------------------------
# Unified Engine
# -----------------------------------------------------------------------------

class SWSTMEngine:
    def __init__(
        self,
        mode: str = "auto",
        flat_num_slots: int = 200,
        hierarchical_num_clusters: int = 50,
        hierarchical_slots_per_expert: int = 2000,
        pq_num_clusters: int = 1000,
        pq_facts_per_cluster: int = 1000,
        pq_num_subvectors: int = 24,
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
        self.pq_num_clusters = pq_num_clusters
        self.pq_facts_per_cluster = pq_facts_per_cluster
        self.pq_num_subvectors = pq_num_subvectors
        self.key_dim = key_dim
        self.temperature = temperature
        self.margin = margin
        self.auto_threshold_flat = auto_threshold_flat
        self.auto_threshold_hier = auto_threshold_hier

        self.memory = None
        self.fact_count = 0
        self.encoder = SentenceTransformer(encoder_model)
        self.pending_keys = []
        self.pending_values = []

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
            self.memory = PQSWSTM(
                num_clusters=self.pq_num_clusters,
                facts_per_cluster=self.pq_facts_per_cluster,
                dim=self.key_dim,
                num_subvectors=self.pq_num_subvectors,
            )
            print(f"[SWSTM] PQ mode: {self.pq_num_clusters} clusters, "
                  f"{self.pq_facts_per_cluster} facts each.")

    def _prepare_batch(self):
        if not self.pending_keys:
            return
        if isinstance(self.memory, HierarchicalSWSTM):
            all_keys = torch.stack(self.pending_keys)
            self.memory.fit_router(all_keys)
            keys_batch = torch.stack(self.pending_keys)
            vals_batch = keys_batch
            self.memory.add(keys_batch, vals_batch, self.pending_values)
            self.pending_keys.clear()
            self.pending_values.clear()
        elif isinstance(self.memory, PQSWSTM):
            pass  # wait for fit_pq()

    def add(self, key: Union[str, dict, bytes, torch.Tensor], value: str) -> str:
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

        if isinstance(self.memory, (HierarchicalSWSTM, PQSWSTM)):
            self.pending_keys.append(vec)
            self.pending_values.append(value)
            if isinstance(self.memory, HierarchicalSWSTM) and len(self.pending_keys) >= 100:
                self._prepare_batch()
        else:
            if vec.dim() == 2 and vec.size(0) == 1:
                vec = vec.squeeze(0)
            self.memory.add(vec, value)

        self.fact_count += 1
        return f"Added fact #{self.fact_count}"

    def get(self, key: Union[str, torch.Tensor], top_k: int = 1) -> List[str]:
        if self.memory is None:
            return []
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
        elif isinstance(self.memory, PQSWSTM):
            return self.memory.retrieve(vec, top_k=top_k)
        return []

    def fit_pq(self):
        if isinstance(self.memory, PQSWSTM) and self.pending_keys:
            all_keys = torch.stack(self.pending_keys)
            all_values = self.pending_values
            self.memory.fit(all_keys, all_values)
            self.pending_keys.clear()
            self.pending_values.clear()
            print(f"[SWSTM] PQ fitted with {self.memory.fact_count} facts.")

    def fit_router(self):
        if isinstance(self.memory, HierarchicalSWSTM):
            self._prepare_batch()
        else:
            print("fit_router() only needed for hierarchical mode.")

    def exact_match_accuracy(self, test_keys: List[str], test_values: List[str]) -> float:
        correct = 0
        for k, v in zip(test_keys, test_values):
            retrieved = self.get(k, top_k=1)
            if retrieved and retrieved[0] == v:
                correct += 1
        return correct / len(test_keys) if test_keys else 0.0

    def train(self, epochs: int = 100, lr: float = 0.001):
        if isinstance(self.memory, FlatSWSTM):
            warnings.warn("Training not fully implemented in Engine; use FlatSWSTM directly.")
        else:
            print("Training only supported for flat SWSTM.")

__all__ = [
    "FlatSWSTM",
    "HierarchicalSWSTM",
    "PQEncoder",
    "PQSWSTM",
    "SWSTMEngine",
]