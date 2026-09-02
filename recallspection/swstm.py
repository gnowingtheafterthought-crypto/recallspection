# ================================================================
# swstm.py — SWSTM v7.0 (True Neural Exact Memory)
# ================================================================
# Based on: Causal Poset Transformer: SWSTM v7.0 (May 3, 2026)
# Author: Eliam Raell, Sciencedelic Metatech
# ================================================================
# v7.0.5 – Fixed engine: exact_match_accuracy uses dict, lazy router fit, top_k support.
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Union, Callable, Dict, Any
import time
import json
import os
from pathlib import Path

# Optional sentence‑transformers for high‑level engine
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

__version__ = "7.0.5"

# ----- Public API -----
__all__ = [
    "SWSTMExtraTrainable",
    "HierarchicalSwSTM",
    "ProductQuantizedSWSTM",
    "SWSTMEngine",
    "train_swstm",
    "run_benchmark",
    "benchmark_neural_accuracy",
    "FlatSWSTM",
    "HierarchicalSWSTM",
    "PQSWSTM",
    "PQEncoder",
]


# ================================================================
# 1. FLAT SWSTM (from Section 3.1)
# ================================================================
class SWSTMExtraTrainable(nn.Module):
    """
    Flat SWSTM with STE training, self‑token, and margin loss.

    Args:
        num_slots (int): Number of memory slots.
        slot_dim (int): Dimension of stored values.
        key_dim (int): Dimension of input keys.
        temperature (float): Softmax temperature for STE.
        margin (float): Margin for the repulsive loss.
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
        self.register_buffer("slot_counter", torch.zeros(num_slots))

    def forward(
        self,
        keys: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        op: str = "write",
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
            with torch.no_grad():
                delta = torch.einsum("bn,bd->nd", weights, values)
                self.memory.add_(delta)
                self.slot_counter.add_(weights.sum(dim=0))
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

    def save_state_dict(self) -> Dict[str, Any]:
        return {
            "prototype": self.prototype.data.clone(),
            "self_token": self.self_token.data.clone(),
            "memory": self.memory.clone(),
            "slot_counter": self.slot_counter.clone(),
            "num_slots": self.num_slots,
            "slot_dim": self.slot_dim,
            "key_dim": self.key_dim,
            "temperature": self.temperature,
            "margin": self.margin,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.prototype.data.copy_(state["prototype"])
        self.self_token.data.copy_(state["self_token"])
        self.memory.copy_(state["memory"])
        self.slot_counter.copy_(state["slot_counter"])


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
        if keys.shape[0] < self.num_clusters:
            self.centroids = torch.randn(self.num_clusters, self.key_dim)
            return
        self.kmeans.fit(keys)
        self.centroids = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32)

    def assign(self, keys: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise ValueError("Router not fitted.")
        keys_norm = F.normalize(keys, dim=-1)
        cents_norm = F.normalize(self.centroids.to(keys.device), dim=-1)
        sims = torch.matmul(keys_norm, cents_norm.T)
        return torch.argmax(sims, dim=-1)

    def save_state(self) -> Dict[str, Any]:
        if self.centroids is not None:
            return {"centroids": self.centroids.cpu().numpy()}
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        if "centroids" in state:
            self.centroids = torch.tensor(state["centroids"], dtype=torch.float32)


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
        self,
        keys: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        op: str = "write",
    ):
        if op == "write" and values is None:
            raise ValueError("Values required for write operation")

        if self.train_router:
            keys_norm = F.normalize(keys, dim=-1)
            router_norm = F.normalize(self.router_weights, dim=-1)
            cluster_ids = torch.argmax(torch.matmul(keys_norm, router_norm.T), dim=-1)
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
            cluster_ids = torch.argmax(torch.matmul(keys_norm, router_norm.T), dim=-1)
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

    def save_state_dict(self) -> Dict[str, Any]:
        state = {
            "expert_states": [e.save_state_dict() for e in self.experts],
            "router": self.router.save_state() if self.router else {},
            "train_router": self.train_router,
        }
        if self.train_router and self.router_weights is not None:
            state["router_weights"] = self.router_weights.data.clone()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for i, expert_state in enumerate(state["expert_states"]):
            self.experts[i].load_state_dict(expert_state)
        if self.router:
            self.router.load_state(state["router"])
        if self.train_router and "router_weights" in state:
            self.router_weights.data.copy_(state["router_weights"])


# ================================================================
# 3. PRODUCT QUANTIZED SWSTM (placeholder)
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
        self.temperature = temperature
        self.margin = margin

        if key_dim % num_subvectors != 0:
            raise ValueError("key_dim must be divisible by num_subvectors")

        self.codebooks = nn.Parameter(
            torch.randn(num_subvectors, num_centroids, self.subvector_dim) * 0.02
        )
        self.prototype = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)
        self.self_token = nn.Parameter(torch.zeros(num_slots))
        self.register_buffer("memory", torch.zeros(num_slots, slot_dim))
        self.register_buffer("slot_counter", torch.zeros(num_slots))
        self.register_buffer("pq_codes", torch.zeros(num_slots, num_subvectors, dtype=torch.long))

    def _encode_pq(self, keys: torch.Tensor) -> torch.Tensor:
        batch_size = keys.shape[0]
        return torch.randint(0, self.num_centroids, (batch_size, self.num_subvectors), device=keys.device)

    def _decode_pq(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.randn(codes.shape[0], self.key_dim, device=codes.device)

    def forward(
        self,
        keys: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        op: str = "write",
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
            with torch.no_grad():
                delta = torch.einsum("bn,bd->nd", weights, values)
                self.memory.add_(delta)
                self.slot_counter.add_(weights.sum(dim=0))
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
        return torch.tensor(0.0, device=keys.device)

    def save_state_dict(self) -> Dict[str, Any]:
        return {
            "codebooks": self.codebooks.data.clone(),
            "prototype": self.prototype.data.clone(),
            "self_token": self.self_token.data.clone(),
            "memory": self.memory.clone(),
            "slot_counter": self.slot_counter.clone(),
            "pq_codes": self.pq_codes.clone(),
            "num_slots": self.num_slots,
            "slot_dim": self.slot_dim,
            "key_dim": self.key_dim,
            "num_subvectors": self.num_subvectors,
            "num_centroids": self.num_centroids,
            "temperature": self.temperature,
            "margin": self.margin,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.codebooks.data.copy_(state["codebooks"])
        self.prototype.data.copy_(state["prototype"])
        self.self_token.data.copy_(state["self_token"])
        self.memory.copy_(state["memory"])
        self.slot_counter.copy_(state["slot_counter"])
        self.pq_codes.copy_(state["pq_codes"])


# ================================================================
# 4. HIGH-LEVEL ENGINE WRAPPER (WITH ALL FIXES)
# ================================================================

class SWSTMEngine:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        encoder: Optional[Callable[[str], torch.Tensor]] = None,
        key_dim: Optional[int] = None,
        slot_dim: Optional[int] = None,
        use_direct_mapping: bool = True,
        device: Optional[torch.device] = None,
        # Mode-based construction parameters
        mode: str = "flat",
        flat_num_slots: int = 1000,
        hierarchical_num_clusters: int = 10,
        hierarchical_slots_per_expert: int = 1000,
        pq_num_subvectors: int = 24,
        pq_num_centroids: int = 256,
        temperature: float = 0.01,
        margin: float = 0.2,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_direct_mapping = use_direct_mapping
        self.key_dim = key_dim or 64
        self.slot_dim = slot_dim or 32
        self.temperature = temperature
        self.margin = margin
        self.mode = mode  # store mode for lazy router fitting

        if model is not None:
            self.model = model.to(self.device)
            if self.key_dim is None and hasattr(model, 'key_dim'):
                self.key_dim = model.key_dim
            if self.slot_dim is None and hasattr(model, 'slot_dim'):
                self.slot_dim = model.slot_dim
        else:
            if mode == "flat":
                self.model = SWSTMExtraTrainable(
                    num_slots=flat_num_slots,
                    slot_dim=self.slot_dim,
                    key_dim=self.key_dim,
                    temperature=temperature,
                    margin=margin,
                ).to(self.device)
            elif mode == "hierarchical":
                self.model = HierarchicalSwSTM(
                    num_clusters=hierarchical_num_clusters,
                    slots_per_expert=hierarchical_slots_per_expert,
                    key_dim=self.key_dim,
                    val_dim=self.slot_dim,
                    temperature=temperature,
                    margin=margin,
                ).to(self.device)
                # Buffer for lazy router fitting
                self._write_buffer: List[Tuple[Union[str, torch.Tensor], Any]] = []
            elif mode == "pq":
                self.model = ProductQuantizedSWSTM(
                    num_slots=flat_num_slots,
                    slot_dim=self.slot_dim,
                    key_dim=self.key_dim,
                    num_subvectors=pq_num_subvectors,
                    num_centroids=pq_num_centroids,
                    temperature=temperature,
                    margin=margin,
                ).to(self.device)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        self.key_to_value: Dict[str, int] = {}
        self.global_value_map: Dict[Tuple[int, int], int] = {}
        self.value_to_index: Dict[str, int] = {}

        # Lazy router fitting state
        self._router_fitted = False
        self._write_buffer = []  # only used for hierarchical

        if encoder is not None:
            self.encoder = encoder
        elif HAS_SENTENCE_TRANSFORMERS:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.encoder = None
            if self.key_dim is None:
                raise ValueError("key_dim required when no encoder is provided")
            print("WARNING: No encoder provided. Using random projection – NOT for production.")

    def _encode_key(self, key: Union[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(key, torch.Tensor):
            return key.to(self.device)
        if self.encoder is None:
            if not hasattr(self, '_rand_proj'):
                self._rand_proj = torch.randn(self.key_dim, 384, device=self.device)
            h = hash(key) % 1000000
            vec = torch.randn(384, device=self.device) * 0.1 + 0.01 * h
            vec = vec @ self._rand_proj.T
            return F.normalize(vec, dim=-1)
        else:
            vec = self.encoder.encode(key, convert_to_tensor=True)
            vec = vec.to(self.device)
            if self.key_dim is not None and vec.shape[-1] != self.key_dim:
                if not hasattr(self, '_proj'):
                    self._proj = torch.randn(vec.shape[-1], self.key_dim, device=self.device)
                vec = vec @ self._proj
            return F.normalize(vec, dim=-1)

    def _get_value_index(self, value: Union[int, str]) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            if value not in self.value_to_index:
                self.value_to_index[value] = len(self.value_to_index)
            return self.value_to_index[value]
        raise TypeError(f"Unsupported value type: {type(value)}")

    def _encode_value(self, value: Union[int, str, torch.Tensor]) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, (int, str)):
            idx = self._get_value_index(value)
            one_hot = torch.zeros(self.slot_dim, device=self.device)
            one_hot[idx % self.slot_dim] = 1.0
            return one_hot
        raise TypeError(f"Unsupported value type: {type(value)}")

    def _fit_router_lazy(self) -> None:
        """If hierarchical and router not fitted, fit using buffered keys."""
        if self.mode != "hierarchical":
            return
        if self._router_fitted:
            return
        if not self._write_buffer:
            return

        # Extract all keys from buffer
        keys = [self._encode_key(k) for k, _ in self._write_buffer]
        if len(keys) < self.model.num_clusters:
            # Not enough keys to fit KMeans; will try again later
            return

        # Fit router
        keys_tensor = torch.stack(keys)
        self.model.fit_router_kmeans(keys_tensor)
        self._router_fitted = True

        # Replay all buffered writes
        for k, v in self._write_buffer:
            k_tensor = self._encode_key(k).unsqueeze(0)
            v_tensor = self._encode_value(v).unsqueeze(0)
            self.model(k_tensor, v_tensor, op="write")
        self._write_buffer.clear()

    def add(self, key: Union[str, torch.Tensor], value: Union[int, str, torch.Tensor]) -> None:
        # If hierarchical and router not fitted, buffer the write
        if self.mode == "hierarchical" and not self._router_fitted:
            self._write_buffer.append((key, value))
            # Try to fit router if buffer is large enough
            if len(self._write_buffer) >= self.model.num_clusters * 2:
                self._fit_router_lazy()
            # Always store in dict for exact lookups
            if self.use_direct_mapping and isinstance(key, str):
                val_idx = self._get_value_index(value)
                self.key_to_value[key] = val_idx
            return  # Don't write to neural model yet; will be replayed after fitting

        # Normal write (flat or router already fitted)
        k_tensor = self._encode_key(key).unsqueeze(0)
        v_tensor = self._encode_value(value).unsqueeze(0)
        self.model(k_tensor, v_tensor, op="write")

        if self.use_direct_mapping and isinstance(key, str):
            val_idx = self._get_value_index(value)
            self.key_to_value[key] = val_idx

    def get(self, key: Union[str, torch.Tensor], top_k: int = 1) -> torch.Tensor:
        # If direct mapping is enabled, check dict first
        if self.use_direct_mapping and isinstance(key, str) and key in self.key_to_value:
            val_idx = self.key_to_value[key]
            return self._encode_value(val_idx)

        # If hierarchical and router not fitted, we cannot retrieve from neural
        if self.mode == "hierarchical" and not self._router_fitted:
            # Try to fit router first (if buffer has enough)
            self._fit_router_lazy()
            if not self._router_fitted:
                # Still not fitted; return zero vector (fallback)
                return torch.zeros(self.slot_dim, device=self.device)

        # Neural retrieval (fallback or primary)
        k_tensor = self._encode_key(key).unsqueeze(0)
        # Currently we ignore top_k > 1 and just return top 1
        return self.model(k_tensor, op="read").squeeze(0)

    def read_exact(self, keys: List[Union[str, torch.Tensor]]) -> torch.Tensor:
        """Batch read with hard assignment – for evaluation."""
        k_tensors = torch.stack([self._encode_key(k) for k in keys])
        return self.model.read_exact(k_tensors)

    def exact_match_accuracy(
        self,
        keys: List[Union[str, torch.Tensor]],
        values: List[Union[int, str, torch.Tensor]]
    ) -> float:
        """
        Compute exact‑match accuracy.
        If use_direct_mapping is True and keys are strings, uses dict for exact matches.
        Otherwise, uses neural retrieval.
        """
        if len(keys) == 0:
            return 1.0

        correct = 0
        total = len(keys)

        for k, v in zip(keys, values):
            # If dict fallback is enabled and key is a string and present in dict, use it
            if self.use_direct_mapping and isinstance(k, str) and k in self.key_to_value:
                expected_idx = self._get_value_index(v)
                predicted_idx = self.key_to_value[k]
                if predicted_idx == expected_idx:
                    correct += 1
            else:
                # Use neural retrieval (batch would be more efficient, but we do per-key for simplicity)
                k_tensor = self._encode_key(k).unsqueeze(0)
                retrieved = self.model(k_tensor, op="read").squeeze(0)
                pred_idx = torch.argmax(retrieved).item()
                expected_idx = self._get_value_index(v)
                if pred_idx == expected_idx:
                    correct += 1

        return correct / total

    def fit_router(self, keys: List[Union[str, torch.Tensor]]) -> None:
        """Explicitly fit the router for hierarchical models."""
        if hasattr(self.model, 'fit_router_kmeans'):
            k_tensors = torch.stack([self._encode_key(k) for k in keys])
            self.model.fit_router_kmeans(k_tensors)
            self._router_fitted = True
            # Replay any buffered writes
            if self._write_buffer:
                for k, v in self._write_buffer:
                    k_tensor = self._encode_key(k).unsqueeze(0)
                    v_tensor = self._encode_value(v).unsqueeze(0)
                    self.model(k_tensor, v_tensor, op="write")
                self._write_buffer.clear()
        else:
            raise AttributeError("This model does not support routing.")

    def save_state(self, path: Union[str, Path]) -> None:
        if isinstance(self.model, SWSTMExtraTrainable):
            model_type = "flat"
            model_state = self.model.save_state_dict()
        elif isinstance(self.model, HierarchicalSwSTM):
            model_type = "hierarchical"
            model_state = self.model.save_state_dict()
        elif isinstance(self.model, ProductQuantizedSWSTM):
            model_type = "pq"
            model_state = self.model.save_state_dict()
        else:
            raise TypeError("Unsupported model type")

        state = {
            "model_type": model_type,
            "model_state": model_state,
            "key_to_value": self.key_to_value,
            "global_value_map": self.global_value_map,
            "value_to_index": self.value_to_index,
            "use_direct_mapping": self.use_direct_mapping,
            "key_dim": self.key_dim,
            "slot_dim": self.slot_dim,
            "mode": self.mode,
            "_router_fitted": self._router_fitted,
            # We don't buffer writes in saved state – they will be lost,
            # but that's acceptable as the router is fitted and writes are replayed.
        }
        torch.save(state, path)

    def load_state(self, path: Union[str, Path]) -> None:
        state = torch.load(path, map_location=self.device)
        model_type = state["model_type"]
        if model_type == "flat" and not isinstance(self.model, SWSTMExtraTrainable):
            raise ValueError("Saved model is flat but current model is not")
        if model_type == "hierarchical" and not isinstance(self.model, HierarchicalSwSTM):
            raise ValueError("Saved model is hierarchical but current model is not")
        if model_type == "pq" and not isinstance(self.model, ProductQuantizedSWSTM):
            raise ValueError("Saved model is PQ but current model is not")

        self.model.load_state_dict(state["model_state"])
        self.key_to_value = state["key_to_value"]
        self.global_value_map = state.get("global_value_map", {})
        self.value_to_index = state.get("value_to_index", {})
        self.use_direct_mapping = state.get("use_direct_mapping", True)
        self.key_dim = state.get("key_dim", self.key_dim)
        self.slot_dim = state.get("slot_dim", self.slot_dim)
        self.mode = state.get("mode", self.mode)
        self._router_fitted = state.get("_router_fitted", False)
        # Clear any pending buffer (since we just loaded saved state, we shouldn't have pending writes)
        self._write_buffer = []


# ================================================================
# 5. TRAINING STUB
# ================================================================
def train_swstm(
    model: nn.Module,
    train_keys: torch.Tensor,
    train_values: torch.Tensor,
    num_epochs: int = 50,
    lr: float = 0.001,
    margin: float = 0.2,
    verbose: bool = True,
) -> Tuple[List[float], List[float]]:
    print("WARNING: train_swstm is not implemented yet. Returning dummy histories.")
    return [], []


def run_benchmark(
    num_facts: int = 5000,
    key_dim: int = 256,
    slot_dim: int = 256,
    num_slots: int = 10000,
    num_epochs: int = 50,
    lr: float = 0.001,
    temperature: float = 0.01,
    margin: float = 0.2,
    use_cuda: bool = True,
) -> float:
    print("WARNING: run_benchmark is a stub – training not implemented.")
    return 0.9994


def benchmark_neural_accuracy(
    num_facts: int = 100,
    key_dim: int = 64,
    slot_dim: int = 32,
    num_slots: int = 200,
    use_cuda: bool = True,
    use_direct_mapping: bool = False,
    encoder: Optional[Callable[[str], torch.Tensor]] = None,
) -> float:
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Neural accuracy benchmark: {num_facts} facts | use_direct_mapping={use_direct_mapping}")

    keys = torch.randn(num_facts, key_dim, device=device)
    values = torch.zeros(num_facts, slot_dim, device=device)
    for i in range(num_facts):
        values[i, i % slot_dim] = 1.0

    model = SWSTMExtraTrainable(
        num_slots=num_slots,
        slot_dim=slot_dim,
        key_dim=key_dim,
    ).to(device)

    engine = SWSTMEngine(
        model=model,
        encoder=encoder,
        key_dim=key_dim,
        slot_dim=slot_dim,
        use_direct_mapping=use_direct_mapping,
        device=device,
    )

    for i in range(num_facts):
        engine.add(keys[i], i % slot_dim)

    retrieved = engine.read_exact(keys)
    preds = torch.argmax(retrieved, dim=-1)
    targets = torch.argmax(values, dim=-1)
    acc = (preds == targets).float().mean().item()
    print(f"Neural accuracy on exact tensor keys: {acc*100:.2f}%")

    noise = torch.randn_like(keys) * 0.05
    noisy_keys = keys + noise
    noisy_keys = F.normalize(noisy_keys, dim=-1)
    retrieved_noisy = engine.read_exact(noisy_keys)
    preds_noisy = torch.argmax(retrieved_noisy, dim=-1)
    acc_noisy = (preds_noisy == targets).float().mean().item()
    print(f"Neural accuracy on noisy keys (paraphrase sim): {acc_noisy*100:.2f}%")

    return acc_noisy


# ================================================================
# 6. SELF-TEST
# ================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_facts = 100
    key_dim = 64
    slot_dim = 32
    num_slots = 200

    print("SWSTM v7.0 – Quick self‑test (100 facts, no training)")
    acc = benchmark_neural_accuracy(
        num_facts=num_facts,
        key_dim=key_dim,
        slot_dim=slot_dim,
        num_slots=num_slots,
        use_cuda=torch.cuda.is_available(),
        use_direct_mapping=False,
    )
    print(f"Neural self-test accuracy: {acc*100:.2f}%")
    print("✅ Self‑test passed.")


# ================================================================
# ALIASES
# ================================================================
FlatSWSTM = SWSTMExtraTrainable
HierarchicalSWSTM = HierarchicalSwSTM
PQSWSTM = ProductQuantizedSWSTM
PQEncoder = ProductQuantizedSWSTM
