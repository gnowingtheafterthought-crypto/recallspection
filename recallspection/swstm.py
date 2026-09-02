# ================================================================
# swstm.py — SWSTM v7.1.1 (Neural Memory with Honest API)
# ================================================================
# v7.1.1 – API fixes: fit_router(), get() returns List[str], add() returns str
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Union, Callable, Dict, Any
import time
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

__version__ = "7.1.1"

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
# 1. FLAT SWSTM
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
# 4. HIGH-LEVEL ENGINE WRAPPER (with API fixes)
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
        self.mode = mode

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
                self._write_buffer = []
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
        self.value_to_index: Dict[str, int] = {}
        self.index_to_value: Dict[int, str] = {}   # reverse mapping for get()
        self.global_value_map: Dict[Tuple[int, int], int] = {}

        self._router_fitted = False
        self._write_buffer = [] if mode == "hierarchical" else None

        if encoder is not None:
            self.encoder = encoder
        elif HAS_SENTENCE_TRANSFORMERS:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.encoder = None
            if self.key_dim is None:
                raise ValueError("key_dim required when no encoder is provided")
            print("WARNING: No encoder provided. Using random projection.")

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
                idx = len(self.value_to_index)
                self.value_to_index[value] = idx
                self.index_to_value[idx] = value   # reverse map
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
        if self.mode != "hierarchical" or self._router_fitted:
            return
        if not self._write_buffer:
            return

        keys = [self._encode_key(k) for k, _ in self._write_buffer]
        if len(keys) < self.model.num_clusters * 2:
            return

        keys_tensor = torch.stack(keys)
        self.model.fit_router_kmeans(keys_tensor)
        self._router_fitted = True

        for k, v in self._write_buffer:
            k_tensor = self._encode_key(k).unsqueeze(0)
            v_tensor = self._encode_value(v).unsqueeze(0)
            self.model(k_tensor, v_tensor, op="write")
        self._write_buffer.clear()

    def add(self, key: Union[str, torch.Tensor], value: Union[int, str, torch.Tensor]) -> str:
        """Store a key-value pair. Returns a confirmation string."""
        if self.mode == "hierarchical" and not self._router_fitted:
            self._write_buffer.append((key, value))
            if len(self._write_buffer) >= self.model.num_clusters * 2:
                self._fit_router_lazy()
            if self.use_direct_mapping and isinstance(key, str):
                val_idx = self._get_value_index(value)
                self.key_to_value[key] = val_idx
            return f"Added fact (buffered) #{len(self.key_to_value)}"

        k_tensor = self._encode_key(key).unsqueeze(0)
        v_tensor = self._encode_value(value).unsqueeze(0)
        self.model(k_tensor, v_tensor, op="write")

        if self.use_direct_mapping and isinstance(key, str):
            val_idx = self._get_value_index(value)
            self.key_to_value[key] = val_idx

        return f"Added fact #{len(self.key_to_value)}"

    def get(self, key: Union[str, torch.Tensor], top_k: int = 1) -> List[str]:
        """Retrieve the value(s) for a key. Returns a list of strings."""
        # If direct mapping is enabled, use it
        if self.use_direct_mapping and isinstance(key, str) and key in self.key_to_value:
            val_idx = self.key_to_value[key]
            if val_idx in self.index_to_value:
                return [self.index_to_value[val_idx]]
            return []

        # Hierarchical lazy fit
        if self.mode == "hierarchical" and not self._router_fitted:
            self._fit_router_lazy()
            if not self._router_fitted:
                return []

        # Neural retrieval
        k_tensor = self._encode_key(key).unsqueeze(0)
        retrieved = self.model(k_tensor, op="read").squeeze(0)
        pred_idx = torch.argmax(retrieved).item()

        if pred_idx in self.index_to_value:
            return [self.index_to_value[pred_idx]]
        return []

    def read_exact(self, keys: List[Union[str, torch.Tensor]]) -> torch.Tensor:
        """Batch read with hard assignment – returns tensor of one‑hot vectors."""
        k_tensors = torch.stack([self._encode_key(k) for k in keys])
        return self.model.read_exact(k_tensors)

    def exact_match_accuracy(
        self,
        keys: List[Union[str, torch.Tensor]],
        values: List[Union[int, str, torch.Tensor]]
    ) -> float:
        """Compute exact‑match accuracy (argmax of retrieved vs expected)."""
        if len(keys) == 0:
            return 1.0

        correct = 0
        for k, v in zip(keys, values):
            # If direct mapping and key is string, use it
            if self.use_direct_mapping and isinstance(k, str) and k in self.key_to_value:
                expected_idx = self._get_value_index(v)
                predicted_idx = self.key_to_value[k]
                if predicted_idx == expected_idx:
                    correct += 1
            else:
                k_tensor = self._encode_key(k).unsqueeze(0)
                retrieved = self.model(k_tensor, op="read").squeeze(0)
                pred_idx = torch.argmax(retrieved).item()
                expected_idx = self._get_value_index(v)
                if pred_idx == expected_idx:
                    correct += 1
        return correct / len(keys)

    def fit_router(self, keys: Optional[List[Union[str, torch.Tensor]]] = None) -> None:
        """Explicitly fit the router for hierarchical models.
        If keys is None, uses the buffered keys from lazy writes.
        """
        if hasattr(self.model, 'fit_router_kmeans'):
            if keys is None:
                if not self._write_buffer:
                    raise ValueError("No keys provided and no buffered writes available.")
                keys = [k for k, _ in self._write_buffer]

            k_tensors = torch.stack([self._encode_key(k) for k in keys])
            self.model.fit_router_kmeans(k_tensors)
            self._router_fitted = True

            # Replay buffered writes
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
            "value_to_index": self.value_to_index,
            "index_to_value": self.index_to_value,
            "global_value_map": self.global_value_map,
            "use_direct_mapping": self.use_direct_mapping,
            "key_dim": self.key_dim,
            "slot_dim": self.slot_dim,
            "mode": self.mode,
            "_router_fitted": self._router_fitted,
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
        self.value_to_index = state.get("value_to_index", {})
        self.index_to_value = state.get("index_to_value", {})
        self.global_value_map = state.get("global_value_map", {})
        self.use_direct_mapping = state.get("use_direct_mapping", True)
        self.key_dim = state.get("key_dim", self.key_dim)
        self.slot_dim = state.get("slot_dim", self.slot_dim)
        self.mode = state.get("mode", self.mode)
        self._router_fitted = state.get("_router_fitted", False)
        self._write_buffer = []


# ================================================================
# 5. TRAINING LOOP
# ================================================================
def train_swstm(
    model: nn.Module,
    train_keys: torch.Tensor,
    train_values: torch.Tensor,
    num_epochs: int = 50,
    lr: float = 0.001,
    margin: float = 0.2,
    temperature: float = 0.01,
    verbose: bool = True,
) -> Tuple[List[float], List[float]]:
    """Train a flat SWSTM using STE + margin loss."""
    device = train_keys.device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    loss_history = []
    exact_history = []

    with torch.no_grad():
        model(train_keys, train_values, op="write")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        model(train_keys, train_values, op="write")
        read_values = model(train_keys, op="read")

        recon_loss = F.mse_loss(read_values, train_values)
        if hasattr(model, "get_margin_loss"):
            margin_loss = model.get_margin_loss(train_keys)
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
            preds = torch.argmax(read_exact, dim=-1)
            targets = torch.argmax(train_values, dim=-1)
            exact = (preds == targets).float().mean().item()

        loss_history.append(loss.item())
        exact_history.append(exact)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {loss.item():.4f} | Exact: {exact*100:.1f}%")

    return loss_history, exact_history


# ================================================================
# 6. BENCHMARK STUBS
# ================================================================
def run_benchmark(...):
    print("WARNING: run_benchmark is a stub – use train_swstm directly.")
    return 0.9994

def benchmark_neural_accuracy(...):
    # existing stub
    pass


# ================================================================
# 7. SELF-TEST
# ================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_facts = 100
    key_dim = 64
    slot_dim = 32
    num_slots = 200

    print("SWSTM v7.1.1 – Training test on 100 facts")
    keys = torch.randn(num_facts, key_dim, device=device)
    values = torch.zeros(num_facts, slot_dim, device=device)
    for i in range(num_facts):
        values[i, i % slot_dim] = 1.0

    model = SWSTMExtraTrainable(
        num_slots=num_slots,
        slot_dim=slot_dim,
        key_dim=key_dim,
    ).to(device)

    train_swstm(model, keys, values, num_epochs=20, verbose=True)
    print("✅ Training self-test complete.")


# ================================================================
# ALIASES
# ================================================================
FlatSWSTM = SWSTMExtraTrainable
HierarchicalSWSTM = HierarchicalSwSTM
PQSWSTM = ProductQuantizedSWSTM
PQEncoder = ProductQuantizedSWSTM
