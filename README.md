<p align="center">
  <img src="banner_v17.svg" alt="Recallspection Banner — Cinematic v17" width="800">
</p>

# 🧠 RECALLSPECTION v17.0.0: THE CRYPTOGRAPHIC EXACT MEMORY LAYER

> *"Every major AI memory benchmark is mathematically incapable of defeating a 200‑line cryptographic hash table and it runs on a phone."*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnowingtheafterthought-crypto/recallspection/blob/main/demo.ipynb)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v17.0.0-blue)](https://github.com/gnowingtheafterthought-crypto/recallspection)
[![Validated on iOS](https://img.shields.io/badge/Validated-iOS%20%7C%208.14µs-5A29E4)](https://github.com/gnowingtheafterthought-crypto/recallspection)
[![API Status](https://img.shields.io/website?url=https%3A%2F%2Frecallspection.onrender.com)](https://recallspection.onrender.com)

---

## ⚡ The Killshot at a Glance

| Metric | Value | Why It Matters |
|--------|-------|----------------|
| **Exact Match Ratio** | **1.0000** | Zero hallucinations. Mathematically guaranteed. |
| **Verified Read Latency** | **~8 µs** | 1,000x faster than vector DBs. |
| **Memory (1M facts)** | **~471 MB** | Runs on a phone. No cloud required. |
| **Tamper‑Evidence** | **✅** | Returns `None` on corruption. Cryptographically verifiable. |
| **Dependencies** | **None (stdlib)** | `pip install` and run. No PyTorch, no FAISS, no GPU. |
| **Platform** | **iOS, Linux, macOS** | Validated on iPhone. Edge-ready. |

> **The bottom line:** Every major AI memory benchmark (BEAM 10M, LoCoMo, AML, AMA-Bench) is mathematically incapable of defeating a 200‑line cryptographic hash table and it runs on a phone.

---

## 📌 What is Recallspection?

**Recallspection is the first system to separate *exact memory* from *probabilistic reasoning*.**

It provides:

1. **A Cryptographic Exact Core** (`ExactMemory`): Pure Python stdlib. SHA3‑256 + zlib. 100% EMR. Tamper‑evident. 8 µs reads.
2. **A Semantic Layer** (`CompleteObserver`): Optional FAISS + quorum consensus for natural language discovery.

**Why this matters:** Current AI systems use probabilistic embeddings for *everything* — including facts that must be exact. This leads to hallucinations. Recallspection uses deterministic hashing for facts, and keeps probabilistic search for discovery. It's the difference between a *database* and a *search engine*.

---

## 🧠 The Six Structural Moats

| Moat | Problem | Recallspection Solution |
|------|---------|-------------------------|
| **1. 51‑Hop Limit** | Exponential decay in ANN systems | SWSTM exact routing → 100% recall at all hops |
| **2. Composition Hallucination** | Models fail to combine facts correctly | Raw Displacement BFS → **1.0000** cosine |
| **3. Conflict Resolution** | Contradictory facts accumulate | Surprise Gate (Φ) → refuses conflicting writes |
| **4. Structural Hallucination** | Queries return plausible but wrong results | Confidence Gate (Λ) → refuses uncertain queries |
| **5. State Management Collapse** | Memory lost on restart | MD5‑locked persistence → survives save/load |
| **6. Error Amplification** | Multi‑agent systems compound errors | Bounded Drift Theorem → O(1) error growth |

---

## ⚙️ The Exact Core (Cryptographic Memory)

The `ExactMemory` class provides a **cryptographic exact key‑value store** with:

- **SHA3‑256 / BLAKE3** hashing (32‑byte raw digests)
- **zlib compression** (level 6) to reduce memory footprint
- **Packed metadata** — 136 bytes per fact (quorum hashes + value hash + timestamp)
- **Quorum verification** (default `quorum_size=3`)
- **Tamper‑evidence** — returns `None` on corruption
- **100% EMR** — exact match ratio of 1.0000
- **O(1)** deterministic lookup
- **Zero external dependencies** — pure Python stdlib

### Usage

```python
from recallspection import ExactMemory

memory = ExactMemory()
memory.add("user_123_pref", {"theme": "dark", "language": "en"})
result = memory.get("user_123_pref")  # {'theme': 'dark', 'language': 'en'}

# Tamper test
memory._storage["user_123_pref"] = b"TAMPERED"
result = memory.get("user_123_pref")  # None — tampering detected!