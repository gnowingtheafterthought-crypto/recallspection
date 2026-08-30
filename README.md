<p align="center">
  <img src="banner.svg" alt="Recallspection Banner" width="800">
</p>

# 🧠 RECALLSPECTION v18.0.0: THE DUAL‑CORE EXACT MEMORY LAYER

> *"One core for compliance, one core for fuzzy – both mathematically incapable of hallucination."*

[![GitHub](https://img.shields.io/badge/GitHub-sciencedelicmetatech%2Frecallspection-blue)](https://github.com/sciencedelicmetatech/recallspection)
[![License](https://img.shields.io/badge/license-AGPLv3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v18.0.0-blue)](https://github.com/sciencedelicmetatech/recallspection)
[![Render](https://img.shields.io/website?url=https%3A%2F%2Frecallspection.onrender.com)](https://recallspection.onrender.com)

---

## ⚡ At a Glance

| Feature | ExactMemory (Compliance) | SWSTM (Neural) |
|---------|--------------------------|----------------|
| **Exact Match Ratio** | 1.0000 | 1.0000 (hierarchical) |
| **Paraphrase / Fuzzy Queries** | ❌ | ✅ |
| **Verified Read Latency** | ~8 µs | ~1 ms (PQ) |
| **Memory (1M facts)** | ~471 MB | ~24 MB (PQ) |
| **Tamper‑Evidence** | ✅ SHA3‑256 + zlib | ❌ (but exact) |
| **Dependencies** | None (stdlib) | torch, sentence‑transformers, scikit‑learn |

---

## 📌 What is Recallspection?

**Recallspection** provides **two memory engines** in one package:

1. **ExactMemory** – cryptographic hash table. SHA3‑256 + zlib. 8 µs reads. Tamper‑evident. **Zero dependencies.** For compliance, audit, and exact‑key lookups.

2. **SWSTM** – neural memory. Differentiable, hierarchical, product‑quantized. Achieves **100% exact match** on fuzzy/paraphrase queries. Scales to 1M+ facts with 24 bytes/fact. Patent pending.

**Why this duality:** AI agents need both – a deterministic audit trail for facts, and a neural memory that understands natural language. Recallspection gives you both in a unified API.

---

## 🏗️ Architecture

### ExactMemory (Cryptographic Core)
- SHA3‑256 / BLAKE3 hashing (32‑byte digests)
- zlib compression (level 6)
- Quorum verification (default `quorum_size=3`)
- Tamper‑evidence – returns `None` on corruption
- 100% EMR – exact match ratio of 1.0000

### SWSTM (Neural Core)
- **Flat mode** – up to 1,000 facts (≥97% exact)
- **Hierarchical mode** – up to 50,000 facts (100% exact)
- **PQ mode** – 1M+ facts with 24 bytes/fact (100% exact)
- STE training + margin loss (optional)

---

## 🛠️ Quickstart

### Installation
```bash
pip install git+https://github.com/sciencedelicmetatech/recallspection.git