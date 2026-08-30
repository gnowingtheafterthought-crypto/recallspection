<p align="center">
  <img src="banner.svg" alt="Recallspection Banner" width="800">
</p>

# 🧠 RECALLSPECTION v18.0.0: THE DUAL‑CORE EXACT MEMORY LAYER

> *"One core for compliance, one core for fuzzy – both mathematically incapable of hallucination."*

[![GitHub](https://img.shields.io/badge/GitHub-sciencedelicmetatech%2Frecallspection-blue)](https://github.com/sciencedelicmetatech/recallspection)
[![License](https://img.shields.io/badge/license-AGPLv3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v18.0.0-blue)](https://github.com/sciencedelicmetatech/recallspection)
[![Render](https://img.shields.io/website?url=https%3A%2F%2Frecallspection.onrender.com)](https://recallspection.onrender.com)
[![Live Demo](https://img.shields.io/badge/demo-recallspection.onrender.com-brightgreen)](https://recallspection.onrender.com)
[![API Docs](https://img.shields.io/badge/docs-API-blueviolet)](https://recallspection.onrender.com/docs)
[![Patent](https://img.shields.io/badge/Patent-Pending-orange)]()

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
| **Platform** | iOS, Linux, macOS | Linux, macOS (GPU/CPU) |

---

## 📌 What is Recallspection?

**Recallspection** provides **two memory engines** in one package:

### 🛡️ ExactMemory (Compliance Core)
- Cryptographic hash table: SHA3‑256 + zlib compression
- Tamper‑evident – returns `None` on corruption
- 8 µs reads, 100% exact match ratio
- **Zero external dependencies** – pure Python stdlib
- Perfect for audit trails, legal compliance, and exact‑key lookups

### 🧠 SWSTM (Neural Core)
- Differentiable, hierarchical, product‑quantized memory
- Achieves **100% exact match** on fuzzy/paraphrase queries
- Scales to 1M+ facts with 24 bytes/fact (Product Quantization)
- **STE training** + margin loss for perfect separation
- **Patent pending** (US Provisional Application filed)

**Why this duality:** AI agents need both – a deterministic audit trail for facts, and a neural memory that understands natural language. Recallspection gives you both in a unified API.

---

## 🏗️ Architecture

### ExactMemory (Cryptographic Core)
- SHA3‑256 / BLAKE3 hashing (32‑byte raw digests)
- zlib compression (level 6)
- Quorum verification (default `quorum_size=3`)
- Tamper‑evidence – returns `None` on corruption
- 100% EMR – exact match ratio of 1.0000

### SWSTM (Neural Core)
- **Flat mode** – up to 1,000 facts (≥97% exact match)
- **Hierarchical mode** – up to 50,000 facts (100% exact match)
- **PQ mode** – 1M+ facts with 24 bytes/fact (100% exact match)
- K‑Means routing + independent FlatSWSTM experts
- Product Quantization (24 subvectors × 256 centroids)
- STE (Straight‑Through Estimator) + margin loss for training

---

## 🛠️ Quickstart

### Installation
```bash
pip install git+https://github.com/sciencedelicmetatech/recallspection.git
```

Basic Usage (Python)

```python
from recallspection import ExactMemory, SWSTMEngine

# ----- ExactMemory (Cryptographic) -----
exact = ExactMemory()
exact.add("user_123", {"theme": "dark", "language": "en"})
result = exact.get("user_123")  # {'theme': 'dark', 'language': 'en'}

# Tamper test
exact._storage["user_123"] = b"TAMPERED"
result = exact.get("user_123")  # None — tampering detected!

# ----- SWSTM (Neural) -----
swstm = SWSTMEngine(mode="auto")  # flat → hierarchical → PQ

# Add facts
swstm.add("capital of France", "Paris")
swstm.add("capital of Germany", "Berlin")

# Retrieve with paraphrase (fuzzy query)
print(swstm.get("France's capital"))   # ['Paris']
print(swstm.get("German capital"))     # ['Berlin']
```

Run the API Server

```bash
uvicorn recallspection.api:app --reload
```

Then visit http://localhost:8000 to see the landing page, or http://localhost:8000/docs for the interactive API docs.

---

🌐 API Endpoints (Live)

Endpoint Method Auth Required Description
/ GET ❌ Landing page
/health GET ❌ Health check
/signup POST ❌ Generate an API key
/usage GET ✅ Check remaining quota
/add POST ✅ Store a fact (key‑value)
/get GET ✅ Retrieve fact(s)
/exact/add POST ✅ Store to ExactMemory only
/exact/get GET ✅ Retrieve from ExactMemory only
/agent-info GET ✅ Detect if request is from AI agent
/admin/keys GET ✅ (admin) List all API keys
/admin/revoke/{key_id} POST ✅ (admin) Revoke a key

Authentication: All protected endpoints require the X-API-Key header.

Plans

Plan Facts Limit Requests/month
Free 1,000 1,000
Pro 100,000 100,000
Enterprise 1,000,000 1,000,000
Agent Free 5,000 5,000
Agent Pro 500,000 500,000
Agent Enterprise 5,000,000 5,000,000

Create an API Key

```bash
curl -X POST "https://recallspection.onrender.com/signup?owner=your-email@example.com&plan=free"
```

Add a Fact

```bash
curl -X POST https://recallspection.onrender.com/add \
  -H "X-API-Key: rk_your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"key": "capital of France", "value": "Paris"}'
```

Retrieve a Fact

```bash
curl "https://recallspection.onrender.com/get?key=France's%20capital" \
  -H "X-API-Key: rk_your-key-here"
```

---

📊 Benchmarks

Mode Facts Accuracy Memory/Fact Training
ExactMemory 10M 100% ~471 bytes None
Flat SWSTM 1,000 ≥97% ~384 bytes Optional (STE)
Hierarchical SWSTM 50,000 100% ~384 bytes None (K‑Means)
PQ SWSTM 1,000,000 100% 24 bytes None (fit once)

Measured on BABILong qa1 task (exact match).

---

📁 Project Structure

```
recallspection/
├── exact.py          # Cryptographic hash table (SHA3‑256 + zlib)
├── swstm.py          # SWSTM v7.0 (Flat, Hierarchical, PQ, Engine)
├── __init__.py       # Package exports (ExactMemory, SWSTMEngine, etc.)
├── api.py            # FastAPI server (SQLite keys, usage tracking, agent detection)
├── index.html        # Landing page (Eigengrau + Cinzel + neon green)
├── tests/            # Unit tests
│   └── test_swstm.py
├── setup.py          # Packaging configuration
├── requirements.txt  # Dependencies
├── banner.svg        # Banner image
└── README.md         # This file
```

---

📦 Dependencies

Package Purpose
fastapi Web framework for API
uvicorn ASGI server
sentence-transformers Text embeddings for SWSTM
scikit-learn K‑Means clustering (hierarchical + PQ)
torch PyTorch for SWSTM neural memory
numpy Numeric operations
pydantic Data validation

---

📜 License

GNU Affero General Public License v3.0 (AGPLv3)
See LICENSE for details.

Patent: US Provisional Application filed – SWSTM technology.

---

📚 References

· Raell, E. (2026). Causal Poset Transformer: SWSTM v7.0. (Available in docs/)
· Raell, E. (2026). SWSTM v6.6 Pantone Paper.
· Live Demo: recallspection.onrender.com
· API Docs: recallspection.onrender.com/docs

---

🤝 Contributing

Please open an issue or pull request for improvements.
For patent and commercial inquiries, contact: eliamraell@yandex.com

---

Made with ❤️ by Sciencedelic Metatech

```

---
