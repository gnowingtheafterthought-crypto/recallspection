<p align="center">
  <img src="banner.svg" alt="Recallspection Banner" width="800">
</p>

# RECALLSPECTION v18.0.0 – VERIFIABLE AI MEMORY
[![CI](https://github.com/sciencedelicmetatech/recallspection/actions/workflows/ci.yml/badge.svg)](https://github.com/sciencedelicmetatech/recallspection/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
> *"We came to kill hallucination, but we found a better purpose: proving exactly what we said, to whom, and when."*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sciencedelicmetatech/recallspection/blob/main/demo.ipynb)
[![License](https://img.shields.io/badge/license-AGPLv3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v18.0.0-blue)](https://github.com/sciencedelicmetatech/recallspection)
[![Validated on iOS](https://img.shields.io/badge/Validated-iOS%20%7C%200.4ms-5A29E4)](https://github.com/sciencedelicmetatech/recallspection)
[![API Status](https://img.shields.io/website?url=https%3A%2F%2Frecallspection.onrender.com)](https://recallspection.onrender.com)

---

## The Honest Glimpse

Recallspection is a **cryptographic, verifiable memory layer** for AI systems that need to **prove exactly what they retrieved, to whom, and when**.

It is **not** a general‑purpose AI memory, nor does it solve hallucination.  
It **is** a tool for **RegTech, Compliance, and Legal AI** where regulators demand proof that the AI quoted the exact source text, unaltered.

### What it does (and doesn't do)

| ✅ What it does | ❌ What it doesn't do |
| :--- | :--- |
| Cryptographically verify every retrieved fact (SHA3‑256 + timestamp). | Solve hallucination (that's a *generation* problem). |
| Return the exact value for structured keys (`fact_3050` → `value_3050`) in **0.4 ms**. | Perform deep reasoning or logic (it's a memory store, not a brain). |
| Handle paraphrases via a semantic fallback (FAISS + Sentence Transformers). | Achieve "24 bytes per fact" (Product Quantization is not yet implemented in the main branch). |
| Run on edge devices (pure Python, zero external dependencies for the core). | Scale to millions of facts on CPU (use GPU for large‑scale semantic indexing). |

---

## 🧭 Architecture: Dual‑Core with Exact‑First Routing

The system combines two memory engines with a strict routing logic:

1. **ExactCore** – A cryptographic hash table (SHA3‑256 + zlib + dict).  
   - **100% deterministic** – exact key → exact value.  
   - **Tamper‑evident** – any corruption returns `None`.  
   - **Latency**: ~0.4 ms on GPU, ~0.5 ms on CPU.

2. **SemanticCore** – A standard RAG pipeline (Sentence Transformers + FAISS).  
   - Handles paraphrases, typos, and natural‑language queries.  
   - **Latency**: ~33 ms on GPU, ~60 ms on CPU.

3. **Exact‑First Routing** – Before any semantic search, the system extracts structured keys (e.g., `fact_3050`, `doc_123`) and performs an exact lookup.  
   - This fixes the **`3050` vs `305`** problem – a known failure mode of pure semantic search.

---

## 📊 Performance Benchmarks (Empirical, GPU)

Run on Google Colab (T4 GPU, 10,000 facts):

| Operation | Latency | Accuracy |
| :--- | :--- | :--- |
| Write (10,000 facts) | **3.48 s** | 100% |
| Exact Lookup (`fact_3050`) | **0.41 ms** | 100% |
| Semantic Lookup (paraphrase) | **32.96 ms** | 100% (on test) |
| Semantic Lookup without routing (`fact 3050`) | 0.13 ms* | **Fails** (returns `value_305`) |

* *Routed to ExactCore via key extraction – bypasses semantic layer entirely.*

**Reproducible**: Run the [Colab notebook](https://colab.research.google.com/github/sciencedelicmetatech/recallspection/blob/main/demo.ipynb) to verify.

---

## 🚀 Installation

```bash
pip install recallspection result = memory.get("user_123_pref")  # None — tampering detected!
