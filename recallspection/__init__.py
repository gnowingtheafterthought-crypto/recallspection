"""
Recallspection — Exact + Semantic Memory for AI Agents

This package provides:
- `ExactMemory`: A cryptographic, stdlib-only, exact key-value store.
- `CompleteObserver`: A semantic memory layer (FAISS + quorum consensus).
"""

from .observer import CompleteObserver
from .exact import ExactMemory, ExactConfig

__version__ = "17.0.0"
__author__ = "Sciencedelic Metatech"

__all__ = [
    "CompleteObserver",
    "ExactMemory",
    "ExactConfig",
]
