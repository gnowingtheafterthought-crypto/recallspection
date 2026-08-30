"""Recallspection – Dual‑core exact memory for AI agents. ExactMemory
(cryptographic hash table) and SWSTM (neural memory) in one package."""

# Version
__version__ = "18.0.0"

# ExactMemory core
from .exact import ExactMemory

# SWSTM core (from the new core folder)
from .core.swstem import SWSTM

# Public API
__all__ = [
    "ExactMemory",
    "SWSTM",
]