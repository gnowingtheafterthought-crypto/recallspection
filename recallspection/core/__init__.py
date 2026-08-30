"""Recallspection – Dual‑core exact memory for AI agents. ExactMemory 
(cryptographic hash table) and SWSTM (neural memory) in one package."""

# Version
__version__ = "18.0.0"

# ExactMemory core
from .exact import ExactMemory

# SWSTM core (only SWSTM class exists)
from .core.swstem import SWSTM

# Alignment Sentinel
from .core.alignment import AlignmentSentinel

# Public API
__all__ = [
    "ExactMemory",
    "SWSTM",
    "AlignmentSentinel",
]
