"""
Recallspection – Dual‑core exact memory for AI agents.
"""

__version__ = "18.0.0"

from .exact import ExactMemory
from .swstm import SWSTMEngine, HybridEngine

__all__ = [
    "ExactMemory",
    "SWSTMEngine",
    "HybridEngine",
]
