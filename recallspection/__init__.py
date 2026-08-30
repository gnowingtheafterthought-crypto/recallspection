"""
Recallspection – Dual‑core exact memory for AI agents.
"""

__version__ = "18.0.0"

from .exact import ExactMemory
from .swstm import (
    FlatSWSTM,
    HierarchicalSWSTM,
    PQEncoder,
    PQSWSTM,
    SWSTMEngine,
)

__all__ = [
    "ExactMemory",
    "FlatSWSTM",
    "HierarchicalSWSTM",
    "PQEncoder",
    "PQSWSTM",
    "SWSTMEngine",
]