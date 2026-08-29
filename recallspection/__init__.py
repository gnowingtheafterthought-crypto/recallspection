"""
Recallspection – Dual‑core exact memory for AI agents.

ExactMemory (cryptographic hash table) and SWSTM (neural memory) in one package.
"""

# Version
__version__ = "18.0.0"

# ExactMemory core
from .exact import ExactMemory

# SWSTM core (all backends + engine)
from .swstm import (
    FlatSWSTM,
    HierarchicalSWSTM,
    PQEncoder,
    PQSWSTM,
    SWSTMEngine,
)

# Public API
__all__ = [
    # ExactMemory
    "ExactMemory",
    # SWSTM
    "FlatSWSTM",
    "HierarchicalSWSTM",
    "PQEncoder",
    "PQSWSTM",
    "SWSTMEngine",
]

# Convenience: if someone imports from recallspection directly,
# they get the most commonly used classes.
