"""
recallspection – Neural Exact Memory (SWSTM v7.0.3)
"""

from .swstm import (
    # Core neural memory classes
    SWSTMExtraTrainable,
    HierarchicalSwSTM,
    ProductQuantizedSWSTM,
    # High‑level engine for easy string‑based usage
    SWSTMEngine,
    # Training and benchmarking
    train_swstm,
    run_benchmark,
    benchmark_neural_accuracy,
    # Backward‑compatibility aliases
    FlatSWSTM,
    HierarchicalSWSTM,
    PQSWSTM,
    PQEncoder,
)

__version__ = "7.0.4"
__all__ = [
    "SWSTMExtraTrainable",
    "HierarchicalSwSTM",
    "ProductQuantizedSWSTM",
    "SWSTMEngine",
    "train_swstm",
    "run_benchmark",
    "benchmark_neural_accuracy",
    "FlatSWSTM",
    "HierarchicalSWSTM",
    "PQSWSTM",
    "PQEncoder",
]
