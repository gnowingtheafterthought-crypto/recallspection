import numpy as np

class SurpriseGate:
    """Dynamic thresholding (Phi) for memory storage prioritization."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(self, similarity: float) -> bool:
        # Invert similarity to get novelty/surprise
        surprise = 1.0 - similarity
        return surprise > self.threshold
