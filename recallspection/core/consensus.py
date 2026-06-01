import numpy as np
from typing import List, Dict

class ConsensusEngine:
    def __init__(self, threshold=0.9, min_agents=3):
        self.threshold = threshold
        self.min_agents = min_agents

    def reach_consensus(self, recalls: List[Dict]) -> Dict:
        if len(recalls) < self.min_agents:
            return {"status": "REJECTED", "reason": "Insufficient Peer Data"}

        votes = {}
        for r in recalls:
            label = r['label']
            weight = r['similarity']
            votes[label] = votes.get(label, 0) + weight

        winner = max(votes, key=votes.get)
        total_weight = sum(votes.values())
        confidence = votes[winner] / total_weight

        if confidence > self.threshold:
            return {
                "status": "ACCEPTED",
                "label": winner,
                "global_confidence": float(confidence)
            }
        return {"status": "DISPUTED", "reason": "Low Network Consensus"}
