"""
Recallspection v17: Complete Observer
=====================================
Deterministic, self-healing, quorum-verified cognitive infrastructure.

"Death to approximation. Long live the Complete Observer."
"""

import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class CompleteObserver:
    """
    The Complete Observer: A deterministic memory system with:
    - Semantic Geometry: Entities as normalized vectors, relationships as displacement vectors
    - O(1) Deterministic Routing: SHA3-256 hashing with ephemeral salts
    - Quorum Verification: Majority consensus (k=4, quorum=3)
    - Autonomous Self-Healing: Background cycles detect and fix geometric inconsistencies
    """
    
    def __init__(self, dimension: int = 128, k: int = 4, quorum: int = 3, 
                 healing_threshold: float = 1e-5, salt: str = "recallspection_v17"):
        """
        Initialize the Complete Observer.
        
        Args:
            dimension: Dimensionality of the semantic space (R^d)
            k: Number of overlapping slots for redundancy
            quorum: Minimum number of matching slots for retrieval
            healing_threshold: Norm error threshold triggering self-healing
            salt: Base salt for deterministic hashing
        """
        self.dimension = dimension
        self.k = k
        self.quorum = quorum
        self.healing_threshold = healing_threshold
        self.base_salt = salt
        
        # Node storage: entity_id -> normalized vector
        self.nodes: Dict[str, np.ndarray] = {}
        
        # Fact storage: hash_slot -> list of (source, target, displacement, metadata)
        self.fact_slots: Dict[int, List[Tuple[str, str, np.ndarray, Dict]]] = defaultdict(list)
        
        # Displacement cache for O(1) routing
        self.displacements: Dict[Tuple[str, str], np.ndarray] = {}
        
        # Corruption tracking
        self.corruption_log: List[Dict] = []
        self.healing_count = 0
        
    def _hash_to_slot(self, key: str, salt: str = "") -> int:
        """
        Deterministic routing via SHA3-256 hashing with ephemeral salts.
        Maps a key to one of k overlapping slots.
        """
        combined_salt = f"{self.base_salt}:{salt}" if salt else self.base_salt
        hash_input = f"{key}:{combined_salt}".encode('utf-8')
        hash_digest = hashlib.sha3_256(hash_input).hexdigest()
        return int(hash_digest[:8], 16) % self.k
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit norm in R^d."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return vector
        return vector / norm
    
    def add_node(self, entity_id: str, embedding: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Add or update a node (entity) in the semantic space.
        Entities are normalized vectors in R^d.
        
        Args:
            entity_id: Unique identifier for the entity
            embedding: Optional pre-computed embedding; if None, generates deterministic embedding
            
        Returns:
            The normalized vector for this entity
        """
        if embedding is None:
            # Generate deterministic embedding from entity_id
            seed = int(hashlib.sha3_256(entity_id.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            embedding = rng.standard_normal(self.dimension)
        
        normalized = self._normalize(embedding)
        self.nodes[entity_id] = normalized
        return normalized
    
    def add_fact(self, source: str, target: str, relation: str = "related", 
                 metadata: Optional[Dict] = None) -> bool:
        """
        Add a fact (relationship) between two entities.
        Relationships are displacement vectors computed from source to target.
        
        Args:
            source: Source entity ID
            target: Target entity ID
            relation: Type of relationship
            metadata: Optional metadata dictionary
            
        Returns:
            True if fact was added successfully
        """
        # Ensure nodes exist
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
        
        # Compute displacement vector (telescopic composition)
        source_vec = self.nodes[source]
        target_vec = self.nodes[target]
        displacement = target_vec - source_vec
        
        # Store displacement for O(1) retrieval
        self.displacements[(source, target)] = displacement
        
        # Route to k overlapping slots using different salts
        for i in range(self.k):
            slot_key = f"{source}:{target}:{relation}"
            slot = self._hash_to_slot(slot_key, salt=f"slot_{i}")
            self.fact_slots[slot].append((source, target, displacement, {
                'relation': relation,
                'metadata': metadata or {},
                'timestamp': len(self.corruption_log) + self.healing_count
            }))
        
        return True
    
    def get(self, source: str, target: str) -> Optional[Dict]:
        """
        Retrieve a fact via quorum verification.
        Requires majority consensus to prevent single-point corruption.
        
        Args:
            source: Source entity ID
            target: Target entity ID
            
        Returns:
            Dictionary with retrieval result or None if quorum not reached
        """
        slot_key = f"{source}:{target}:related"
        
        # Collect votes from all k slots
        votes = []
        for i in range(self.k):
            slot = self._hash_to_slot(slot_key, salt=f"slot_{i}")
            if slot in self.fact_slots:
                for stored_source, stored_target, displacement, meta in self.fact_slots[slot]:
                    if stored_source == source and stored_target == target:
                        votes.append({
                            'displacement': displacement,
                            'metadata': meta
                        })
        
        # Quorum verification
        if len(votes) < self.quorum:
            return {
                'status': 'QUORUM_FAILED',
                'votes_received': len(votes),
                'quorum_required': self.quorum
            }
        
        # Verify consistency among votes
        if len(votes) > 0:
            reference_disp = votes[0]['displacement']
            consistent_votes = 1
            
            for vote in votes[1:]:
                diff_norm = np.linalg.norm(vote['displacement'] - reference_disp)
                if diff_norm < self.healing_threshold:
                    consistent_votes += 1
            
            if consistent_votes >= self.quorum:
                return {
                    'status': 'VERIFIED',
                    'displacement': reference_disp,
                    'confidence': consistent_votes / len(votes),
                    'metadata': votes[0]['metadata']
                }
            else:
                # Potential corruption detected
                self._check_corruption(source, target)
                return {
                    'status': 'CORRUPTION_DETECTED',
                    'consistent_votes': consistent_votes,
                    'quorum_required': self.quorum
                }
        
        return None
    
    def compose_path(self, path: List[str]) -> Dict:
        """
        Compose a multi-hop path using telescopic displacement addition.
        Path composition is telescopic with zero algorithmic drift.
        
        Args:
            path: List of entity IDs representing the path
            
        Returns:
            Dictionary with composed displacement and verification status
        """
        if len(path) < 2:
            return {'status': 'INVALID_PATH', 'reason': 'Path must have at least 2 nodes'}
        
        total_displacement = np.zeros(self.dimension)
        hops_verified = 0
        hop_results = []
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            
            # Try to get direct displacement
            if (source, target) in self.displacements:
                displacement = self.displacements[(source, target)]
            else:
                # Try retrieval
                result = self.get(source, target)
                if result and result.get('status') == 'VERIFIED':
                    displacement = result['displacement']
                elif (source in self.nodes and target in self.nodes):
                    # Compute on the fly
                    displacement = self.nodes[target] - self.nodes[source]
                else:
                    hop_results.append({
                        'hop': i,
                        'source': source,
                        'target': target,
                        'status': 'MISSING'
                    })
                    continue
            
            total_displacement += displacement
            hops_verified += 1
            hop_results.append({
                'hop': i,
                'source': source,
                'target': target,
                'status': 'VERIFIED'
            })
        
        # Verify telescopic property: total_disp should equal end - start
        if path[0] in self.nodes and path[-1] in self.nodes:
            expected_displacement = self.nodes[path[-1]] - self.nodes[path[0]]
            drift = np.linalg.norm(total_displacement - expected_displacement)
            
            return {
                'status': 'COMPOSED',
                'total_displacement': total_displacement,
                'expected_displacement': expected_displacement,
                'drift': drift,
                'hops_verified': hops_verified,
                'total_hops': len(path) - 1,
                'hop_details': hop_results,
                'exact_match_rate': 1.0 if drift < self.healing_threshold else 0.0
            }
        
        return {
            'status': 'COMPOSED',
            'total_displacement': total_displacement,
            'hops_verified': hops_verified,
            'hop_details': hop_results
        }
    
    def _check_corruption(self, source: str, target: str) -> List[Dict]:
        """
        Check for geometric inconsistencies in stored facts.
        Detects norm errors exceeding the healing threshold.
        
        Args:
            source: Source entity ID
            target: Target entity ID
            
        Returns:
            List of detected corruptions
        """
        corruptions = []
        slot_key = f"{source}:{target}:related"
        
        # Collect all displacements for this fact
        all_displacements = []
        for i in range(self.k):
            slot = self._hash_to_slot(slot_key, salt=f"slot_{i}")
            if slot in self.fact_slots:
                for stored_source, stored_target, displacement, meta in self.fact_slots[slot]:
                    if stored_source == source and stored_target == target:
                        all_displacements.append(displacement)
        
        if len(all_displacements) < 2:
            return corruptions
        
        # Check for inconsistencies
        reference = all_displacements[0]
        for i, disp in enumerate(all_displacements[1:], 1):
            norm_error = np.linalg.norm(disp - reference)
            if norm_error > self.healing_threshold:
                corruption_record = {
                    'type': 'DISPLACEMENT_MISMATCH',
                    'source': source,
                    'target': target,
                    'norm_error': float(norm_error),
                    'slot_index': i
                }
                corruptions.append(corruption_record)
                self.corruption_log.append(corruption_record)
        
        return corruptions
    
    def _heal(self, source: str, target: str) -> Dict:
        """
        Autonomously heal detected corruptions by recomputing displacements
        from source/target nodes.
        
        Args:
            source: Source entity ID
            target: Target entity ID
            
        Returns:
            Healing result dictionary
        """
        if source not in self.nodes or target not in self.nodes:
            return {'status': 'HEAL_FAILED', 'reason': 'Missing nodes'}
        
        # Recompute canonical displacement from nodes
        canonical_displacement = self.nodes[target] - self.nodes[source]
        
        # Update all slots with corrected displacement
        healed_slots = 0
        slot_key = f"{source}:{target}:related"
        
        for i in range(self.k):
            slot = self._hash_to_slot(slot_key, salt=f"slot_{i}")
            if slot in self.fact_slots:
                new_facts = []
                for stored_source, stored_target, displacement, meta in self.fact_slots[slot]:
                    if stored_source == source and stored_target == target:
                        # Replace with canonical displacement
                        new_facts.append((stored_source, stored_target, canonical_displacement, meta))
                        healed_slots += 1
                    else:
                        new_facts.append((stored_source, stored_target, displacement, meta))
                self.fact_slots[slot] = new_facts
        
        # Update displacement cache
        self.displacements[(source, target)] = canonical_displacement
        self.healing_count += 1
        
        return {
            'status': 'HEALED',
            'slots_corrected': healed_slots,
            'canonical_displacement': canonical_displacement,
            'healing_count': self.healing_count
        }
    
    def run_self_healing_cycle(self) -> Dict:
        """
        Run a background self-healing cycle to detect and fix all corruptions.
        
        Returns:
            Summary of the healing cycle
        """
        healed_facts = 0
        checked_facts = 0
        
        # Check all known displacements
        for (source, target) in list(self.displacements.keys()):
            checked_facts += 1
            corruptions = self._check_corruption(source, target)
            if corruptions:
                result = self._heal(source, target)
                if result['status'] == 'HEALED':
                    healed_facts += 1
        
        return {
            'status': 'CYCLE_COMPLETE',
            'facts_checked': checked_facts,
            'facts_healed': healed_facts,
            'total_healing_operations': self.healing_count
        }
    
    def get_stats(self) -> Dict:
        """Return statistics about the observer state."""
        return {
            'total_nodes': len(self.nodes),
            'total_facts': len(self.displacements),
            'total_slots': len(self.fact_slots),
            'corruptions_detected': len(self.corruption_log),
            'healing_operations': self.healing_count,
            'dimension': self.dimension,
            'k': self.k,
            'quorum': self.quorum
        }
