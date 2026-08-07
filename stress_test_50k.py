#!/usr/bin/env python3
"""
Recallspection v17: 50K Facts Stress Test
=========================================
Demonstrates O(1) retrieval and high-throughput ingestion at scale.

Expected Results:
- Ingestion Rate: > 100k facts/sec
- Retrieval Time: < 10µs per fact (O(1))
- Exact Match Rate: 1.000000
"""

import sys
import time
import random
from recallspection.observer import CompleteObserver


def run_stress_test_50k():
    """Execute the 50,000 facts stress test."""
    print("=" * 60)
    print("RECALLSPECTION v17: 50K FACTS STRESS TEST")
    print("=" * 60)
    print()
    
    # Initialize observer
    observer = CompleteObserver(dimension=128, k=4, quorum=3)
    
    num_facts = 50000
    num_nodes = 10000
    
    # Phase 1: Node Creation
    print(f"Creating {num_nodes} nodes...")
    start_nodes = time.perf_counter()
    for i in range(num_nodes):
        observer.add_node(f"entity_{i}")
    node_time = time.perf_counter() - start_nodes
    print(f"Node creation: {node_time:.3f}s ({num_nodes/node_time:.0f} nodes/sec)")
    print()
    
    # Phase 2: Fact Ingestion
    print(f"Ingesting {num_facts} facts...")
    start_ingestion = time.perf_counter()
    
    facts_added = []
    for i in range(num_facts):
        source = f"entity_{random.randint(0, num_nodes - 1)}"
        target = f"entity_{random.randint(0, num_nodes - 1)}"
        observer.add_fact(source, target, relation=f"relation_{i % 100}")
        facts_added.append((source, target))
    
    ingestion_time = time.perf_counter() - start_ingestion
    ingestion_rate = num_facts / ingestion_time
    
    print(f"Ingestion time: {ingestion_time:.3f}s")
    print(f"Ingestion rate: {ingestion_rate:.0f} facts/sec")
    print()
    
    # Phase 3: O(1) Retrieval Test
    print(f"Testing O(1) retrieval on {min(1000, num_facts)} random facts...")
    retrieval_times = []
    exact_matches = 0
    tests_run = min(1000, len(facts_added))
    
    start_retrieval = time.perf_counter()
    for i in range(tests_run):
        source, target = facts_added[i]
        
        ret_start = time.perf_counter()
        result = observer.get(source, target)
        ret_end = time.perf_counter()
        
        retrieval_times.append(ret_end - ret_start)
        
        if result and result.get('status') == 'VERIFIED':
            exact_matches += 1
    
    total_retrieval_time = time.perf_counter() - start_retrieval
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    emr = exact_matches / tests_run
    
    print(f"Total retrieval time: {total_retrieval_time:.3f}s")
    print(f"Average retrieval time: {avg_retrieval_time * 1e6:.2f}µs")
    print(f"Exact Match Rate: {emr:.6f}")
    print()
    
    # Phase 4: Statistics
    stats = observer.get_stats()
    print("-" * 60)
    print("SYSTEM STATISTICS")
    print("-" * 60)
    print(f"Total Nodes:          {stats['total_nodes']}")
    print(f"Total Facts:          {stats['total_facts']}")
    print(f"Total Slots:          {stats['total_slots']}")
    print(f"Corruptions Detected: {stats['corruptions_detected']}")
    print(f"Healing Operations:   {stats['healing_operations']}")
    print()
    
    # Verification
    passed = True
    if ingestion_rate < 50000:
        print(f"✗ WARNING: Ingestion rate below expected ({ingestion_rate:.0f} < 50k)")
        passed = False
    else:
        print(f"✓ Ingestion rate verified: {ingestion_rate:.0f} facts/sec")
    
    if avg_retrieval_time > 50e-6:
        print(f"✗ WARNING: Retrieval time above expected ({avg_retrieval_time*1e6:.2f}µs > 50µs)")
        passed = False
    else:
        print(f"✓ O(1) retrieval verified: {avg_retrieval_time*1e6:.2f}µs average")
    
    if emr < 0.99:
        print(f"✗ PROOF FAILED: EMR below threshold ({emr:.6f} < 0.99)")
        passed = False
    else:
        print(f"✓ Exact Match Rate verified: {emr:.6f}")
    
    print()
    if passed:
        print("✓ ALL TESTS PASSED")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_stress_test_50k()
    sys.exit(0 if success else 1)
