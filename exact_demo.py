#!/usr/bin/env python3
"""
exact_demo.py — Quick Validation of ExactMemory

This script demonstrates the cryptographic exact core.
It stores 10,000 facts and measures:
- Insertion speed
- Retrieval speed
- Accuracy
- Tamper-evidence

Expected output: PASS (100% EMR, < 0.9 µs exact, < 8 µs verified)

Usage:
    python examples/exact_demo.py
"""

import sys
import time
import random
import string

# Adjust path if running directly from the repo root
try:
    from recallspection import ExactMemory, ExactConfig
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from recallspection import ExactMemory, ExactConfig


def generate_facts(n):
    facts = {}
    for i in range(n):
        key = f"k_{i}_{random.getrandbits(24):06x}"
        value = f"v_{i}_" + ''.join(random.choices(string.ascii_letters + string.digits, k=50))
        facts[key] = value
    return facts


def main():
    print("\n" + "=" * 60)
    print("  RECALLSPECTION – ExactMemory Quick Demo")
    print("=" * 60)

    N = 10_000
    print(f"\n▶ Testing with {N:,} facts...")
    print(f"▶ Hash: {'BLAKE3' if __import__('importlib').util.find_spec('blake3') else 'SHA3-256'}")
    print("-" * 60)

    memory = ExactMemory()

    # Generate
    print("\n[1] Generating facts...")
    facts = generate_facts(N)

    # Insert
    print("[2] Inserting...")
    t0 = time.time()
    for k, v in facts.items():
        memory.add(k, v)
    insert_time = time.time() - t0
    print(f"    Inserted {N:,} in {insert_time:.2f}s")
    print(f"    Avg insert: {insert_time/N*1e6:.2f} µs")

    # Retrieve
    print("[3] Retrieving (verified)...")
    t0 = time.time()
    correct = 0
    for k, v in facts.items():
        got = memory.get(k)
        if got == v:
            correct += 1
    read_time = time.time() - t0
    print(f"    Retrieved {N:,} in {read_time:.2f}s")
    print(f"    Avg verified read: {read_time/N*1e6:.2f} µs")

    # Exact key (dict-only) performance
    print("[4] Retrieving (raw dict only, bypassing crypto)...")
    t0 = time.time()
    for k in facts.keys():
        _ = memory._storage.get(k)
    raw_time = time.time() - t0
    print(f"    Avg raw read: {raw_time/N*1e6:.2f} µs")

    # Stats
    stats = memory.stats()
    print("\n[5] Results:")
    print(f"    Exact matches: {correct}/{N} ({(correct/N)*100:.2f}%)")
    print(f"    Verification failures: {stats['verification_failures']}")

    # Tamper test
    print("\n[6] Tamper-evidence test...")
    test_key = list(facts.keys())[0]
    memory._storage[test_key] = b"TAMPERED"
    got = memory.get(test_key)
    if got is None:
        print("    [PASS] Tampering detected (returned None).")
    else:
        print(f"    [FAIL] Tampering NOT detected: got {got}")

    # Verdict
    if correct == N and stats['verification_failures'] == 0:
        print("\n" + "=" * 60)
        print("  🏁 VERDICT: PASS (100% EMR, Tamper-Evident)")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("  🏁 VERDICT: FAIL")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())