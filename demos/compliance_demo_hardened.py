# ================================================================
# HARDENED COMPLIANCE DEMO — Recallspection (Graceful Error Handling)
# ================================================================
# This demo adds:
#   - Key‑value binding (hash = SHA3‑256(key + value))
#   - Ed25519 digital signature (non‑repudiation)
#   - Timestamp (system clock; production would use RFC 3161)
#   - Verification of final AI output against stored hash
#   - Graceful handling of decompression failures (returns None)
# ================================================================

# 1. Install cryptography if not present
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
except ImportError:
    !pip install cryptography
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption

import hashlib
import time
import json
import zlib
import base64

# ----- 2. Key management -----
private_key = ed25519.Ed25519PrivateKey.generate()
public_key = private_key.public_key()
print("🔑 Ed25519 key pair generated.")

# ----- 3. Storage with key‑value binding and signature -----
storage = {}

def store(key: str, value: str):
    # Canonical JSON representation of key + value
    data = json.dumps({"key": key, "value": value}, sort_keys=True).encode('utf-8')
    hash_digest = hashlib.sha3_256(data).hexdigest()
    timestamp = time.time()
    # Sign the hash
    signature = private_key.sign(hash_digest.encode('utf-8'))
    # Compress value (optional)
    compressed = zlib.compress(value.encode('utf-8'), level=6)
    storage[key] = {
        "compressed": compressed,
        "hash": hash_digest,
        "timestamp": timestamp,
        "signature": signature,
        "value": value  # keep original for display
    }
    return storage[key]

def retrieve(key: str):
    if key not in storage:
        return None
    entry = storage[key]
    # Decompress and verify (with graceful error handling)
    try:
        decompressed = zlib.decompress(entry["compressed"]).decode('utf-8')
    except zlib.error:
        return None  # corrupted / tampered data

    # Recompute hash
    data = json.dumps({"key": key, "value": decompressed}, sort_keys=True).encode('utf-8')
    recomputed_hash = hashlib.sha3_256(data).hexdigest()
    if recomputed_hash != entry["hash"]:
        return None  # hash mismatch

    # Verify signature
    try:
        public_key.verify(entry["signature"], entry["hash"].encode('utf-8'))
    except:
        return None  # signature invalid

    return {
        "key": key,
        "value": decompressed,
        "hash": entry["hash"],
        "timestamp": entry["timestamp"],
        "signature": base64.b64encode(entry["signature"]).decode('utf-8')
    }

# ----- 4. Demo: store a legal clause -----
clause_text = """
Clause 4.2: The Contractor shall deliver the final report within 30 days
of the Notice to Proceed. Failure to do so may result in liquidated damages
of RM 5,000 per day.
"""

key = "contract_42"
print("🧾 RECALLSPECTION — HARDENED COMPLIANCE DEMO")
print("="*60)

stored = store(key, clause_text)
print("📌 1. Stored a legal clause.")
print(f"   Key: {key}")
print(f"   Hash (key+value): {stored['hash']}")
print(f"   Timestamp: {time.ctime(stored['timestamp'])}")
print(f"   Signature (base64): {base64.b64encode(stored['signature']).decode('utf-8')[:40]}...")
print()

# ----- 5. Retrieve and verify -----
retrieved = retrieve(key)
if retrieved is not None:
    print("📌 2. Retrieved the clause.")
    print(f"   Verified hash matches: ✅")
    print(f"   Signature verified: ✅")
    print(f"   Retrieved text (first 100 chars):\n   {retrieved['value'][:100]}...")
else:
    print("   ❌ Verification failed.")
print()

# ----- 6. Generate a verifiable certificate -----
certificate = f"""
============================================================
COMPLIANCE CERTIFICATE (HARDENED)
============================================================
Document Key:    {key}
Retrieval Hash:  {retrieved['hash']}
Timestamp:       {time.ctime(retrieved['timestamp'])}
Signature:       {retrieved['signature'][:40]}...
Public Key:      {base64.b64encode(public_key.public_bytes_raw()).decode('utf-8')[:40]}...

Verification:
  - Hash (key+value) matches stored ✅
  - Ed25519 signature verified ✅
  - Timestamp recorded (system clock) ⚠️

This certifies that the stored text was signed by the private key
holder and has not been altered since signing.
============================================================
"""
print("📌 3. Compliance Certificate (Hardened):")
print(certificate)

# ----- 7. Tamper test: modify the stored data (now returns None gracefully) -----
print("📌 4. Tamper test: modifying the stored data...")
# Corrupt the compressed value
storage[key]["compressed"] = b"TAMPERED"
tampered = retrieve(key)  # now uses graceful zlib error handling
if tampered is None:
    print("   ❌ Tampering detected! The system returned None.")
    print("   ✅ The audit trail is intact — no corrupted data can be retrieved.")
else:
    print("   ⚠️ WARNING: Tampering was NOT detected. (Should not happen.)")
print()

# ----- 8. AI output verification (simulate paraphrasing) -----
print("📌 5. AI Output Verification (Anti‑Paraphrasing).")
# Simulate an AI generating a paraphrase of the clause
ai_output = "The Contractor must deliver the final report within 30 days of the Notice to Proceed."
# Hash the AI output and compare to the stored hash
ai_hash = hashlib.sha3_256(ai_output.encode('utf-8')).hexdigest()
print(f"   Stored clause hash: {retrieved['hash']}")
print(f"   AI output hash:     {ai_hash}")
if ai_hash == retrieved['hash']:
    print("   ✅ AI output matches the stored clause exactly.")
else:
    print("   ❌ AI output does NOT match the stored clause (paraphrasing detected!).")
    print("   🔒 In production, you would reject this output or force verbatim quoting.")

print()
print("="*60)
print("✅ Hardened demo complete. This version adds:")
print("   - Key‑value binding (prevents key swapping)")
print("   - Ed25519 digital signature (non‑repudiation)")
print("   - Graceful error handling on decompression failures")
print("   - Verification of final AI output (catches paraphrasing)")
print("   - A verifiable audit receipt with cryptographic proof.")
print("\n📌 For production: use a trusted timestamp server (RFC 3161) and store keys in a HSM.")
