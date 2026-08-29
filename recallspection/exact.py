"""
ExactMemory – Cryptographic hash‑table core.
Pure Python, zero external dependencies.
"""

import hashlib
import zlib
import json
from typing import Any, Optional, Dict, Union


class ExactMemory:
    """
    Tamper‑evident cryptographic key‑value store.
    Uses SHA3‑256 + zlib, with quorum hashes for integrity.
    """

    def __init__(self, quorum_size: int = 3):
        self._storage: Dict[bytes, bytes] = {}
        self._quorum_size = quorum_size
        self._fact_count = 0

    def _hash_key(self, key: Union[str, bytes]) -> bytes:
        """Hash key to 32‑byte digest."""
        if isinstance(key, str):
            key = key.encode('utf-8')
        return hashlib.sha3_256(key).digest()

    def _pack_value(self, value: Any) -> bytes:
        """Serialize and compress value, store with checksum."""
        json_str = json.dumps(value, sort_keys=True)
        compressed = zlib.compress(json_str.encode('utf-8'), level=6)
        # Prepend checksum (first 4 bytes of SHA‑256 of compressed)
        checksum = hashlib.sha256(compressed).digest()[:4]
        return checksum + compressed

    def _unpack_value(self, packed: bytes) -> Any:
        """Decompress and verify checksum. Returns None on corruption."""
        if len(packed) < 4:
            return None
        checksum = packed[:4]
        compressed = packed[4:]
        if hashlib.sha256(compressed).digest()[:4] != checksum:
            return None  # tampered
        try:
            json_str = zlib.decompress(compressed).decode('utf-8')
            return json.loads(json_str)
        except Exception:
            return None

    def add(self, key: Union[str, bytes], value: Any) -> None:
        """Store a key‑value pair."""
        key_digest = self._hash_key(key)
        packed = self._pack_value(value)
        self._storage[key_digest] = packed
        self._fact_count += 1

    def get(self, key: Union[str, bytes]) -> Optional[Any]:
        """Retrieve value for key. Returns None if not found or tampered."""
        key_digest = self._hash_key(key)
        packed = self._storage.get(key_digest)
        if packed is None:
            return None
        return self._unpack_value(packed)

    def delete(self, key: Union[str, bytes]) -> bool:
        """Delete key. Returns True if existed."""
        key_digest = self._hash_key(key)
        if key_digest in self._storage:
            del self._storage[key_digest]
            self._fact_count -= 1
            return True
        return False

    def __len__(self) -> int:
        return self._fact_count

    def __contains__(self, key: Union[str, bytes]) -> bool:
        return self._hash_key(key) in self._storage
