"""Canonical serialization and content hashing.

Every identity in dsio — config hashes, cache keys, provenance digests — routes through
:func:`canonical_json`. If two structures serialize to the same
bytes here, they must be the same thing everywhere in the system. That means the encoding
has to be total and deterministic, with no float ambiguity and no dict-ordering ambiguity.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

DIGEST_PREFIX_LEN = 12
"""Characters of a sha256 hex digest used for human-facing short keys."""


class NonCanonicalValueError(ValueError):
    """Raised when a value cannot be encoded deterministically."""


def _canonicalize(value: Any) -> Any:
    """Convert ``value`` into a JSON-safe structure with deterministic ordering.

    Fails closed on anything whose encoding would be ambiguous. A NaN that silently
    round-trips would let two different configs collide onto one hash, which is the
    worst possible failure for a cache key.
    """
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise NonCanonicalValueError(
                f"non-finite float {value!r} cannot be part of a content hash"
            )
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise NonCanonicalValueError(
                    f"dict keys must be str for canonical encoding, got {type(key).__name__}"
                )
            out[key] = _canonicalize(item)
        return dict(sorted(out.items()))
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set | frozenset):
        # Sets have no inherent order; sort the *encoded* members so the result is stable
        # regardless of insertion order or hash seed.
        return sorted((_canonicalize(item) for item in value), key=json.dumps)
    raise NonCanonicalValueError(
        f"type {type(value).__name__} has no canonical encoding; "
        "convert it to a primitive before hashing"
    )


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON encoding of ``value``."""
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def sha256_of(value: Any) -> str:
    """Return the full hex sha256 of ``value``'s canonical encoding."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def short_digest(value: Any) -> str:
    """Return a short, human-quotable digest for cache directories and run ids."""
    return sha256_of(value)[:DIGEST_PREFIX_LEN]


def sha256_of_bytes(payload: bytes) -> str:
    """Return the hex sha256 of raw bytes (artifacts, lockfiles, patches)."""
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read incrementally so large artifacts stream."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
