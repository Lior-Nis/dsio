"""Pure value types and encodings shared by every dsio subsystem.

This package is a leaf: it imports nothing else from dsio. Everything else may import it.
"""

from dsio.contracts.base import DsioModel
from dsio.contracts.hashing import (
    NonCanonicalValueError,
    canonical_json,
    sha256_of,
    sha256_of_bytes,
    sha256_of_file,
    short_digest,
)
from dsio.contracts.io import atomic_write, fsync_dir

__all__ = [
    "DsioModel",
    "NonCanonicalValueError",
    "atomic_write",
    "canonical_json",
    "fsync_dir",
    "sha256_of",
    "sha256_of_bytes",
    "sha256_of_file",
    "short_digest",
]
