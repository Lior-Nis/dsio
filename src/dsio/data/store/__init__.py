"""One immutable, memory-mapped store for homogeneous numeric samples.

The payload is written once as continuous rows. Persisted entities are addressable storage
samples; windowed training examples remain cheap derived views over the same bytes.
"""

from dsio.data.store.builder import SignalStoreBuilder
from dsio.data.store.layout import (
    DATA_ROOT_ENV,
    DEFAULT_DATA_ROOT,
    ENTITIES_FILE,
    INDEX_FILE,
    MANIFEST_FILE,
    SIGNAL_FILE,
    STORE_SCHEMA_VERSION,
    Entity,
    StoredSample,
    StoreError,
    StoreManifest,
)
from dsio.data.store.reader import SignalStore, data_root, list_stores

__all__ = [
    "DATA_ROOT_ENV",
    "DEFAULT_DATA_ROOT",
    "ENTITIES_FILE",
    "INDEX_FILE",
    "MANIFEST_FILE",
    "SIGNAL_FILE",
    "STORE_SCHEMA_VERSION",
    "Entity",
    "SignalStore",
    "SignalStoreBuilder",
    "StoreError",
    "StoreManifest",
    "StoredSample",
    "data_root",
    "list_stores",
]
