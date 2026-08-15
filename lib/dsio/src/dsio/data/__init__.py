"""Canonical stores, lazy windowed views, and the data root.

Store once, index many: a corpus is written once as continuous signal, and every window
configuration is an index of offsets over it. See docs/adr/0005 for why the payload is
flat binary rather than a chunked format.
"""

from dsio.data.format import DTypeCode, IndexFormatError, IndexHeader
from dsio.data.readers import ReadError, SignalReader, open_reader
from dsio.data.store import (
    DATA_ROOT_ENV,
    Entity,
    SignalStore,
    SignalStoreBuilder,
    StoreError,
    StoreManifest,
    data_root,
    list_stores,
)
from dsio.data.views import (
    WindowIndex,
    WindowSpec,
    WindowView,
    build_index,
    load_or_build,
)

__all__ = [
    "DATA_ROOT_ENV",
    "DTypeCode",
    "Entity",
    "IndexFormatError",
    "IndexHeader",
    "ReadError",
    "SignalReader",
    "SignalStore",
    "SignalStoreBuilder",
    "StoreError",
    "StoreManifest",
    "WindowIndex",
    "WindowSpec",
    "WindowView",
    "build_index",
    "data_root",
    "list_stores",
    "load_or_build",
    "open_reader",
]
