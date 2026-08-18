"""Canonical stores, lazy windowed views, and the data root.

Store once, index many: a corpus is written once as continuous signal, and every window
configuration is an index of offsets over it. See docs/adr/0005 for why the payload is
flat binary rather than a chunked format.
"""

from dsio.data.cache import (
    CACHE_ROOT_ENV,
    BytesCodec,
    CacheEntry,
    CacheError,
    CachePolicy,
    Codec,
    CodePolicy,
    ConfigPolicy,
    EnvPolicy,
    ExplicitVersionPolicy,
    JsonCodec,
    NpzCodec,
    NumpyCodec,
    StageCache,
    cache_root,
)
from dsio.data.format import DTypeCode, IndexFormatError, IndexHeader
from dsio.data.readers import ReadError, SignalReader, open_reader
from dsio.data.remote import (
    REMOTE_ENV,
    RemoteError,
    RemoteIntegrityError,
    Transfer,
    pull,
    push,
    resolve_remote,
    status,
    write_remotes,
)
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
    "CACHE_ROOT_ENV",
    "DATA_ROOT_ENV",
    "REMOTE_ENV",
    "BytesCodec",
    "CacheEntry",
    "CacheError",
    "CachePolicy",
    "CodePolicy",
    "Codec",
    "ConfigPolicy",
    "DTypeCode",
    "Entity",
    "EnvPolicy",
    "ExplicitVersionPolicy",
    "IndexFormatError",
    "IndexHeader",
    "JsonCodec",
    "NpzCodec",
    "NumpyCodec",
    "ReadError",
    "RemoteError",
    "RemoteIntegrityError",
    "SignalReader",
    "SignalStore",
    "SignalStoreBuilder",
    "StageCache",
    "StoreError",
    "StoreManifest",
    "Transfer",
    "WindowIndex",
    "WindowSpec",
    "WindowView",
    "build_index",
    "cache_root",
    "data_root",
    "list_stores",
    "load_or_build",
    "open_reader",
    "pull",
    "push",
    "resolve_remote",
    "status",
    "write_remotes",
]
