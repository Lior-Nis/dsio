"""Canonical stores, lazy windowed views, and the data root.

Store once, index many: a corpus is written once as continuous signal, and every window
configuration is an index of offsets over it. See docs/adr/0005 for why the payload is
flat binary rather than a chunked format.
"""

from dsio.data.adapters import (
    KeyedExamples,
    SignalExamples,
    TableExamples,
    entity_examples,
)
from dsio.data.examples import Examples, ExamplesError, assert_consistent, group_attribute
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
from dsio.data.staging import StagingError, stage
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
    "REMOTE_ENV",
    "DTypeCode",
    "Entity",
    "Examples",
    "ExamplesError",
    "IndexFormatError",
    "IndexHeader",
    "KeyedExamples",
    "ReadError",
    "RemoteError",
    "RemoteIntegrityError",
    "SignalExamples",
    "SignalReader",
    "SignalStore",
    "SignalStoreBuilder",
    "StagingError",
    "StoreError",
    "StoreManifest",
    "TableExamples",
    "Transfer",
    "WindowIndex",
    "WindowSpec",
    "WindowView",
    "assert_consistent",
    "build_index",
    "data_root",
    "entity_examples",
    "group_attribute",
    "list_stores",
    "load_or_build",
    "open_reader",
    "pull",
    "push",
    "resolve_remote",
    "stage",
    "status",
    "write_remotes",
]
