"""The on-disk index format for a canonical signal store.

A store is two files sharing a prefix: ``signal.bin`` holds raw C-contiguous samples, and
``signal.idx`` holds everything needed to interpret them. That split is Megatron-LM's, and
it is here for the same reason: the payload stays a flat array the OS can map, while all
structure lives in a small file that is cheap to read and easy to version.

Three rules the layout enforces, each of which is a bug someone else has already shipped:

**The header carries an explicit version.** A format without one cannot be changed safely,
and every long-lived store format eventually changes.

**The dtype is recorded, not assumed.** Reading float32 bytes as float16 produces plausible
numbers rather than an error, which is the worst possible failure for a data layer.

**Entity boundaries are explicit.** Signal is a concatenation of recordings — sessions,
machines, symbols — and a window must never span two of them. Storing the offsets makes
that checkable rather than a convention.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

MAGIC = b"DSIOIDX\x00"
FORMAT_VERSION = 1
HEADER_SIZE = 64
"""Fixed header size, so the offset array always starts at a known, aligned position."""


class IndexFormatError(ValueError):
    """Raised when an index file is not a readable dsio index."""


class DTypeCode(IntEnum):
    """Numeric dtype codes. Values are permanent — never reuse or renumber."""

    FLOAT32 = 1
    FLOAT64 = 2
    FLOAT16 = 3
    INT16 = 4
    INT32 = 5
    INT8 = 6
    UINT8 = 7

    @classmethod
    def from_dtype(cls, dtype: np.dtype | str) -> DTypeCode:
        resolved = np.dtype(dtype)
        try:
            return _BY_DTYPE[resolved]
        except KeyError:
            raise IndexFormatError(
                f"dtype {resolved} has no dsio code; supported: "
                f"{', '.join(sorted(str(d) for d in _BY_DTYPE))}"
            ) from None

    def to_dtype(self) -> np.dtype:
        return _BY_CODE[self]


_BY_DTYPE: dict[np.dtype, DTypeCode] = {
    np.dtype("float32"): DTypeCode.FLOAT32,
    np.dtype("float64"): DTypeCode.FLOAT64,
    np.dtype("float16"): DTypeCode.FLOAT16,
    np.dtype("int16"): DTypeCode.INT16,
    np.dtype("int32"): DTypeCode.INT32,
    np.dtype("int8"): DTypeCode.INT8,
    np.dtype("uint8"): DTypeCode.UINT8,
}
_BY_CODE: dict[DTypeCode, np.dtype] = {code: dt for dt, code in _BY_DTYPE.items()}

# magic(8) version(u4) dtype(u1) pad(3) channels(u4) pad(4) n_entities(u8) n_rows(u8)
_HEADER_STRUCT = struct.Struct("<8sIBxxxIxxxxQQ")


@dataclass(frozen=True, slots=True)
class IndexHeader:
    """Everything needed to interpret ``signal.bin``."""

    version: int
    dtype: np.dtype
    channels: int
    n_entities: int
    n_rows: int

    @property
    def itemsize(self) -> int:
        return int(self.dtype.itemsize)

    @property
    def row_bytes(self) -> int:
        return self.itemsize * self.channels

    @property
    def payload_bytes(self) -> int:
        return self.row_bytes * self.n_rows

    def pack(self) -> bytes:
        packed = _HEADER_STRUCT.pack(
            MAGIC,
            self.version,
            int(DTypeCode.from_dtype(self.dtype)),
            self.channels,
            self.n_entities,
            self.n_rows,
        )
        return packed.ljust(HEADER_SIZE, b"\x00")

    @classmethod
    def unpack(cls, raw: bytes) -> IndexHeader:
        if len(raw) < HEADER_SIZE:
            raise IndexFormatError(
                f"index header is {len(raw)} bytes, expected at least {HEADER_SIZE}"
            )
        magic, version, dtype_code, channels, n_entities, n_rows = _HEADER_STRUCT.unpack(
            raw[: _HEADER_STRUCT.size]
        )
        if magic != MAGIC:
            raise IndexFormatError(
                f"not a dsio index: magic is {magic!r}, expected {MAGIC!r}"
            )
        if version != FORMAT_VERSION:
            raise IndexFormatError(
                f"index format version {version} is not supported by this build "
                f"(expected {FORMAT_VERSION}); rebuild the store or upgrade dsio"
            )
        try:
            dtype = DTypeCode(dtype_code).to_dtype()
        except ValueError:
            raise IndexFormatError(f"unknown dtype code {dtype_code} in index") from None
        return cls(
            version=version,
            dtype=dtype,
            channels=channels,
            n_entities=n_entities,
            n_rows=n_rows,
        )


def write_index(path: Path, header: IndexHeader, entity_offsets: np.ndarray) -> None:
    """Write an index file: header followed by ``n_entities + 1`` int64 row offsets.

    Offsets are cumulative starts with a terminating total, so entity ``i`` occupies rows
    ``[offsets[i], offsets[i + 1])`` and its length needs no separate array.
    """
    offsets = np.ascontiguousarray(entity_offsets, dtype=np.int64)
    if offsets.ndim != 1:
        raise IndexFormatError(f"entity offsets must be 1-D, got shape {offsets.shape}")
    if offsets.size != header.n_entities + 1:
        raise IndexFormatError(
            f"expected {header.n_entities + 1} offsets for {header.n_entities} entities, "
            f"got {offsets.size}"
        )
    if offsets.size and offsets[0] != 0:
        raise IndexFormatError(f"first entity offset must be 0, got {offsets[0]}")
    if offsets.size and offsets[-1] != header.n_rows:
        raise IndexFormatError(
            f"final offset {offsets[-1]} does not match n_rows {header.n_rows}"
        )
    if np.any(np.diff(offsets) < 0):
        raise IndexFormatError("entity offsets must be non-decreasing")

    from dsio.contracts import atomic_write

    atomic_write(path, header.pack() + offsets.tobytes())


def read_index(path: Path) -> tuple[IndexHeader, np.ndarray]:
    """Read an index file, returning its header and entity offsets."""
    raw = path.read_bytes()
    header = IndexHeader.unpack(raw)
    expected = HEADER_SIZE + (header.n_entities + 1) * 8
    if len(raw) != expected:
        raise IndexFormatError(
            f"index {path} is {len(raw)} bytes, expected {expected} for "
            f"{header.n_entities} entities; the file is truncated or corrupt"
        )
    offsets = np.frombuffer(raw, dtype=np.int64, offset=HEADER_SIZE)
    return header, offsets
