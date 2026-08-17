"""Content-addressed stage cache.

The rule the whole module serves: **materialise what is expensive and deterministic; index
what is cheap and combinatorial.** Tokenisation and embedding extraction belong here.
Windowing and shuffling do not — they are offsets, and :mod:`dsio.data.views` already owns
them.

Key derivation is a composable list of policies, following Flyte's `CachePolicy` shape
rather than a single fixed rule. The default composes code, config and environment.

Two consequences worth stating plainly, because they resolve the argument that every
caching system eventually has:

**Code is in the key by default, so stale results are structurally impossible.** Hamilton
documents its own version of this failing open — edit a helper, get a silent stale hit. DVC
and Flyte leave it to a human to bump a version string, and Flyte's own docstring concedes
you must "manually bump this version if the function body has changed". Hashing the source
means a drift *warning* is never needed, because drift cannot occur: different code is a
different key.

**Upstream contributes its output digest, not its key.** That is early cutoff, and it is
what makes aggressive code hashing affordable. Reformat a parsing stage and it recomputes —
but if the bytes it produces are unchanged, every downstream stage keeps its key and its
cache. Nix calls this content-addressed derivations; Bazel gets it from action outputs.

What is deliberately **not** in the key: anything that changes only how fast you got the
answer. Worker counts, device, log level. Anything that changes the output bytes is in.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import sys
import textwrap
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

import numpy as np
from pydantic import Field

from dsio.contracts import DsioModel, atomic_write, canonical_json, sha256_of, sha256_of_bytes

T = TypeVar("T")

CACHE_ROOT_ENV = "DSIO_CACHE_ROOT"
DEFAULT_CACHE_ROOT = Path("cache")

META_FILE = "meta.json"

SPEED_ONLY_KEYS = frozenset(
    {
        "num_workers",
        "workers",
        "device",
        "devices",
        "gpus",
        "log_level",
        "verbose",
        "progress",
        "prefetch_factor",
        "pin_memory",
        "persistent_workers",
    }
)
"""Config keys excluded from every key: they change speed, never output bytes."""


class CacheError(RuntimeError):
    """Raised when a cache entry is missing, corrupt, or contradicts its metadata."""


def cache_root() -> Path:
    return Path(os.environ.get(CACHE_ROOT_ENV, DEFAULT_CACHE_ROOT))


# --- key derivation -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageContext:
    """Everything a policy may look at when contributing to a key."""

    name: str
    version: int
    config: dict[str, Any]
    fn: Callable[..., Any] | None
    upstream_digests: tuple[str, ...]
    extra_code: tuple[Callable[..., Any], ...] = ()


@runtime_checkable
class CachePolicy(Protocol):
    """One contribution to a cache key."""

    @property
    def name(self) -> str: ...

    def contribute(self, ctx: StageContext) -> Any: ...


def normalised_source(fn: Callable[..., Any]) -> str:
    """Source of ``fn`` with comments, docstrings and formatting removed.

    Reformatting a stage must not invalidate it. Parsing to an AST and dumping the tree
    discards comments and layout, and the docstring is stripped explicitly because it is a
    real statement in the tree.

    The top-level name is blanked too: the stage's identity is the name passed to
    :meth:`StageCache.run`, so the Python identifier is incidental and renaming a function
    should not throw away its results.

    For a callable object rather than a function, the *class* source is used. The fallback
    never includes ``repr()``, because that embeds a memory address and would give the same
    logic a different key on every process — a cache that never hits.
    """
    target: Any = fn
    if not (inspect.isfunction(fn) or inspect.ismethod(fn) or inspect.isclass(fn)):
        target = type(fn)

    try:
        source = textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError):
        module = getattr(target, "__module__", "?")
        qualname = getattr(target, "__qualname__", getattr(target, "__name__", "?"))
        return f"<unavailable:{module}.{qualname}>"

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            node.name = "_"
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


class CodePolicy:
    """Hash the stage function's normalised source.

    **Known limit, stated because every system with this feature has it and most do not
    say so:** only the function's own body is hashed. A helper it calls is invisible, so
    editing the helper yields a cache hit on stale logic. Hamilton documents the identical
    gap. Pass the helpers as ``extra_code`` to close it for a given stage, or bump
    ``version`` when you change one.
    """

    name: ClassVar[str] = "code"

    def contribute(self, ctx: StageContext) -> Any:
        if ctx.fn is None:
            return None
        sources = [normalised_source(ctx.fn)]
        sources.extend(normalised_source(extra) for extra in ctx.extra_code)
        return [sha256_of_bytes(s.encode("utf-8")) for s in sources]


class ConfigPolicy:
    """Hash the stage's config subtree, minus keys that only affect speed."""

    name: ClassVar[str] = "config"

    def __init__(self, ignore: frozenset[str] = SPEED_ONLY_KEYS) -> None:
        self.ignore = ignore

    def contribute(self, ctx: StageContext) -> Any:
        return _strip(ctx.config, self.ignore)


class EnvPolicy:
    """Hash the dependency lockfile and interpreter version.

    A stage's output can change because a library changed underneath it, with no edit to
    the config or the code. Snakemake ships a `software-env` rerun trigger for exactly
    this; Flyte folds the container image into its version parameters.
    """

    name: ClassVar[str] = "env"

    def __init__(self, lock_path: Path | None = None) -> None:
        self.lock_path = lock_path if lock_path is not None else Path("uv.lock")

    def contribute(self, ctx: StageContext) -> Any:
        from dsio.contracts import sha256_of_file

        lock = sha256_of_file(str(self.lock_path)) if self.lock_path.is_file() else None
        return {"python": ".".join(map(str, sys.version_info[:2])), "lock": lock}


class ExplicitVersionPolicy:
    """Contribute only the declared version — the Flyte/DVC posture.

    Use when a stage's logic is stable and you want changes to be a deliberate act, not a
    consequence of touching the file. Combine with :class:`EnvPolicy`, never alone.
    """

    name: ClassVar[str] = "explicit"

    def contribute(self, ctx: StageContext) -> Any:
        return ctx.version


DEFAULT_POLICIES: tuple[CachePolicy, ...] = (CodePolicy(), ConfigPolicy(), EnvPolicy())


def _strip(value: Any, ignore: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v, ignore) for k, v in value.items() if k not in ignore}
    if isinstance(value, list | tuple):
        return [_strip(v, ignore) for v in value]
    return value


def compute_key(ctx: StageContext, policies: Sequence[CachePolicy]) -> str:
    """Derive the cache key from the stage identity plus every policy's contribution."""
    payload = {
        "stage": ctx.name,
        "version": ctx.version,
        "upstream": list(ctx.upstream_digests),
        "policies": {policy.name: policy.contribute(ctx) for policy in policies},
    }
    return sha256_of(payload)


# --- codecs -------------------------------------------------------------------------


@runtime_checkable
class Codec(Protocol):
    """How a stage's result is written and read."""

    @property
    def extension(self) -> str: ...

    def dump(self, value: Any) -> bytes: ...

    def load(self, raw: bytes) -> Any: ...


class NumpyCodec:
    extension: ClassVar[str] = "npy"

    def dump(self, value: Any) -> bytes:
        import io

        buffer = io.BytesIO()
        np.save(buffer, np.asarray(value), allow_pickle=False)
        return buffer.getvalue()

    def load(self, raw: bytes) -> Any:
        import io

        return np.load(io.BytesIO(raw), allow_pickle=False)


class NpzCodec:
    """Several named arrays, for a stage returning a dict."""

    extension: ClassVar[str] = "npz"

    def dump(self, value: Any) -> bytes:
        import io

        buffer = io.BytesIO()
        arrays = {str(k): np.asarray(v) for k, v in dict(value).items()}
        np.savez(buffer, **arrays)  # type: ignore[arg-type]
        return buffer.getvalue()

    def load(self, raw: bytes) -> Any:
        import io

        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}


class JsonCodec:
    extension: ClassVar[str] = "json"

    def dump(self, value: Any) -> bytes:
        return canonical_json(value).encode("utf-8")

    def load(self, raw: bytes) -> Any:
        return json.loads(raw)


class BytesCodec:
    extension: ClassVar[str] = "bin"

    def dump(self, value: Any) -> bytes:
        return bytes(value)

    def load(self, raw: bytes) -> Any:
        return raw


# --- entries ------------------------------------------------------------------------


class EntryMeta(DsioModel):
    """What was computed, from what, and what came out."""

    stage: str
    key: str
    version: int
    created_at: str
    output_digest: str
    output_bytes: int
    seconds: float
    codec: str
    upstream: tuple[str, ...] = ()
    policies: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A materialised stage result."""

    path: Path
    meta: EntryMeta
    codec: Codec

    @property
    def digest(self) -> str:
        """The *output* digest — what upstream contributes downstream. See early cutoff."""
        return self.meta.output_digest

    @property
    def data_path(self) -> Path:
        return self.path / f"data.{self.codec.extension}"

    def load(self) -> Any:
        """Read the artifact, verifying its digest.

        Existence is not integrity: Kedro's ``--only-missing-outputs`` treats a truncated
        file as valid forever. A digest mismatch is a hard failure, never a warning.
        """
        raw = self.data_path.read_bytes()
        actual = sha256_of_bytes(raw)
        if actual != self.meta.output_digest:
            raise CacheError(
                f"cache entry {self.meta.stage}/{self.meta.key[:12]} has digest "
                f"{actual[:12]} but recorded {self.meta.output_digest[:12]}; the artifact "
                "is corrupt. Delete it and recompute."
            )
        return self.codec.load(raw)


class StageCache:
    """Content-addressed cache of expensive, deterministic stage outputs."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        policies: Sequence[CachePolicy] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else cache_root()
        self.policies = tuple(policies) if policies is not None else DEFAULT_POLICIES

    def key_for(
        self,
        name: str,
        *,
        version: int = 1,
        config: dict[str, Any] | None = None,
        fn: Callable[..., Any] | None = None,
        upstream: Sequence[CacheEntry] = (),
        extra_code: Sequence[Callable[..., Any]] = (),
    ) -> str:
        ctx = StageContext(
            name=name,
            version=version,
            config=config or {},
            fn=fn,
            upstream_digests=tuple(entry.digest for entry in upstream),
            extra_code=tuple(extra_code),
        )
        return compute_key(ctx, self.policies)

    def run(
        self,
        name: str,
        fn: Callable[[], T],
        *,
        version: int = 1,
        config: dict[str, Any] | None = None,
        codec: Codec | None = None,
        upstream: Sequence[CacheEntry] = (),
        extra_code: Sequence[Callable[..., Any]] = (),
        force: bool = False,
    ) -> CacheEntry:
        """Return the cached entry for this stage, computing it only if absent.

        ``upstream`` entries contribute their *output* digests, so a rebuilt upstream that
        produced identical bytes leaves this stage's key — and its cache — untouched.
        """
        chosen = codec or NumpyCodec()
        ctx = StageContext(
            name=name,
            version=version,
            config=config or {},
            fn=fn,
            upstream_digests=tuple(entry.digest for entry in upstream),
            extra_code=tuple(extra_code),
        )
        key = compute_key(ctx, self.policies)
        path = self.root / name / key

        if not force:
            existing = self._read(path, chosen)
            if existing is not None:
                return existing

        started = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter() - started

        raw = chosen.dump(value)
        meta = EntryMeta(
            stage=name,
            key=key,
            version=version,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            output_digest=sha256_of_bytes(raw),
            output_bytes=len(raw),
            seconds=elapsed,
            codec=type(chosen).__name__,
            upstream=ctx.upstream_digests,
            policies={p.name: p.contribute(ctx) for p in self.policies},
        )
        atomic_write(path / f"data.{chosen.extension}", raw)
        atomic_write(
            path / META_FILE,
            json.dumps(meta.model_dump(mode="json"), indent=2, sort_keys=True).encode(),
        )
        return CacheEntry(path=path, meta=meta, codec=chosen)

    def _read(self, path: Path, codec: Codec) -> CacheEntry | None:
        meta_path = path / META_FILE
        if not meta_path.is_file():
            return None
        meta = EntryMeta.model_validate(json.loads(meta_path.read_text()))
        entry = CacheEntry(path=path, meta=meta, codec=codec)
        if not entry.data_path.is_file():
            raise CacheError(
                f"cache entry {meta.stage}/{meta.key[:12]} has metadata but no artifact at "
                f"{entry.data_path}; a write was interrupted. Delete the directory."
            )
        return entry

    def entries(self, name: str | None = None) -> list[EntryMeta]:
        base = self.root / name if name else self.root
        if not base.is_dir():
            return []
        return [
            EntryMeta.model_validate(json.loads(path.read_text()))
            for path in sorted(base.glob(f"{'' if name else '*/'}*/{META_FILE}"))
        ]

    def size_bytes(self) -> int:
        if not self.root.is_dir():
            return 0
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
