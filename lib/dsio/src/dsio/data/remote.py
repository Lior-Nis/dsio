"""Content-addressed remotes: the committed manifest is the whole index.

The property this exists to deliver is the fresh-clone test: clone into an empty directory,
``uv sync --locked``, ``dsio data pull``, ``dsio reproduce <run_id>`` — with no manual steps.
That is the test FORGE failed, because its checkpoints reached for a hardcoded encoder path
that existed on exactly one machine.

**The manifest is the index, so there is no remote index to keep in step.** ``manifest.yaml``
is committed to git and already names every file with its sha256. Pull reads the local
committed manifest, fetches each object by its hash, and verifies. No listing call, no
catalogue that can disagree with what git says, and no way for the remote to hand back
something other than the bytes the manifest names.

**Objects are stored by content hash, never by path.** The layout is
``<prefix>/objects/<first two hex>/<sha256>``. Three consequences, and the third is the one
that matters:

1. re-pushing an unchanged store is a few existence checks and no transfer;
2. two stores sharing a file store it once;
3. **old manifests never break.** A split file binds itself to ``store_manifest_sha256``;
   a run record pins its data. Under path-addressing, re-ingesting a corpus overwrites
   ``signal.bin`` and every prior result becomes unreproducible in silence. Under
   content-addressing the old bytes are still there, under their own name, and a
   twelve-month-old run still pulls exactly what it was trained on.

Anything fsspec can reach works: ``s3://``, ``gs://``, ``az://``, ``hf://datasets/user/repo``
and a plain local directory for testing. HuggingFace Hub is the interesting one for large
corpora — Xet's content-defined chunking dedupes *within* files, which the scheme here
cannot do on its own because a one-row append changes the whole digest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from dsio.contracts import atomic_write, sha256_of_file
from dsio.data.store import (
    ENTITIES_FILE,
    INDEX_FILE,
    MANIFEST_FILE,
    SIGNAL_FILE,
    SignalStore,
    StoreManifest,
    data_root,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fsspec import AbstractFileSystem

REMOTE_ENV = "DSIO_DATA_REMOTE"
REMOTES_FILE = "remotes.yaml"
OBJECTS_DIR = "objects"

#: Files a store is made of, paired with the manifest field naming each one's digest.
PAYLOAD: tuple[tuple[str, str], ...] = (
    (SIGNAL_FILE, "signal_sha256"),
    (INDEX_FILE, "index_sha256"),
    (ENTITIES_FILE, "entities_sha256"),
)


class RemoteError(RuntimeError):
    """Raised when a remote is unconfigured, unreachable, or missing an object."""


class RemoteIntegrityError(RemoteError):
    """Raised when a remote returns bytes that do not match the committed manifest.

    Separate from :class:`RemoteError` because the two demand different responses: a
    missing object means push it, while wrong bytes mean something is seriously wrong with
    the remote and nothing fetched from it should be trusted until that is understood.
    """


@dataclass(frozen=True)
class Transfer:
    """What one object transfer did. ``skipped`` is the common case on a second push."""

    filename: str
    digest: str
    bytes: int
    skipped: bool


def resolve_remote(name: str, explicit: str | None = None, root: Path | None = None) -> str:
    """Find the remote URL for a store: explicit, then env, then committed mapping.

    The committed ``stores/remotes.yaml`` is what makes the fresh-clone test pass without
    manual steps — an environment variable cannot be cloned. The other two exist so a
    one-off push somewhere else does not require editing a tracked file.
    """
    if explicit:
        return explicit
    from_env = os.environ.get(REMOTE_ENV)
    if from_env:
        return from_env

    mapping_path = (root or data_root()) / REMOTES_FILE
    if mapping_path.is_file():
        mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
        remotes = mapping.get("remotes", mapping)
        if isinstance(remotes, dict):
            url = remotes.get(name) or remotes.get("default")
            if isinstance(url, str) and url:
                return url

    raise RemoteError(
        f"no remote configured for store {name!r}. Pass --remote, set {REMOTE_ENV}, or "
        f"commit a mapping at {mapping_path} like:\n"
        "  remotes:\n"
        "    default: s3://bucket/dsio\n"
        f"    {name}: hf://datasets/you/{name}"
    )


def filesystem(url: str) -> tuple[AbstractFileSystem, str]:
    """Open an fsspec filesystem for ``url`` and return it with the path inside it."""
    try:
        import fsspec
    except ModuleNotFoundError as error:  # pragma: no cover - depends on the install
        raise RemoteError(
            "remotes need fsspec, which is an optional extra; install dsio[data] "
            "(and the backend package, e.g. s3fs or huggingface_hub)"
        ) from error
    fs, _, paths = fsspec.get_fs_token_paths(url)
    return fs, paths[0]


def object_key(prefix: str, digest: str) -> str:
    """Where an object lives on the remote.

    Fanned out by the first two hex characters. Some backends degrade badly on a directory
    with a million entries, and a corpus of a few hundred stores plus their history reaches
    that; the fan-out costs nothing and is what every content-addressed store does.
    """
    return f"{prefix.rstrip('/')}/{OBJECTS_DIR}/{digest[:2]}/{digest}"


def push(
    store: SignalStore | Path | str,
    *,
    remote: str | None = None,
    dry_run: bool = False,
) -> list[Transfer]:
    """Upload a store's payload, skipping objects the remote already has.

    Verifies locally before uploading. Pushing bytes that do not match the manifest would
    publish a corrupt store under a digest that promises otherwise, which is worse than
    failing.
    """
    signal = store if isinstance(store, SignalStore) else SignalStore(Path(store))
    signal.verify()
    manifest = signal.manifest()
    url = resolve_remote(signal.path.name, remote, root=signal.path.parent)
    fs, prefix = filesystem(url)

    transfers: list[Transfer] = []
    for filename, field in PAYLOAD:
        digest = getattr(manifest, field)
        source = signal.path / filename
        key = object_key(prefix, digest)
        size = source.stat().st_size
        if fs.exists(key):
            transfers.append(Transfer(filename, digest, size, skipped=True))
            continue
        if not dry_run:
            fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
            fs.put_file(str(source), key)
        transfers.append(Transfer(filename, digest, size, skipped=False))
    return transfers


def pull(
    name: str,
    *,
    remote: str | None = None,
    root: Path | None = None,
    force: bool = False,
) -> list[Transfer]:
    """Fetch a store's payload using its committed manifest, verifying every object.

    Fails closed on a digest mismatch and leaves nothing behind: each object is written to
    a temporary name, hashed, and only then moved into place. A partially-written store
    that verifies as complete is precisely the failure the manifest exists to prevent.
    """
    directory = (root or data_root()) / name
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise RemoteError(
            f"no committed manifest at {manifest_path}. The manifest is the index — it is "
            "what tells pull which bytes belong to this store — so it must be in git "
            "before anything can be fetched."
        )
    manifest = StoreManifest.model_validate(yaml.safe_load(manifest_path.read_text()))
    url = resolve_remote(name, remote, root=directory.parent)
    fs, prefix = filesystem(url)

    transfers: list[Transfer] = []
    for filename, field in PAYLOAD:
        digest = getattr(manifest, field)
        destination = directory / filename
        if not force and destination.is_file() and sha256_of_file(str(destination)) == digest:
            transfers.append(
                Transfer(filename, digest, destination.stat().st_size, skipped=True)
            )
            continue

        key = object_key(prefix, digest)
        if not fs.exists(key):
            raise RemoteError(
                f"{name}/{filename} should have digest {digest[:12]} but the remote has no "
                f"such object at {key}. Either it was never pushed, or this manifest is "
                "newer than the remote."
            )
        directory.mkdir(parents=True, exist_ok=True)
        staging = destination.with_suffix(destination.suffix + ".partial")
        fs.get_file(key, str(staging))

        actual = sha256_of_file(str(staging))
        if actual != digest:
            staging.unlink(missing_ok=True)
            raise RemoteIntegrityError(
                f"{name}/{filename} downloaded with digest {actual[:12]}, but the committed "
                f"manifest says {digest[:12]}; refusing to install it"
            )
        os.replace(staging, destination)
        transfers.append(Transfer(filename, digest, destination.stat().st_size, skipped=False))

    SignalStore(directory).verify()
    return transfers


def status(
    name: str,
    *,
    remote: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Report, per file, whether the bytes are here, there, both or neither."""
    directory = (root or data_root()) / name
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise RemoteError(f"no committed manifest at {manifest_path}")
    manifest = StoreManifest.model_validate(yaml.safe_load(manifest_path.read_text()))
    url = resolve_remote(name, remote, root=directory.parent)
    fs, prefix = filesystem(url)

    files = []
    for filename, field in PAYLOAD:
        digest = getattr(manifest, field)
        local = directory / filename
        present = local.is_file() and sha256_of_file(str(local)) == digest
        files.append(
            {
                "file": filename,
                "digest": digest[:16],
                "local": present,
                "remote": bool(fs.exists(object_key(prefix, digest))),
                "bytes": local.stat().st_size if local.is_file() else None,
            }
        )
    return {
        "store": name,
        "remote": url,
        "complete": all(entry["local"] for entry in files),
        "pushed": all(entry["remote"] for entry in files),
        "files": files,
    }


def write_remotes(mapping: dict[str, str], root: Path | None = None) -> Path:
    """Write the committed store-to-remote mapping."""
    directory = root or data_root()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REMOTES_FILE
    atomic_write(
        path,
        (
            "# dsio remotes. Committed to git so a fresh clone can `dsio data pull`\n"
            "# with no manual steps; an environment variable cannot be cloned.\n"
            + yaml.safe_dump({"remotes": mapping}, sort_keys=True)
        ).encode(),
    )
    return path
