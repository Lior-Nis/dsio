"""Model artifact registry: pinned references, fail-closed integrity.

Two rules make this different from a directory of checkpoints.

**References are pinned, never "latest".** A :class:`ModelRef` carries name, version and
digest. A checkpoint that reloads a dependency from a
hardcoded path, so reproduction failed on a fresh clone; a checkpoint that names a moving
target is not a reproducible artifact.

**Loads fail closed.** A digest mismatch raises rather than returning a model. Returning
the wrong weights is worse than returning nothing, because training continues and the
numbers look plausible.

The registry stores raw bytes and never imports a modelling framework, so it can hold a
pickled sklearn pipeline, a torch state dict, or a JSON blob of coefficients without
knowing the difference.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from dsio.contracts import DsioModel, atomic_write, sha256_of, sha256_of_bytes

REGISTRY_ROOT_ENV = "DSIO_REGISTRY_ROOT"
DEFAULT_REGISTRY_ROOT = Path("models")

MANIFEST_FILE = "manifest.jsonl"
ARTIFACT_FILE = "artifact.bin"
LOCK_SUFFIX = ".lock"


class RegistryIntegrityError(RuntimeError):
    """Raised when the on-disk registry contradicts its manifest."""


class ModelRef(DsioModel):
    """A pinned reference to one model version.

    There is deliberately no way to express "latest". Resolving a moving alias at load
    time is how a reproduction silently becomes a different experiment.
    """

    name: str
    version: int
    digest: str

    def __str__(self) -> str:
        return f"{self.name}:v{self.version}@{self.digest[:12]}"


class ModelVersion(DsioModel):
    """A manifest row. Append-only and authoritative for what is valid."""

    name: str
    version: int
    digest: str
    created_at: str
    size_bytes: int

    run_id: str | None = None
    config_hash: str | None = None
    code_hash: str | None = None
    data_snapshot_ids: tuple[str, ...] = ()
    seed: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    provenance_digest: str | None = None

    @property
    def ref(self) -> ModelRef:
        return ModelRef(name=self.name, version=self.version, digest=self.digest)


def compute_provenance_digest(
    *,
    digest: str,
    config_hash: str | None,
    code_hash: str | None,
    data_snapshot_ids: tuple[str, ...],
    seed: int | None,
    metrics: dict[str, float],
) -> str:
    """Hash the whole training context, not just the weights.

    Two models with identical bytes but different training data are different models.
    Committing to the surrounding context is what makes that distinction survive.
    """
    return sha256_of(
        {
            "digest": digest,
            "config_hash": config_hash,
            "code_hash": code_hash,
            "data_snapshot_ids": sorted(data_snapshot_ids),
            "seed": seed,
            "metrics": dict(sorted(metrics.items())),
        }
    )


@contextmanager
def _name_lock(root: Path, name: str) -> Iterator[None]:
    """Serialize writers for one model name across processes."""
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f"{name}{LOCK_SUFFIX}"
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)


class ModelRegistry:
    """Filesystem registry of versioned model artifacts."""

    def __init__(self, root: Path | str | None = None) -> None:
        if root is not None:
            self.root = Path(root)
        else:
            self.root = Path(os.environ.get(REGISTRY_ROOT_ENV, DEFAULT_REGISTRY_ROOT))

    def save(
        self,
        name: str,
        payload: bytes,
        *,
        run_id: str | None = None,
        config_hash: str | None = None,
        code_hash: str | None = None,
        data_snapshot_ids: tuple[str, ...] = (),
        seed: int | None = None,
        metrics: dict[str, float] | None = None,
    ) -> ModelVersion:
        """Store ``payload`` as the next version of ``name`` and return its manifest row."""
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"invalid model name {name!r}")

        with _name_lock(self.root, name):
            version = self._next_version(name)
            digest = sha256_of_bytes(payload)
            directory = self.root / name / f"v{version}"
            directory.mkdir(parents=True, exist_ok=False)
            atomic_write(directory / ARTIFACT_FILE, payload)

            row = ModelVersion(
                name=name,
                version=version,
                digest=digest,
                created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                size_bytes=len(payload),
                run_id=run_id,
                config_hash=config_hash,
                code_hash=code_hash,
                data_snapshot_ids=data_snapshot_ids,
                seed=seed,
                metrics=metrics or {},
                provenance_digest=compute_provenance_digest(
                    digest=digest,
                    config_hash=config_hash,
                    code_hash=code_hash,
                    data_snapshot_ids=data_snapshot_ids,
                    seed=seed,
                    metrics=metrics or {},
                ),
            )
            self._append_manifest(name, row)
            return row

    def load(self, ref: ModelRef) -> bytes:
        """Return the artifact bytes for ``ref``, verifying its digest.

        Raises :class:`RegistryIntegrityError` if the manifest, the reference, and the
        bytes on disk do not all agree.
        """
        row = self.get_version(ref.name, ref.version)
        if row.digest != ref.digest:
            raise RegistryIntegrityError(
                f"{ref.name}:v{ref.version} manifest digest {row.digest[:12]} does not match "
                f"the requested {ref.digest[:12]}; the reference is stale or the manifest changed"
            )

        path = self.root / ref.name / f"v{ref.version}" / ARTIFACT_FILE
        if not path.is_file():
            raise RegistryIntegrityError(
                f"{ref.name}:v{ref.version} is in the manifest but {path} is missing"
            )
        payload = path.read_bytes()
        actual = sha256_of_bytes(payload)
        if actual != row.digest:
            raise RegistryIntegrityError(
                f"{ref.name}:v{ref.version} artifact digest {actual[:12]} does not match "
                f"manifest {row.digest[:12]}; the artifact on disk has been corrupted"
            )
        return payload

    def get_version(self, name: str, version: int) -> ModelVersion:
        for row in self.versions(name):
            if row.version == version:
                return row
        raise RegistryIntegrityError(f"{name}:v{version} is not in the manifest")

    def versions(self, name: str) -> list[ModelVersion]:
        """Return manifest rows for ``name``, oldest first.

        Duplicate versions are an integrity failure, not a last-one-wins situation: they
        mean two writers allocated the same number and one artifact is unreachable.
        """
        path = self.root / name / MANIFEST_FILE
        if not path.is_file():
            return []
        rows: list[ModelVersion] = []
        seen: set[int] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = ModelVersion.model_validate(json.loads(line))
                if row.version in seen:
                    raise RegistryIntegrityError(
                        f"{name} manifest contains version {row.version} more than once"
                    )
                seen.add(row.version)
                rows.append(row)
        return sorted(rows, key=lambda item: item.version)

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            path.parent.name for path in self.root.glob(f"*/{MANIFEST_FILE}")
        )

    def _next_version(self, name: str) -> int:
        """Allocate off both the manifest and the directory listing.

        Taking the max of the two means a crash between creating ``v3/`` and appending
        its manifest row can never hand ``3`` out a second time — the orphaned directory
        still reserves the number.
        """
        manifest_max = max((row.version for row in self.versions(name)), default=0)
        directory = self.root / name
        listing_max = 0
        if directory.is_dir():
            for child in directory.iterdir():
                if child.is_dir() and child.name.startswith("v") and child.name[1:].isdigit():
                    listing_max = max(listing_max, int(child.name[1:]))
        return max(manifest_max, listing_max) + 1

    def _append_manifest(self, name: str, row: ModelVersion) -> None:
        path = self.root / name / MANIFEST_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def promotion_blockers(record: Any) -> list[str]:
    """Return reasons ``record`` may not be promoted, empty if it may.

    This is where the clean-tree gate lives. Exploration is never blocked; promotion to a
    registered model is, because that artifact is the one that outlives the session.
    """
    blockers: list[str] = []
    git = getattr(record, "git", None)
    if git is None or getattr(git, "sha", None) is None:
        blockers.append("no git provenance was captured")
    elif getattr(git, "dirty", False):
        blockers.append("the working tree was dirty; commit the changes and rerun")
    env = getattr(record, "env", None)
    if env is None or getattr(env, "lock_sha256", None) is None:
        blockers.append("no dependency lockfile was captured")
    return blockers
