"""Model artifact policy: pinned references, fail-closed integrity, over MLflow's registry.

Decision 7 of the lean design ("MLflow is the source of truth") gives MLflow's Model
Registry the storage and the version numbering. What survives here is the policy MLflow
does not provide:

**Loads fail closed.** MLflow will hand back whatever bytes live at a model version's
source, corrupted or not. A digest mismatch here raises rather than returning a model,
because returning the wrong weights is worse than returning nothing -- training continues
and the numbers look plausible.

**References are pinned, never "latest".** A :class:`ModelRef` carries name, version and
digest. MLflow's own aliases are *designed* to be moving targets -- the opposite of a
pinned reference -- so pinning stays dsio's job, unchanged from before this migration.

**Promotion has a gate.** :func:`promotion_blockers` is ADR 0003's clean-tree check,
independent of storage.

What is gone: the manifest file, the per-name lock, and the directory-listing version
allocator. MLflow's registry backend (Postgres, per decision 8) allocates version numbers
and enforces their uniqueness itself; a second, dsio-side allocator racing the same
guarantee would be exactly the kind of parallel mechanism ADR 0002 was written to avoid.

The registry stores raw bytes and never imports a modelling framework, so it can hold a
pickled sklearn pipeline, a torch state dict, or a JSON blob of coefficients without
knowing the difference.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow.artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import Field

from dsio.contracts import DsioModel, sha256_of, sha256_of_bytes

# Mirrors `dsio.train.tracking.TRACKING_URI_ENV`/`DEFAULT_TRACKING_URI` exactly, and is
# duplicated rather than imported: `dsio.artifacts` is a foundation module (pyproject.toml's
# import-linter contract, "Foundation modules import no other dsio module") that
# `dsio.train.tracking` itself depends on, so importing the other way would be a cycle. A
# run has already resolved and proved this same URI reachable via `require_mlflow()` before
# any code here runs; this constant exists so a registry constructed with no argument agrees
# with that resolution rather than falling back to MLflow's own default (a local `./mlruns`
# directory), which would silently split a run's metrics and its model artifact across two
# unrelated backends.
TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
DEFAULT_TRACKING_URI = "http://localhost:5000"

# The artifact file name every saved payload is logged under, inside whichever MLflow run
# ends up hosting it.
ARTIFACT_FILE = "artifact.bin"

# I2: every payload used to be logged as bare `artifact.bin` at its host run's artifact
# root, so a second `save()` into the *same* run_id (`run_ssl_pretrain`'s `registry.
# save(..., run_id=...)` -- reachable today, latent only because exactly one save
# happens per run) overwrote the first version's bytes in place: `load` of the earlier
# version then raised `RegistryIntegrityError` on a digest mismatch it could no longer
# explain, because the bytes it needed were already gone. Content-addressing the upload
# path with the payload's own digest makes two different payloads land at two different
# paths -- collision-free -- and makes saving the *same* payload twice a no-op re-upload
# to the same path, not a destructive one. See `_artifact_subdir`.
_ARTIFACT_PREFIX = "dsio-models"

# `save()` without a `run_id` needs somewhere to host a throwaway run -- MLflow's model
# registry has no notion of a run-less version, so something has to hold one. This used
# to be MLflow's built-in Default experiment (id `"0"`, provisioned once at database
# init). I3: that id is not safe to trust -- its `artifact_location` on any stack created
# before `49cad22` is a bare, non-proxied path (`/artifacts`), which raises
# `PermissionError` the moment a client tries to write to it, and the Default experiment
# can never be recreated to pick up a later compose fix. A registry-owned, by-name
# experiment is created lazily instead (`_ensure_scratch_experiment`): as easy to
# recreate as any other experiment, and never inherits whatever the Default experiment's
# artifact_location happens to be.
_SCRATCH_EXPERIMENT_NAME = "dsio-model-registry-scratch"

# Tag keys the registry writes on every model version it creates. `dsio.` prefixed so they
# read unambiguously next to whatever tags MLflow itself or another tool adds.
_DIGEST_TAG = "dsio.digest"
_PROVENANCE_DIGEST_TAG = "dsio.provenance_digest"
_SIZE_BYTES_TAG = "dsio.size_bytes"
_CONFIG_HASH_TAG = "dsio.config_hash"
_CODE_HASH_TAG = "dsio.code_hash"
_DATA_SNAPSHOT_IDS_TAG = "dsio.data_snapshot_ids"
_SEED_TAG = "dsio.seed"
_METRICS_TAG = "dsio.metrics"


class RegistryIntegrityError(RuntimeError):
    """Raised when a model version's registry entry contradicts its stored bytes."""


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
    """A model version's provenance, decoded from MLflow's registry entry for it."""

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
def _no_progress_bar() -> Iterator[None]:
    """Suppress MLflow's tqdm upload/download bars, which are noise in test output and in
    a script's stdout. The same save/restore-one-variable shape as `dsio.train.tracking.
    _bounded_probe_timeout`, for the same reason: touch the environment for one call only.
    """
    from mlflow.environment_variables import MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR

    name = MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR.name
    saved = os.environ.get(name)
    os.environ[name] = "false"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


def _resolve_tracking_uri(tracking_uri: str | None) -> str:
    if tracking_uri is not None:
        return tracking_uri
    return os.environ.get(TRACKING_URI_ENV) or DEFAULT_TRACKING_URI


def _format_timestamp(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ModelRegistry:
    """A thin policy layer over MLflow's Model Registry.

    Storage, version numbering and listing are MLflow's; digest-on-save,
    verify-on-load and fail-closed-on-mismatch are dsio's, because MLflow does not do them.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = _resolve_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

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
        """Register ``payload`` as the next version of ``name`` and return its record.

        ``run_id``, when given, is the MLflow run this artifact belongs to (a training
        run's own id, so the model lands among that run's other artifacts). Without one,
        a throwaway run under MLflow's built-in Default experiment hosts it instead --
        MLflow's registry has no notion of a run-less model version, so something has to.
        """
        if not name or "'" in name:
            raise ValueError(f"invalid model name {name!r}")

        digest = sha256_of_bytes(payload)
        metrics = metrics or {}
        provenance_digest = compute_provenance_digest(
            digest=digest,
            config_hash=config_hash,
            code_hash=code_hash,
            data_snapshot_ids=data_snapshot_ids,
            seed=seed,
            metrics=metrics,
        )

        self._ensure_registered_model(name)

        owning_run_id = run_id
        scratch_run = owning_run_id is None
        if scratch_run:
            owning_run_id = self.client.create_run(
                experiment_id=self._ensure_scratch_experiment()
            ).info.run_id
        assert owning_run_id is not None, "either run_id was given, or a scratch run was just made"

        # I2: content-addressed, not a bare filename at the run's artifact root -- see
        # `_ARTIFACT_PREFIX`'s comment. Two different payloads land at two different
        # paths inside the same run; the same payload saved twice re-uploads to the same
        # path, which MLflow's artifact store already treats as an idempotent overwrite
        # of identical bytes, not data loss.
        artifact_subdir = f"{_ARTIFACT_PREFIX}/{name}/{digest[:16]}"
        with TemporaryDirectory() as scratch_dir, _no_progress_bar():
            local_path = Path(scratch_dir) / ARTIFACT_FILE
            local_path.write_bytes(payload)
            self.client.log_artifact(owning_run_id, str(local_path), artifact_path=artifact_subdir)

        if scratch_run:
            # Only a container for the bytes just logged; nothing about it should look
            # like a live or abandoned training run.
            self.client.set_terminated(owning_run_id, "FINISHED")

        tags = {
            _DIGEST_TAG: digest,
            _PROVENANCE_DIGEST_TAG: provenance_digest,
            _SIZE_BYTES_TAG: str(len(payload)),
        }
        if config_hash is not None:
            tags[_CONFIG_HASH_TAG] = config_hash
        if code_hash is not None:
            tags[_CODE_HASH_TAG] = code_hash
        if data_snapshot_ids:
            tags[_DATA_SNAPSHOT_IDS_TAG] = json.dumps(list(data_snapshot_ids))
        if seed is not None:
            tags[_SEED_TAG] = str(seed)
        if metrics:
            tags[_METRICS_TAG] = json.dumps(metrics, sort_keys=True)

        source = f"runs:/{owning_run_id}/{artifact_subdir}/{ARTIFACT_FILE}"
        model_version = self.client.create_model_version(
            name, source, run_id=owning_run_id, tags=tags
        )
        return self._to_model_version(model_version)

    def load(self, ref: ModelRef) -> bytes:
        """Return the artifact bytes for ``ref``, verifying its digest.

        Raises :class:`RegistryIntegrityError` if the registry does not know ``ref``, if
        its recorded digest disagrees with ``ref`` (a stale reference, or a registry entry
        that changed under it), if the artifact it points at is gone, or if the bytes on
        disk no longer hash to what the registry recorded (corruption).
        """
        try:
            model_version = self.client.get_model_version(ref.name, str(ref.version))
        except MlflowException as error:
            raise RegistryIntegrityError(
                f"{ref.name}:v{ref.version} is not in the registry"
            ) from error

        stored_digest = (model_version.tags or {}).get(_DIGEST_TAG)
        if stored_digest != ref.digest:
            shown = stored_digest[:12] if stored_digest else "<none>"
            raise RegistryIntegrityError(
                f"{ref.name}:v{ref.version} registry digest {shown} does not match "
                f"the requested {ref.digest[:12]}; the reference is stale or the "
                "registry entry changed"
            )

        with _no_progress_bar():
            try:
                local_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=model_version.source, tracking_uri=self.tracking_uri
                )
            except MlflowException as error:
                raise RegistryIntegrityError(
                    f"{ref.name}:v{ref.version} is registered but its artifact at "
                    f"{model_version.source} is missing"
                ) from error

        payload = Path(local_path).read_bytes()
        actual = sha256_of_bytes(payload)
        if actual != stored_digest:
            raise RegistryIntegrityError(
                f"{ref.name}:v{ref.version} artifact digest {actual[:12]} does not match "
                f"the registry's {stored_digest[:12]}; the artifact has been corrupted"
            )
        return payload

    def versions(self, name: str) -> list[ModelVersion]:
        """Return known versions of ``name``, oldest first. Empty if never registered.

        A projection of MLflow's own listing (``search_model_versions``) into dsio's typed
        shape -- not a second index. MLflow's registry is what actually enumerates and
        orders these; this only decodes the tags each version carries.
        """
        if not name or "'" in name:
            raise ValueError(f"invalid model name {name!r}")
        rows = self.client.search_model_versions(f"name='{name}'")
        return sorted((self._to_model_version(row) for row in rows), key=lambda v: v.version)

    def _ensure_registered_model(self, name: str) -> None:
        try:
            self.client.create_registered_model(name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise

    def _ensure_scratch_experiment(self) -> str:
        """Return the id of this registry's own scratch experiment, creating it once.

        I3: replaces trusting MLflow's built-in Default experiment (id ``"0"``) -- see
        ``_SCRATCH_EXPERIMENT_NAME``'s module-level comment for why that id is not safe
        on a stack whose Default experiment predates a ``--default-artifact-root`` fix.
        Looked up by name, not cached on ``self``, so a registry instance stays correct
        even if the experiment does not exist yet on first use.
        """
        experiment = self.client.get_experiment_by_name(_SCRATCH_EXPERIMENT_NAME)
        if experiment is not None:
            return experiment.experiment_id
        try:
            return self.client.create_experiment(_SCRATCH_EXPERIMENT_NAME)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            # Lost a create race to another process; it exists under this name now.
            experiment = self.client.get_experiment_by_name(_SCRATCH_EXPERIMENT_NAME)
            assert experiment is not None, "just failed to create it because it exists"
            return experiment.experiment_id

    def _to_model_version(self, model_version: Any) -> ModelVersion:
        tags: dict[str, str] = model_version.tags or {}
        return ModelVersion(
            name=model_version.name,
            version=int(model_version.version),
            digest=tags.get(_DIGEST_TAG, ""),
            created_at=_format_timestamp(model_version.creation_timestamp),
            size_bytes=int(tags.get(_SIZE_BYTES_TAG, 0)),
            run_id=model_version.run_id or None,
            config_hash=tags.get(_CONFIG_HASH_TAG),
            code_hash=tags.get(_CODE_HASH_TAG),
            data_snapshot_ids=(
                tuple(json.loads(tags[_DATA_SNAPSHOT_IDS_TAG]))
                if _DATA_SNAPSHOT_IDS_TAG in tags
                else ()
            ),
            seed=int(tags[_SEED_TAG]) if _SEED_TAG in tags else None,
            metrics=json.loads(tags[_METRICS_TAG]) if _METRICS_TAG in tags else {},
            provenance_digest=tags.get(_PROVENANCE_DIGEST_TAG),
        )


def promotion_blockers(record: Any) -> list[str]:
    """Return reasons ``record`` may not be promoted, empty if it may.

    This is where the clean-tree gate lives. Exploration is never blocked; promotion to a
    registered model is, because that artifact is the one that outlives the session.
    Independent of storage: this reads ``record.git``/``record.env`` (``dsio.runs.record.
    RunRecord``), not the registry.
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
