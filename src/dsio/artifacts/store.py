"""Digest-on-save, verify-on-load, fail-closed-on-mismatch, over MLflow's registry.

MLflow already does almost all of this. Its registry allocates immutable integer version
numbers and enforces their uniqueness in Postgres, stores the artifact, links it to a run,
and carries arbitrary tags. A ``models:/name/3`` reference is already pinned; only
*aliases* move, and nothing here uses aliases.

The one thing MLflow does not do is check that the bytes it hands back are the bytes that
were stored. ``ModelVersion`` has no checksum field, and the only integrity-shaped thing
in its artifact layer is ``verify_artifact_path``, which validates a path rather than
content. So MLflow will return a corrupted model without complaint — and returning the
wrong weights is worse than returning nothing, because training continues and the numbers
look plausible.

That gap is this module: a sha256 written as a tag on save, compared on load, raising on
mismatch. Everything else is MLflow's, and is left to MLflow rather than mirrored here.

Two scars worth keeping, because both are bugs this already hit:

**Artifact paths are content-addressed** (``_ARTIFACT_PREFIX``). Payloads used to be logged
as a bare ``artifact.bin`` at the host run's artifact root, so a second ``save()`` into the
same run overwrote the first version's bytes in place — and ``load`` of the earlier version
then failed a digest check it could no longer explain, because the bytes were gone. Keying
the path by the payload's own digest makes two payloads land in two places, and the same
payload saved twice an idempotent re-upload.

**The scratch experiment is registry-owned** (``_SCRATCH_EXPERIMENT_NAME``). A run-less save
needs a run to host it, and MLflow's built-in Default experiment (id ``"0"``) is not safe to
borrow: on any stack created before the compose fix its ``artifact_location`` is a bare
non-proxied path that raises ``PermissionError`` on write, and it can never be recreated.

The registry stores raw bytes and never imports a modelling framework, so it holds a pickled
sklearn pipeline, a torch state dict or a JSON blob of coefficients without knowing which.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow.artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from dsio.contracts import DsioModel, sha256_of_bytes

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

ARTIFACT_FILE = "artifact.bin"
_ARTIFACT_PREFIX = "dsio-models"
_SCRATCH_EXPERIMENT_NAME = "dsio-model-registry-scratch"

#: The one tag this module owns. Everything else a caller wants recorded goes through
#: `save(tags=...)` and is MLflow's to store, because MLflow tags are already the storage —
#: a typed mirror of them here would be a second copy of a schema MLflow defines.
_DIGEST_TAG = "dsio.digest"


class RegistryIntegrityError(RuntimeError):
    """Raised when a model version's registry entry contradicts its stored bytes."""


class ModelRef(DsioModel):
    """The handle a saved model is referred to by: name, version, and content digest.

    Name and version alone would be enough to *find* the artifact — MLflow pins those. The
    digest is what makes finding it and trusting it the same operation, and it is the field
    that survives serialisation into a run's ``encoder.json`` and back.
    """

    name: str
    version: int
    digest: str

    def __str__(self) -> str:
        return f"{self.name}:v{self.version}@{self.digest[:12]}"


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


class ModelRegistry:
    """Integrity policy over MLflow's Model Registry.

    Storage, version numbering, listing and run linkage are MLflow's. Digest-on-save and
    verify-on-load are this class's, because MLflow does not do them.
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
        tags: Mapping[str, str] | None = None,
    ) -> ModelRef:
        """Register ``payload`` as the next version of ``name`` and return its reference.

        ``run_id``, when given, is the MLflow run this artifact belongs to, so the model
        lands among that run's other artifacts. Without one, a throwaway run under this
        registry's own scratch experiment hosts it — MLflow's registry has no notion of a
        run-less model version, so something has to.

        ``tags`` are recorded verbatim on the model version. Provenance a caller wants kept
        (a config hash, a data snapshot id, a seed) belongs here: MLflow tags are where it
        would be stored anyway, and a typed wrapper would only restate MLflow's schema.
        """
        if not name or "'" in name:
            raise ValueError(f"invalid model name {name!r}")

        digest = sha256_of_bytes(payload)
        self._ensure_registered_model(name)

        owning_run_id = run_id
        scratch_run = owning_run_id is None
        if scratch_run:
            owning_run_id = self.client.create_run(
                experiment_id=self._ensure_scratch_experiment()
            ).info.run_id
        assert owning_run_id is not None, "either run_id was given, or a scratch run was just made"

        artifact_subdir = f"{_ARTIFACT_PREFIX}/{name}/{digest[:16]}"
        with TemporaryDirectory() as scratch_dir, _no_progress_bar():
            local_path = Path(scratch_dir) / ARTIFACT_FILE
            local_path.write_bytes(payload)
            self.client.log_artifact(owning_run_id, str(local_path), artifact_path=artifact_subdir)

        if scratch_run:
            # Only a container for the bytes just logged; nothing about it should look
            # like a live or abandoned training run.
            self.client.set_terminated(owning_run_id, "FINISHED")

        source = f"runs:/{owning_run_id}/{artifact_subdir}/{ARTIFACT_FILE}"
        model_version = self.client.create_model_version(
            name, source, run_id=owning_run_id, tags={**(tags or {}), _DIGEST_TAG: digest}
        )
        return ModelRef(name=name, version=int(model_version.version), digest=digest)

    def load(self, ref: ModelRef) -> bytes:
        """Return the artifact bytes for ``ref``, verifying its digest.

        Raises :class:`RegistryIntegrityError` if the registry does not know ``ref``, if its
        recorded digest disagrees with ``ref`` (a stale reference, or a registry entry that
        changed under it), if the artifact it points at is gone, or if the bytes no longer
        hash to what the registry recorded (corruption). Four ways to be wrong, and every
        one of them returns nothing rather than the wrong weights.
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

    def versions(self, name: str) -> list[Any]:
        """MLflow's own model versions for ``name``, oldest first. Empty if never registered.

        Returns MLflow's entities rather than a dsio type. This class is a thin policy layer
        over MLflow's registry, and re-describing ``ModelVersion``'s fields here would be a
        second copy of a schema that already exists — the exact parallel mechanism ADR 0002
        was written against.
        """
        if not name or "'" in name:
            raise ValueError(f"invalid model name {name!r}")
        rows = self.client.search_model_versions(f"name='{name}'")
        return sorted(rows, key=lambda row: int(row.version))

    def ref_for(self, name: str, version: int) -> ModelRef:
        """The reference for an already-registered version, digest read from the registry.

        For a caller that knows a name and a version but not the digest — which is the
        position anything reading MLflow's listing is in.
        """
        try:
            row = self.client.get_model_version(name, str(version))
        except MlflowException as error:
            raise RegistryIntegrityError(f"{name}:v{version} is not in the registry") from error
        digest = (row.tags or {}).get(_DIGEST_TAG)
        if digest is None:
            raise RegistryIntegrityError(
                f"{name}:v{version} carries no {_DIGEST_TAG} tag; it was not saved by dsio"
            )
        return ModelRef(name=name, version=version, digest=digest)

    def _ensure_registered_model(self, name: str) -> None:
        try:
            self.client.create_registered_model(name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise

    def _ensure_scratch_experiment(self) -> str:
        """Return the id of this registry's own scratch experiment, creating it once.

        Replaces trusting MLflow's built-in Default experiment (id ``"0"``) -- see
        ``_SCRATCH_EXPERIMENT_NAME``'s module-level note for why that id is not safe on a
        stack whose Default experiment predates a ``--default-artifact-root`` fix. Looked up
        by name, not cached on ``self``, so a registry instance stays correct even if the
        experiment does not exist yet on first use.
        """
        experiment = self.client.get_experiment_by_name(_SCRATCH_EXPERIMENT_NAME)
        if experiment is not None:
            return experiment.experiment_id
        try:
            return self.client.create_experiment(_SCRATCH_EXPERIMENT_NAME)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            again = self.client.get_experiment_by_name(_SCRATCH_EXPERIMENT_NAME)
            assert again is not None, "RESOURCE_ALREADY_EXISTS means it is there"
            return again.experiment_id
