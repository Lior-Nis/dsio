"""The encoder handoff between two runs: an artifact, a digest, and a refusal.

Pretraining produces an encoder that a later run loads as its backbone. That is a transfer
*inside* a two-stage training pipeline, not a deployment, and the distinction decides how
it should be stored.

**Not MLflow's Model Registry.** The registry is a promotion-time mechanism: named models,
integer versions, and aliases like ``@champion`` that let serving point at "the current
model" without a redeploy. None of that applies to an intermediate artifact that only the
next run reads. Registering one at save time spends a deployment concept on a training
detail — and costs a registered model, a version row, and (for a run-less save) a scratch
experiment to host it. ``mlflow.register_model(uri, name)`` promotes an existing artifact
in one call, so nothing here forecloses the registry; it defers it to promotion, which is
where ADR 0003 already says the gate belongs.

**But still a digest.** MLflow does not verify what it hands back: ``ModelVersion`` has no
checksum field and the artifact layer's only integrity-shaped function validates a path,
not content. So a corrupted encoder loads silently, training continues, and the numbers
look plausible. A sha256 recorded at save and compared at load is the whole point of this
module.

Dropping the registry also removes a way to be wrong. The registry stored the digest a
*second* time, as a tag, so a load had to reconcile three things: the reference, the tag,
and the bytes. Two of those could disagree with each other in ways no caller could act on.
Now there is the reference and there are the bytes.

Artifact paths stay content-addressed. Two saves into one run must not collide: a payload
logged under a bare filename at the run's artifact root is overwritten in place by the
next one, and the first reference then fails a digest check it cannot explain because the
bytes are gone.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow.artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from dsio.contracts import DsioModel, sha256_of_bytes

# Mirrors `dsio.train.tracking.TRACKING_URI_ENV`/`DEFAULT_TRACKING_URI`. A run has already
# resolved and proved this URI reachable via `require_mlflow()` before anything here runs;
# this exists so a call with no explicit URI agrees with that resolution rather than
# falling back to MLflow's own default (a local `./mlruns` directory), which would split a
# run's metrics and its encoder across two unrelated backends.
TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
DEFAULT_TRACKING_URI = "http://localhost:5000"

ARTIFACT_FILE = "artifact.bin"
_ARTIFACT_PREFIX = "dsio-artifacts"


class ArtifactIntegrityError(RuntimeError):
    """Raised when a stored artifact is missing, or is not the bytes that were saved."""


class ArtifactRef(DsioModel):
    """Everything needed to fetch one artifact back and prove it is the right one.

    ``run_id`` and ``path`` locate it; ``digest`` is what makes locating it and trusting it
    the same operation. There is deliberately nothing here that can drift — no name to
    re-resolve, no version to bump, no alias to move.
    """

    run_id: str
    path: str
    digest: str

    @property
    def uri(self) -> str:
        return f"runs:/{self.run_id}/{self.path}"

    def __str__(self) -> str:
        return f"{self.path}@{self.digest[:12]}"


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    if tracking_uri is not None:
        return tracking_uri
    return os.environ.get(TRACKING_URI_ENV) or DEFAULT_TRACKING_URI


@contextmanager
def _no_progress_bar() -> Iterator[None]:
    """Suppress MLflow's tqdm upload/download bars, which are noise in test output and in
    a script's stdout. Touches the environment for one call only, then restores it.
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


def save_artifact(
    payload: bytes, *, run_id: str, name: str, tracking_uri: str | None = None
) -> ArtifactRef:
    """Log ``payload`` under ``run_id`` and return the reference that fetches it back.

    ``name`` groups artifacts within the run and appears in the path, so a run that saves
    more than one kind of thing stays readable. The digest, not the name, is what identity
    rests on.
    """
    if not name or "/" in name:
        raise ValueError(f"invalid artifact name {name!r}")

    digest = sha256_of_bytes(payload)
    subdir = f"{_ARTIFACT_PREFIX}/{name}/{digest[:16]}"
    client = MlflowClient(tracking_uri=resolve_tracking_uri(tracking_uri))

    with TemporaryDirectory() as scratch, _no_progress_bar():
        local = Path(scratch) / ARTIFACT_FILE
        local.write_bytes(payload)
        client.log_artifact(run_id, str(local), artifact_path=subdir)

    return ArtifactRef(run_id=run_id, path=f"{subdir}/{ARTIFACT_FILE}", digest=digest)


def load_artifact(ref: ArtifactRef, *, tracking_uri: str | None = None) -> bytes:
    """Return the bytes ``ref`` names, or raise rather than return the wrong ones.

    Two ways to fail, both closed: the artifact is gone, or what came back does not hash
    to what was saved. A corrupted encoder that loads quietly is worse than one that does
    not load, because training continues on top of it and the numbers look plausible.
    """
    with _no_progress_bar():
        try:
            local = mlflow.artifacts.download_artifacts(
                artifact_uri=ref.uri, tracking_uri=resolve_tracking_uri(tracking_uri)
            )
        except (MlflowException, OSError) as error:
            raise ArtifactIntegrityError(f"{ref} is missing at {ref.uri}") from error

    payload = Path(local).read_bytes()
    actual = sha256_of_bytes(payload)
    if actual != ref.digest:
        raise ArtifactIntegrityError(
            f"{ref} hashes to digest {actual[:12]} but was saved with digest "
            f"{ref.digest[:12]}; the artifact has been corrupted or replaced"
        )
    return payload
