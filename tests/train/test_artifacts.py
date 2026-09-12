"""The encoder handoff: an artifact round-trips, or it refuses.

Fake-backed against MLflow's own `file:` tracking URI (`tests/conftest.py`'s autouse
isolation) -- real MLflow client code, no server, no Docker.

Every refusal below is paired with the acceptance that would otherwise make it trivial to
fake: a loader that raises on everything passes every corruption test and is useless, so
`test_a_clean_artifact_round_trips` is what stops that.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import pytest

pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient  # noqa: E402

from dsio.train.artifacts import (  # noqa: E402
    ArtifactIntegrityError,
    ArtifactRef,
    load_artifact,
    resolve_tracking_uri,
    save_artifact,
)


def _client() -> MlflowClient:
    """A client resolved exactly as `dsio.train.artifacts` resolves one.

    A bare `MlflowClient()` honours `mlflow.set_tracking_uri()`, which another test in the
    suite sets globally; the module reads `MLFLOW_TRACKING_URI` instead. Using different
    resolutions here wrote the run to one store and looked for it in the other -- passing
    in isolation, failing in the suite with "Run not found".
    """
    return MlflowClient(tracking_uri=resolve_tracking_uri())


def _run_id() -> str:
    """A fresh run to hang artifacts on.

    `uuid4`, not `id(...)`: CPython recycles object ids after collection, so an id-derived
    experiment name collides with an earlier test's the moment the first client is freed --
    which passes in isolation and fails in the suite.
    """
    client = _client()
    experiment_id = client.create_experiment(f"artifacts-{uuid4()}")
    return client.create_run(experiment_id=experiment_id).info.run_id


def _on_disk(ref: ArtifactRef) -> Path:
    """The real file MLflow's `file:` backend wrote, found the way any external tool would.

    Deliberately not something `dsio.train.artifacts` exposes: "give me the underlying
    path" would leak the abstraction the digest check exists to make unnecessary.
    """
    artifact_uri = _client().get_run(ref.run_id).info.artifact_uri
    return Path(urlparse(artifact_uri).path) / ref.path


def test_a_clean_artifact_round_trips() -> None:
    ref = save_artifact(b"payload", run_id=_run_id(), name="enc")
    assert load_artifact(ref) == b"payload"


def test_the_reference_carries_the_digest_of_what_was_saved() -> None:
    """Identity is the content, not a name or a version that something else could rebind."""
    from dsio.contracts import sha256_of_bytes

    ref = save_artifact(b"payload", run_id=_run_id(), name="enc")
    assert ref.digest == sha256_of_bytes(b"payload")


def test_two_saves_into_one_run_do_not_destroy_each_other() -> None:
    """Content-addressed paths. A payload logged under a bare filename at the run's
    artifact root is overwritten in place by the next save, and the first reference then
    fails a digest check it cannot explain because the bytes are gone -- possibly weeks
    later. Two real saves into one real run, both loaded back, is the property that
    matters, not that the path string looks content-addressed.
    """
    run_id = _run_id()
    first = save_artifact(b"first payload", run_id=run_id, name="enc")
    second = save_artifact(b"second payload", run_id=run_id, name="enc")

    assert first.path != second.path
    assert load_artifact(first) == b"first payload"
    assert load_artifact(second) == b"second payload"


def test_a_corrupted_artifact_fails_closed() -> None:
    """Returning the wrong weights is worse than returning nothing: training continues and
    the numbers look plausible."""
    ref = save_artifact(b"payload", run_id=_run_id(), name="enc")
    path = _on_disk(ref)
    assert path.is_file()
    original = path.read_bytes()

    path.write_bytes(b"tampered")
    try:
        with pytest.raises(ArtifactIntegrityError, match="corrupted"):
            load_artifact(ref)
    finally:
        path.write_bytes(original)
    # Restoring the bytes proves the refusal was about them and not about `ref` itself.
    assert load_artifact(ref) == b"payload"


def test_a_missing_artifact_fails_closed() -> None:
    ref = save_artifact(b"payload", run_id=_run_id(), name="enc")
    _on_disk(ref).unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        load_artifact(ref)


def test_an_unknown_run_fails_closed() -> None:
    ref = ArtifactRef(run_id="nosuchrun", path="dsio-artifacts/enc/x/artifact.bin", digest="0" * 64)
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        load_artifact(ref)


def test_a_stale_digest_fails_closed() -> None:
    """A reference to real bytes with the wrong digest -- a stale pin, or a swapped
    artifact -- is refused rather than loaded."""
    saved = save_artifact(b"payload", run_id=_run_id(), name="enc")
    stale = ArtifactRef(run_id=saved.run_id, path=saved.path, digest="0" * 64)
    with pytest.raises(ArtifactIntegrityError, match="corrupted or replaced"):
        load_artifact(stale)


def test_an_invalid_name_is_refused_before_anything_is_written() -> None:
    with pytest.raises(ValueError, match="invalid artifact name"):
        save_artifact(b"payload", run_id=_run_id(), name="has/slash")


@pytest.mark.live
def test_live_round_trip_against_the_compose_stack() -> None:
    """The `file:` backend cannot reproduce a proxied-artifact-store failure. Needs
    `docker compose up -d`; run with `uv run --extra cpu pytest -m live`."""
    import time

    client = MlflowClient(tracking_uri="http://localhost:5000")
    experiment_id = client.create_experiment(f"live-artifact-{time.time_ns()}")
    run_id = client.create_run(experiment_id=experiment_id).info.run_id

    ref = save_artifact(b"live payload", run_id=run_id, name="enc",
                        tracking_uri="http://localhost:5000")
    assert load_artifact(ref, tracking_uri="http://localhost:5000") == b"live payload"
