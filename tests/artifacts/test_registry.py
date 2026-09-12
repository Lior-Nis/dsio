"""Registry integrity invariants, fake-backed against MLflow's own `file:` tracking URI
(`tests/conftest.py`'s autouse isolation) -- real MLflow client code, no server, no Docker.

Every corruption case below is paired with the acceptance it would otherwise make trivial
to fake: a registry that fails closed on everything passes every corruption test and is
useless, so `test_versions_increment_and_pin` and `test_an_uncorrupted_model_loads` prove
an honest save/load round-trip still works.

`test_a_corrupted_artifact_fails_closed` is spec Verification item 6: corrupt a stored
model, confirm load raises on digest mismatch rather than returning weights. It corrupts
the real bytes MLflow's file-store backend wrote to disk -- found the same way a caller
would, through `MlflowClient.get_run(...).info.artifact_uri` -- not a mock and not
something dsio's own registry code controls, so this proves the fail-closed property
survives the move to MLflow's storage rather than merely asserting dsio's wrapper code
still runs.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pytest

from dsio.artifacts.store import (
    _ARTIFACT_PREFIX,
    _DIGEST_TAG,
    ARTIFACT_FILE,
    ModelRef,
    ModelRegistry,
    RegistryIntegrityError,
)


def _artifact_path(registry: ModelRegistry, ref: ModelRef) -> Path:
    """Resolve the real on-disk file MLflow's `file:` backend wrote for ``version``.

    Not part of `ModelRegistry`'s own API -- deliberately: exposing "give me the
    underlying path" would leak the abstraction this layer exists to hide. A test that
    wants to simulate disk corruption reaches for it the same way any external tool
    poking at the `file:` store would, through MLflow's own client.

    Takes the whole `ModelRef`, not just a bare `run_id`: I2 content-addresses the upload
    (`dsio-models/<name>/<digest[:16]>/artifact.bin`, not a bare `artifact.bin` at the
    run's artifact root), so finding the file back needs the name and digest too. The
    hosting run is MLflow's to know, so it is looked up rather than carried around.
    """
    run_id = registry.client.get_model_version(ref.name, str(ref.version)).run_id
    assert run_id is not None
    artifact_uri = registry.client.get_run(run_id).info.artifact_uri
    return (
        Path(urlparse(artifact_uri).path)
        / _ARTIFACT_PREFIX
        / ref.name
        / ref.digest[:16]
        / ARTIFACT_FILE
    )


def test_versions_increment_and_pin() -> None:
    """MLflow's registry allocates and orders versions; dsio no longer needs its own
    manifest or lock to do it. Two saves to the same name land at v1, v2, in order."""
    registry = ModelRegistry()
    first = registry.save("m", b"one")
    second = registry.save("m", b"two")
    assert (first.version, second.version) == (1, 2)
    assert registry.load(first) == b"one"
    assert registry.load(second) == b"two"


def test_an_uncorrupted_model_loads() -> None:
    """The acceptance half of every refusal test below: nothing here should ever make a
    clean save/load round-trip fail."""
    registry = ModelRegistry()
    ref = registry.save("m", b"payload", tags={"dsio.seed": "7"})
    assert registry.load(ref) == b"payload"


def test_two_saves_into_one_run_do_not_destroy_each_other() -> None:
    """I2: every payload used to be uploaded as a bare ``artifact.bin`` at its host run's
    artifact root, so a second ``save()`` into the *same* ``run_id`` overwrote the first
    version's bytes in place -- reachable today through ``run_ssl_pretrain``'s own
    ``registry.save(..., run_id=...)`` call, latent only because exactly one save
    happens per run currently. Loading the earlier version afterward raised
    ``RegistryIntegrityError`` on a digest mismatch it could no longer explain, because
    the bytes it needed were already gone -- possibly weeks after the second save. Two
    real saves into one real shared run, both loaded back, is the actual property that
    matters, not just that the upload path string looks content-addressed.
    """
    registry = ModelRegistry()
    experiment_id = registry.client.create_experiment("shared-run-for-two-saves")
    run_id = registry.client.create_run(experiment_id=experiment_id).info.run_id

    first = registry.save("m", b"first payload", run_id=run_id)
    second = registry.save("m", b"second payload", run_id=run_id)

    assert first.version != second.version
    assert registry.load(first) == b"first payload"
    assert registry.load(second) == b"second payload"


def test_scratchless_save_does_not_use_the_default_experiment() -> None:
    """I3: ``save()`` without a ``run_id`` used to hardcode MLflow's built-in Default
    experiment (id ``"0"``, provisioned once at database init and never recreated) to
    host its throwaway run. On any stack whose Default experiment predates a
    ``--default-artifact-root`` fix, that id resolves a bare, non-proxied
    ``artifact_location`` and every run-less save raises ``PermissionError`` --
    permanently, because experiment ``"0"`` cannot be recreated. This suite's `file:`
    backend cannot reproduce that permission failure directly (every experiment's
    location works there), so what it can and must prove is the actual code-level fix:
    ``save()`` resolves its own named experiment, not id ``"0"``, and reuses that same
    experiment on repeated calls rather than creating a fresh one each time.
    ``tests/train/test_tracking.py``'s live suite is the sibling that exercises the real
    permission failure against the compose stack.
    """
    registry = ModelRegistry()
    ref = registry.save("m", b"payload")
    run_id = registry.client.get_model_version(ref.name, str(ref.version)).run_id
    assert run_id is not None

    hosting_run = registry.client.get_run(run_id)
    assert hosting_run.info.experiment_id != "0"

    registry.save("m", b"another payload")
    matches = [
        experiment
        for experiment in registry.client.search_experiments()
        if experiment.experiment_id == hosting_run.info.experiment_id
    ]
    assert len(matches) == 1, "repeated run-less saves must reuse one scratch experiment"


def test_a_corrupted_artifact_fails_closed() -> None:
    """Spec Verification item 6. Returning the wrong weights is worse than returning
    nothing, because training continues and the numbers look plausible."""
    registry = ModelRegistry()
    ref = registry.save("m", b"payload")
    path = _artifact_path(registry, ref)
    assert path.is_file()
    original = path.read_bytes()
    path.write_bytes(b"tampered")
    try:
        with pytest.raises(RegistryIntegrityError, match="corrupted"):
            registry.load(ref)
    finally:
        path.write_bytes(original)
    # Restoring the bytes MLflow stored is enough to prove the failure was about the
    # bytes, not about `ref` itself: the same reference now loads clean again.
    assert registry.load(ref) == b"payload"


def test_a_missing_artifact_fails_closed() -> None:
    registry = ModelRegistry()
    ref = registry.save("m", b"payload")
    _artifact_path(registry, ref).unlink()
    with pytest.raises(RegistryIntegrityError, match="missing"):
        registry.load(ref)


def test_stale_reference_fails_closed() -> None:
    """A reference naming a real version but the wrong digest -- a stale pin, or a
    swapped artifact -- must be refused before any bytes are even read back."""
    registry = ModelRegistry()
    registry.save("m", b"payload")
    with pytest.raises(RegistryIntegrityError, match="does not match"):
        registry.load(ModelRef(name="m", version=1, digest="0" * 64))


def test_unknown_version_fails_closed() -> None:
    registry = ModelRegistry()
    registry.save("m", b"payload")
    with pytest.raises(RegistryIntegrityError, match="not in the registry"):
        registry.load(ModelRef(name="m", version=99, digest="0" * 64))


def test_tags_reach_the_registry_entry() -> None:
    """Provenance is stored as MLflow tags, so the property worth testing is that they
    arrive -- the pairing that would go silently wrong if `save` dropped them."""
    registry = ModelRegistry()
    ref = registry.save("m", b"payload", tags={"dsio.config_hash": "cfg-1", "k": "v"})

    stored = registry.client.get_model_version("m", str(ref.version)).tags
    assert stored["dsio.config_hash"] == "cfg-1"
    assert stored["k"] == "v"
    assert stored[_DIGEST_TAG] == ref.digest, "the digest tag is the one this module owns"


def test_a_caller_supplied_tag_cannot_forge_the_digest() -> None:
    """`save` writes the digest tag last, so a caller passing one cannot displace it --
    otherwise a wrong digest in the tags would defeat every fail-closed check in `load`."""
    registry = ModelRegistry()
    ref = registry.save("m", b"payload", tags={_DIGEST_TAG: "0" * 64})

    assert ref.digest != "0" * 64
    assert registry.load(ref) == b"payload"


def test_ref_for_recovers_a_reference_from_name_and_version() -> None:
    """The position anything reading MLflow's own listing is in: it knows a name and a
    version, and needs the digest to load safely."""
    registry = ModelRegistry()
    saved = registry.save("m", b"payload")

    assert registry.ref_for("m", saved.version) == saved
    with pytest.raises(RegistryIntegrityError, match="not in the registry"):
        registry.ref_for("m", 99)


# --- live: the real compose stack -----------------------------------------------------


@pytest.mark.live
def test_live_registry_round_trip_with_and_without_a_run_id() -> None:
    """I3, at the only level that is honest about what broke: this suite's `file:`
    backend cannot reproduce a `PermissionError` from a stale Default-experiment
    `artifact_location` (every experiment's location resolves fine there), and no
    existing test saved without a `run_id` against the live, Postgres-backed stack --
    the registry tests are all `file:`-backed, and the one other live artifact test
    (`tests/train/test_tracking.py`) creates a fresh experiment every run rather than
    going through `ModelRegistry.save`'s run-less, scratch-experiment path at all. Needs
    `docker compose up -d` (compose.yaml); excluded by default, run with
    `uv run --extra cpu pytest -m live`.
    """
    import time

    registry = ModelRegistry(tracking_uri="http://localhost:5000")
    name = f"live-registry-roundtrip-{time.time_ns()}"

    # Without a run_id: exercises `_ensure_scratch_experiment` end to end -- the exact
    # path that used to resolve MLflow's built-in Default experiment (id "0") and, on a
    # stack whose Default experiment predates the `--default-artifact-root` fix, raised
    # `PermissionError` before a single byte was written.
    scratchless = registry.save(name, b"scratchless payload")
    assert registry.load(scratchless) == b"scratchless payload"

    # With a run_id: the ordinary path a training run uses, on the same live stack.
    experiment_id = registry.client.create_experiment(f"{name}-owning")
    run_id = registry.client.create_run(experiment_id=experiment_id).info.run_id
    with_run = registry.save(name, b"owned-run payload", run_id=run_id)
    assert registry.load(with_run) == b"owned-run payload"

    assert scratchless.version != with_run.version
