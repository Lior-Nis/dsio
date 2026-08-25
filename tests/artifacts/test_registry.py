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
    ModelRef,
    ModelRegistry,
    RegistryIntegrityError,
    compute_provenance_digest,
    promotion_blockers,
)
from dsio.runs.provenance import EnvState, GitState


def _artifact_path(registry: ModelRegistry, run_id: str, filename: str = "artifact.bin") -> Path:
    """Resolve the real on-disk file MLflow's `file:` backend wrote for ``run_id``.

    Not part of `ModelRegistry`'s own API -- deliberately: exposing "give me the
    underlying path" would leak the abstraction this layer exists to hide. A test that
    wants to simulate disk corruption reaches for it the same way any external tool
    poking at the `file:` store would, through MLflow's own client.
    """
    artifact_uri = registry.client.get_run(run_id).info.artifact_uri
    return Path(urlparse(artifact_uri).path) / filename


def test_versions_increment_and_pin() -> None:
    """MLflow's registry allocates and orders versions; dsio no longer needs its own
    manifest or lock to do it. Two saves to the same name land at v1, v2, in order."""
    registry = ModelRegistry()
    first = registry.save("m", b"one")
    second = registry.save("m", b"two")
    assert (first.version, second.version) == (1, 2)
    assert registry.load(first.ref) == b"one"
    assert registry.load(second.ref) == b"two"


def test_an_uncorrupted_model_loads() -> None:
    """The acceptance half of every refusal test below: nothing here should ever make a
    clean save/load round-trip fail."""
    registry = ModelRegistry()
    version = registry.save("m", b"payload", seed=7, config_hash="cfg")
    assert registry.load(version.ref) == b"payload"


def test_a_corrupted_artifact_fails_closed() -> None:
    """Spec Verification item 6. Returning the wrong weights is worse than returning
    nothing, because training continues and the numbers look plausible."""
    registry = ModelRegistry()
    version = registry.save("m", b"payload")
    assert version.run_id is not None
    path = _artifact_path(registry, version.run_id)
    assert path.is_file()
    original = path.read_bytes()
    path.write_bytes(b"tampered")
    try:
        with pytest.raises(RegistryIntegrityError, match="corrupted"):
            registry.load(version.ref)
    finally:
        path.write_bytes(original)
    # Restoring the bytes MLflow stored is enough to prove the failure was about the
    # bytes, not about `version.ref` itself: the same reference now loads clean again.
    assert registry.load(version.ref) == b"payload"


def test_a_missing_artifact_fails_closed() -> None:
    registry = ModelRegistry()
    version = registry.save("m", b"payload")
    assert version.run_id is not None
    _artifact_path(registry, version.run_id).unlink()
    with pytest.raises(RegistryIntegrityError, match="missing"):
        registry.load(version.ref)


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


def test_provenance_digest_separates_identical_weights() -> None:
    """Same bytes, different training data, must not be the same model."""
    common = {"digest": "abc", "code_hash": "c0de", "seed": 1, "metrics": {}}
    left = compute_provenance_digest(config_hash="x", data_snapshot_ids=("a",), **common)
    right = compute_provenance_digest(config_hash="x", data_snapshot_ids=("b",), **common)
    assert left != right


def test_provenance_digest_is_order_independent() -> None:
    common = {"digest": "abc", "config_hash": "x", "code_hash": "c0de", "seed": 1, "metrics": {}}
    left = compute_provenance_digest(data_snapshot_ids=("a", "b"), **common)
    right = compute_provenance_digest(data_snapshot_ids=("b", "a"), **common)
    assert left == right


def test_saved_provenance_round_trips_through_the_registry() -> None:
    """`save`'s lineage arguments must actually reach the registry entry `versions()`
    decodes back -- the pairing that would go silently wrong if a tag were dropped."""
    registry = ModelRegistry()
    version = registry.save(
        "m",
        b"payload",
        run_id=None,
        config_hash="cfg-1",
        code_hash="code-1",
        data_snapshot_ids=("snap-b", "snap-a"),
        seed=3,
    )
    (read_back,) = [v for v in registry.versions("m") if v.version == version.version]
    assert read_back.config_hash == "cfg-1"
    assert read_back.code_hash == "code-1"
    assert read_back.data_snapshot_ids == ("snap-b", "snap-a")
    assert read_back.seed == 3
    assert read_back.provenance_digest == version.provenance_digest
    assert read_back.size_bytes == len(b"payload")


def test_promotion_requires_a_clean_tree() -> None:
    """The one gate in the system. Exploration is free; promotion is not."""

    class Record:
        git = GitState(sha="abc", dirty=True, code_hash="abc-dirty-xyz")
        env = EnvState(python="3.12", platform="linux", hostname="h", lock_sha256="l")

    blockers = promotion_blockers(Record())
    assert any("dirty" in reason for reason in blockers)


def test_promotion_requires_provenance() -> None:
    class Record:
        git = GitState()
        env = EnvState(python="3.12", platform="linux", hostname="h")

    blockers = promotion_blockers(Record())
    assert any("git" in reason for reason in blockers)
    assert any("lockfile" in reason for reason in blockers)


def test_clean_run_may_be_promoted() -> None:
    class Record:
        git = GitState(sha="abc", dirty=False, code_hash="abc")
        env = EnvState(python="3.12", platform="linux", hostname="h", lock_sha256="l")

    assert promotion_blockers(Record()) == []
