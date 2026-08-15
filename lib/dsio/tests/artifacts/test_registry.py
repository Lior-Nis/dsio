"""Registry integrity invariants. Every one of these must fail closed."""

from __future__ import annotations

import pytest

from dsio.artifacts import (
    ModelRef,
    ModelRegistry,
    RegistryIntegrityError,
    compute_provenance_digest,
    promotion_blockers,
)
from dsio.runs import EnvState, GitState


def test_versions_increment_and_pin() -> None:
    registry = ModelRegistry()
    first = registry.save("m", b"one")
    second = registry.save("m", b"two")
    assert (first.version, second.version) == (1, 2)
    assert registry.load(first.ref) == b"one"
    assert registry.load(second.ref) == b"two"


def test_corrupt_artifact_fails_closed() -> None:
    """Returning the wrong weights is worse than returning nothing."""
    registry = ModelRegistry()
    version = registry.save("m", b"payload")
    path = registry.root / "m" / "v1" / "artifact.bin"
    path.write_bytes(b"tampered")
    with pytest.raises(RegistryIntegrityError, match="corrupted"):
        registry.load(version.ref)


def test_stale_reference_fails_closed() -> None:
    registry = ModelRegistry()
    registry.save("m", b"payload")
    with pytest.raises(RegistryIntegrityError, match="does not match"):
        registry.load(ModelRef(name="m", version=1, digest="0" * 64))


def test_missing_artifact_fails_closed() -> None:
    registry = ModelRegistry()
    version = registry.save("m", b"payload")
    (registry.root / "m" / "v1" / "artifact.bin").unlink()
    with pytest.raises(RegistryIntegrityError, match="missing"):
        registry.load(version.ref)


def test_duplicate_manifest_version_fails_closed() -> None:
    """Two writers allocating one number leaves an artifact unreachable."""
    registry = ModelRegistry()
    version = registry.save("m", b"payload")
    manifest = registry.root / "m" / "manifest.jsonl"
    manifest.write_text(manifest.read_text() + manifest.read_text())
    with pytest.raises(RegistryIntegrityError, match="more than once"):
        registry.versions("m")
    assert version.version == 1


def test_orphaned_directory_still_reserves_its_version() -> None:
    """A crash between mkdir and manifest append must not let the number be reused."""
    registry = ModelRegistry()
    registry.save("m", b"one")
    (registry.root / "m" / "v7").mkdir(parents=True)
    assert registry.save("m", b"two").version == 8


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
