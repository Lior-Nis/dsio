"""Remote invariants, including the fresh-clone test the whole design is aimed at.

Exercised against a local filesystem through fsspec rather than a mock. A mock would prove
the calls were made in the right order; this proves the bytes arrive and verify, which is
the only claim worth making.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from dsio.data import SignalStore
from dsio.data.remote import (
    REMOTE_ENV,
    RemoteError,
    RemoteIntegrityError,
    object_key,
    pull,
    push,
    resolve_remote,
    status,
    write_remotes,
)
from dsio.data.store import MANIFEST_FILE, SIGNAL_FILE

pytest.importorskip("fsspec")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A store under ``stores/``, and an empty directory to serve as the remote."""
    root = tmp_path / "project"
    rng = np.random.default_rng(0)
    with SignalStore.builder(root / "stores" / "cohort", channels=2) as builder:
        for subject in range(4):
            builder.add(
                f"p{subject}",
                rng.standard_normal((500, 2)).astype("float32"),
                group=f"p{subject}",
            )
    (tmp_path / "remote").mkdir()
    return root


@pytest.fixture
def remote(tmp_path: Path) -> str:
    return str(tmp_path / "remote")


# --- resolving the remote -----------------------------------------------------------


def test_committed_mapping_is_what_makes_a_fresh_clone_work(project: Path, remote: str) -> None:
    """An environment variable cannot be cloned; a committed file can."""
    write_remotes({"cohort": remote}, root=project / "stores")
    assert resolve_remote("cohort", root=project / "stores") == remote


def test_a_default_entry_covers_every_store(project: Path, remote: str) -> None:
    write_remotes({"default": remote}, root=project / "stores")
    assert resolve_remote("anything", root=project / "stores") == remote


def test_explicit_beats_environment_beats_committed(
    project: Path, remote: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_remotes({"cohort": "file:///committed"}, root=project / "stores")
    monkeypatch.setenv(REMOTE_ENV, "file:///from-env")
    assert resolve_remote("cohort", root=project / "stores") == "file:///from-env"
    assert resolve_remote("cohort", "file:///explicit", root=project / "stores") == (
        "file:///explicit"
    )


def test_no_remote_configured_says_exactly_what_to_write(project: Path) -> None:
    with pytest.raises(RemoteError, match="remotes:"):
        resolve_remote("cohort", root=project / "stores")


# --- push ----------------------------------------------------------------------------


def test_push_uploads_every_payload_file(project: Path, remote: str) -> None:
    transfers = push(project / "stores" / "cohort", remote=remote)
    assert len(transfers) == 3
    assert all(not item.skipped for item in transfers)
    for item in transfers:
        assert Path(object_key(remote, item.digest)).is_file()


def test_pushing_an_unchanged_store_transfers_nothing(project: Path, remote: str) -> None:
    """Content addressing means a second push is a few existence checks."""
    push(project / "stores" / "cohort", remote=remote)
    second = push(project / "stores" / "cohort", remote=remote)
    assert all(item.skipped for item in second)


def test_push_refuses_a_locally_corrupt_store(project: Path, remote: str) -> None:
    """Publishing bytes that contradict the manifest puts a corrupt store behind a digest
    that promises otherwise, which is worse than failing."""
    signal = project / "stores" / "cohort" / SIGNAL_FILE
    data = bytearray(signal.read_bytes())
    data[100:110] = b"\x00" * 10
    signal.write_bytes(bytes(data))
    with pytest.raises(Exception, match="modified or corrupted"):
        push(project / "stores" / "cohort", remote=remote)


def test_dry_run_reports_without_sending(project: Path, remote: str) -> None:
    transfers = push(project / "stores" / "cohort", remote=remote, dry_run=True)
    assert all(not item.skipped for item in transfers)
    assert not any(Path(object_key(remote, item.digest)).exists() for item in transfers)


# --- pull ----------------------------------------------------------------------------


def test_fresh_clone_pulls_and_verifies(project: Path, remote: str, tmp_path: Path) -> None:
    """The headline test. Clone with only the committed manifest, pull, and have a store.

    This is the property FORGE lacked: its checkpoints reached for a hardcoded encoder path
    that existed on one machine, so a fresh clone could not reproduce anything.
    """
    push(project / "stores" / "cohort", remote=remote)
    original = SignalStore(project / "stores" / "cohort").read(0, 500)

    clone = tmp_path / "clone" / "stores" / "cohort"
    clone.mkdir(parents=True)
    # git carries the manifest and nothing else; the payload is gitignored.
    shutil.copy(project / "stores" / "cohort" / MANIFEST_FILE, clone / MANIFEST_FILE)

    transfers = pull("cohort", remote=remote, root=tmp_path / "clone" / "stores")
    assert len(transfers) == 3
    restored = SignalStore(clone)
    restored.verify()
    assert np.array_equal(restored.read(0, 500), original)


def test_pull_fetches_only_what_git_did_not_carry(
    project: Path, remote: str, tmp_path: Path
) -> None:
    """The real clone shape, not an idealised one.

    dsio's .gitignore commits manifest.yaml *and* entities.jsonl — the entity table is
    small and is what makes a split file reviewable, since group membership lives there —
    while the two large binaries are ignored. Pull must therefore skip the file git already
    provided rather than re-downloading it, which falls out of digest checking rather than
    needing a rule of its own.
    """
    push(project / "stores" / "cohort", remote=remote)
    source = project / "stores" / "cohort"
    clone = tmp_path / "clone" / "stores" / "cohort"
    clone.mkdir(parents=True)
    for tracked in (MANIFEST_FILE, "entities.jsonl"):
        shutil.copy(source / tracked, clone / tracked)

    transfers = {item.filename: item for item in pull("cohort", remote=remote, root=clone.parent)}
    assert transfers["entities.jsonl"].skipped is True
    assert transfers["signal.bin"].skipped is False
    assert transfers["signal.idx"].skipped is False
    SignalStore(clone).verify()


def test_pull_skips_files_that_already_match(project: Path, remote: str) -> None:
    push(project / "stores" / "cohort", remote=remote)
    transfers = pull("cohort", remote=remote, root=project / "stores")
    assert all(item.skipped for item in transfers)


def test_pull_force_refetches(project: Path, remote: str) -> None:
    push(project / "stores" / "cohort", remote=remote)
    transfers = pull("cohort", remote=remote, root=project / "stores", force=True)
    assert all(not item.skipped for item in transfers)


def test_pull_without_a_committed_manifest_explains_why(
    project: Path, remote: str, tmp_path: Path
) -> None:
    """The manifest is the index. Without it there is nothing to fetch *against*."""
    (tmp_path / "bare" / "stores" / "cohort").mkdir(parents=True)
    with pytest.raises(RemoteError, match="manifest is the index"):
        pull("cohort", remote=remote, root=tmp_path / "bare" / "stores")


def test_pull_fails_when_the_object_was_never_pushed(project: Path, remote: str) -> None:
    with pytest.raises(RemoteError, match="no such object"):
        pull("cohort", remote=remote, root=project / "stores", force=True)


def test_pull_refuses_bytes_that_contradict_the_manifest(
    project: Path, remote: str, tmp_path: Path
) -> None:
    """A tampered or truncated object must never be installed, and must leave no residue.

    Content addressing makes this nearly impossible by construction, which is exactly why
    it is worth asserting: if it ever fires, something is badly wrong with the remote and
    nothing fetched from it should be trusted.
    """
    push(project / "stores" / "cohort", remote=remote)
    manifest = SignalStore(project / "stores" / "cohort").manifest()
    corrupt = Path(object_key(remote, manifest.signal_sha256))
    corrupt.write_bytes(b"not the signal you are looking for")

    clone = tmp_path / "clone" / "stores" / "cohort"
    clone.mkdir(parents=True)
    shutil.copy(project / "stores" / "cohort" / MANIFEST_FILE, clone / MANIFEST_FILE)

    with pytest.raises(RemoteIntegrityError, match="refusing to install"):
        pull("cohort", remote=remote, root=tmp_path / "clone" / "stores")
    assert not (clone / SIGNAL_FILE).exists()
    assert not list(clone.glob("*.partial")), "a failed pull must leave nothing behind"


def test_an_old_manifest_still_resolves_after_reingest(
    project: Path, remote: str, tmp_path: Path
) -> None:
    """The reason objects are addressed by content and not by path.

    Re-ingesting a corpus under path addressing overwrites signal.bin, and every split file
    bound to the old store digest — and every run that cites it — becomes unreproducible in
    silence. Here the old bytes keep their own name and a year-old run still pulls exactly
    what it was trained on.
    """
    push(project / "stores" / "cohort", remote=remote)
    old_manifest = (project / "stores" / "cohort" / MANIFEST_FILE).read_bytes()
    old_digest = SignalStore(project / "stores" / "cohort").manifest().signal_sha256

    # Re-ingest the same corpus with an extra subject, and push again.
    shutil.rmtree(project / "stores" / "cohort")
    rng = np.random.default_rng(1)
    with SignalStore.builder(project / "stores" / "cohort", channels=2) as builder:
        for subject in range(5):
            builder.add(
                f"p{subject}", rng.standard_normal((500, 2)).astype("float32"), group=f"p{subject}"
            )
    push(project / "stores" / "cohort", remote=remote)
    new_digest = SignalStore(project / "stores" / "cohort").manifest().signal_sha256
    assert new_digest != old_digest

    # A checkout of the older commit carries the older manifest, and it still resolves.
    historical = tmp_path / "historical" / "stores" / "cohort"
    historical.mkdir(parents=True)
    (historical / MANIFEST_FILE).write_bytes(old_manifest)
    pull("cohort", remote=remote, root=tmp_path / "historical" / "stores")
    assert SignalStore(historical).manifest().signal_sha256 == old_digest


# --- status ---------------------------------------------------------------------------


def test_status_reports_here_there_or_neither(project: Path, remote: str) -> None:
    before = status("cohort", remote=remote, root=project / "stores")
    assert before["complete"] is True and before["pushed"] is False

    push(project / "stores" / "cohort", remote=remote)
    after = status("cohort", remote=remote, root=project / "stores")
    assert after["complete"] is True and after["pushed"] is True
    assert {entry["file"] for entry in after["files"]} == {
        "signal.bin",
        "signal.idx",
        "entities.jsonl",
    }


def test_object_keys_fan_out_by_digest_prefix() -> None:
    """A flat directory of a million objects degrades badly on several backends."""
    key = object_key("s3://bucket/dsio", "abcdef" + "0" * 58)
    assert key == "s3://bucket/dsio/objects/ab/" + "abcdef" + "0" * 58
