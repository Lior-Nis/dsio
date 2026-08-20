from pathlib import Path

import pytest

from dsio.data.staging import StagingError, stage


def test_builds_once_and_skips_on_repeat(tmp_path: Path):
    calls = []

    def build(out: Path) -> None:
        calls.append(out)
        out.write_bytes(b"payload")

    first = stage("windows", {"length": 500}, build, root=tmp_path)
    second = stage("windows", {"length": 500}, build, root=tmp_path)

    assert first == second
    assert len(calls) == 1
    assert first.read_bytes() == b"payload"


def test_different_config_is_a_different_path(tmp_path: Path):
    def build(out: Path) -> None:
        out.write_bytes(b"x")

    a = stage("windows", {"length": 500}, build, root=tmp_path)
    b = stage("windows", {"length": 250}, build, root=tmp_path)
    assert a != b


def test_a_failed_build_leaves_nothing_behind(tmp_path: Path):
    def build(out: Path) -> None:
        out.write_bytes(b"partial")
        raise RuntimeError("boom")

    with pytest.raises(StagingError):
        stage("windows", {"length": 500}, build, root=tmp_path)

    assert list(tmp_path.rglob("*.partial")) == []

    def good(out: Path) -> None:
        out.write_bytes(b"complete")

    assert stage("windows", {"length": 500}, good, root=tmp_path).read_bytes() == b"complete"


def test_a_failed_directory_build_leaves_nothing_behind(tmp_path: Path):
    """The default staging shape is a directory (a SignalStore root), not a file.

    A build that creates a directory before raising must still be cleaned up: naive
    ``Path.unlink`` raises ``IsADirectoryError`` on a directory, which (if unguarded)
    would replace the real build error and leave the half-built directory stranded —
    unusable forever, since the next attempt sees the target still missing, rebuilds
    into the same partial path, and the builder's own ``mkdir`` hits
    ``FileExistsError``.
    """

    def build(out: Path) -> None:
        out.mkdir(parents=True)
        (out / "data.bin").write_bytes(b"half-written")
        raise RuntimeError("boom")

    with pytest.raises(StagingError):
        stage("windows", {"length": 500}, build, root=tmp_path)

    assert list(tmp_path.rglob("*.partial")) == []

    def good(out: Path) -> None:
        out.mkdir(parents=True)
        (out / "data.bin").write_bytes(b"complete")

    result = stage("windows", {"length": 500}, good, root=tmp_path)
    assert (result / "data.bin").read_bytes() == b"complete"
