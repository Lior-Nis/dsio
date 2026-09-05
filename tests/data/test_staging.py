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


def test_non_identity_config_does_not_invalidate_a_stage(tmp_path: Path):
    """Tuning is not identity.

    Hashing the whole config means bumping a worker count or a prefetch depth rebuilds a
    byte-identical artifact. On a large corpus that is hours, and it is invisible — the
    stage simply appears to be missing.
    """
    calls = []

    def build(out: Path) -> None:
        calls.append(out)
        out.write_bytes(b"payload")

    identity = ("length", "stride")
    first = stage(
        "windows",
        {"length": 500, "stride": 250, "num_workers": 4},
        build,
        root=tmp_path,
        identity_fields=identity,
    )
    second = stage(
        "windows",
        {"length": 500, "stride": 250, "num_workers": 16},
        build,
        root=tmp_path,
        identity_fields=identity,
    )

    assert first == second
    assert len(calls) == 1


def test_identity_config_still_invalidates_a_stage(tmp_path: Path):
    def build(out: Path) -> None:
        out.write_bytes(b"x")

    identity = ("length", "stride")
    a = stage(
        "w", {"length": 500, "stride": 250, "n": 1}, build, root=tmp_path, identity_fields=identity
    )
    b = stage(
        "w", {"length": 250, "stride": 250, "n": 1}, build, root=tmp_path, identity_fields=identity
    )
    assert a != b


def test_an_identity_field_absent_from_the_config_is_refused(tmp_path: Path):
    """A typo'd identity field silently *narrows* the key, so two different configs
    collide onto one stage. That is the worst failure a cache key has."""

    def build(out: Path) -> None:
        out.write_bytes(b"x")

    with pytest.raises(StagingError, match="stide"):
        stage(
            "w",
            {"length": 500, "stride": 250},
            build,
            root=tmp_path,
            identity_fields=("length", "stide"),
        )


def test_hashing_the_whole_config_remains_the_default(tmp_path: Path):
    """Opt-in only: without identity_fields, every key is identity-bearing."""

    def build(out: Path) -> None:
        out.write_bytes(b"x")

    a = stage("w", {"length": 500, "num_workers": 4}, build, root=tmp_path)
    b = stage("w", {"length": 500, "num_workers": 16}, build, root=tmp_path)
    assert a != b
