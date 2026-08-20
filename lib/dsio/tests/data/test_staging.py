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

    def good(out: Path) -> None:
        out.write_bytes(b"complete")

    assert stage("windows", {"length": 500}, good, root=tmp_path).read_bytes() == b"complete"
