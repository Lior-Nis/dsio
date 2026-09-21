"""Prefect owns caching for pure deterministic values."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_cache_key_is_deterministic_and_input_sensitive() -> None:
    from dsio.tracking import prefect_cache_key

    context = SimpleNamespace(task=SimpleNamespace(task_key="pure-task"))
    assert prefect_cache_key(
        context, {"left": 1, "right": [2, 3]}
    ) == prefect_cache_key(
        context,
        {"right": (2, 3), "left": 1},
    )
    assert prefect_cache_key(context, {"value": 1}) != prefect_cache_key(
        context, {"value": 2}
    )
    other = SimpleNamespace(task=SimpleNamespace(task_key="other-task"))
    assert prefect_cache_key(context, {"value": 1}) != prefect_cache_key(
        other, {"value": 1}
    )


def test_prefect_uses_the_identity_key_for_a_pure_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(os.environ):
        if name.startswith("PREFECT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    from prefect import flow, task
    from prefect.testing.utilities import prefect_test_harness

    from dsio.tracking import prefect_cache_key

    calls: list[str] = []

    @task(cache_key_fn=prefect_cache_key, persist_result=True)
    def pure_double(value: int) -> int:
        calls.append("double")
        return value * 2

    @task(cache_key_fn=prefect_cache_key, persist_result=True)
    def pure_square(value: int) -> int:
        calls.append("square")
        return value * value

    @flow
    def run() -> tuple[int, int]:
        return pure_double(4), pure_square(4)

    with prefect_test_harness():
        assert run() == (8, 16)
        assert run() == (8, 16)
    assert calls == ["double", "square"]
