"""A consumer project owns its Prefect flow and calls DSio as a library."""

from __future__ import annotations

from pathlib import Path

import pytest
from prefect import flow, task
from prefect.testing.utilities import prefect_test_harness

from dsio.contracts import sha256_of


@task
def identify_dataset(dataset: dict[str, object]) -> str:
    return sha256_of(dataset)


@flow
def project_flow() -> str:
    return identify_dataset({"name": "algae", "revision": 1})


def test_project_owned_flow_calls_public_dsio_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))

    with prefect_test_harness():
        result = project_flow()

    assert result == "2ca217b09c12c36031e1078aab6a81e705f1281b1eb841c876f98cef5cb03556"
