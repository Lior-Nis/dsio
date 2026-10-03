"""Small assertions shared by the independent Kaggle consumer tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlflow import MlflowClient


def assert_execution_evidence(
    run_id: str,
    destination: Path,
    *,
    optimizer: str,
    optimizer_parameters: dict[str, float],
    fold: int = 0,
    batch_size: int = 4,
    num_workers: int = 0,
    dataset: object | None = None,
    collator: Mapping[str, Any] | None = None,
    objective: object | None = None,
    model: object | None = None,
) -> None:
    client = MlflowClient()
    run = client.get_run(run_id)
    artifact = client.download_artifacts(run_id, "provenance.json", str(destination))
    provenance = json.loads(Path(artifact).read_text())
    configuration = provenance["configuration"]
    execution = configuration["execution"]

    assert execution["requested_accelerator"] == "cpu"
    assert execution["requested_devices"] == "1"
    assert execution["requested_precision"] == "32-true"
    assert execution["resolved_device"] == "cpu"
    assert execution["resolved_devices"] == "1"
    assert execution["resolved_precision"] == "32-true"
    for key, value in execution.items():
        assert run.data.params[f"execution.{key}"] == value
    assert configuration["fold"] == fold
    assert configuration["batch_size"] == batch_size
    assert configuration["num_workers"] == num_workers
    assert configuration["optimizer_parameters"] == optimizer_parameters
    assert provenance["components"]["optimizer"] == optimizer
    if objective is not None:
        assert provenance["components"]["objective"] == objective
    if model is not None:
        assert provenance["components"]["model"] == model
    if dataset is not None:
        assert collator is not None
        assert provenance["components"]["dataset_factory"] == dataset
        assert provenance["components"]["collator"] == collator


def assert_replay_run_ids_differ(first: dict[str, object], second: dict[str, object]) -> None:
    keys = {key for key in first if key.endswith("_run_id")}
    assert keys
    assert keys <= second.keys()
    for key in keys:
        assert first[key] != second[key]


def assert_downstream_evidence(
    result: dict[str, object],
    destination: Path,
    *,
    datasets: Mapping[str, object] | None = None,
    collator: Mapping[str, Any] | None = None,
    collators: Mapping[str, Mapping[str, Any]] | None = None,
    dynamic_axes: Mapping[str, list[int]] | None = None,
) -> None:
    client = MlflowClient()
    identities = result["identities"]
    assert isinstance(identities, dict)
    for stage in ("evaluation", "inference"):
        run_id = result[f"{stage}_run_id"]
        assert isinstance(run_id, str)
        artifact = client.download_artifacts(run_id, "provenance.json", str(destination / stage))
        configuration = json.loads(Path(artifact).read_text())["configuration"]
        assert configuration["export_identity"] == identities["export"]

    if datasets is not None:
        assert (collator is None) != (collators is None)
        assert set(datasets) == {"export", "evaluation", "inference"}
        if collators is not None:
            assert set(collators) == set(datasets)
        for stage, dataset in datasets.items():
            run_id = result[f"{stage}_run_id"]
            assert isinstance(run_id, str)
            artifact = client.download_artifacts(
                run_id,
                "provenance.json",
                str(destination / f"{stage}-components"),
            )
            provenance = json.loads(Path(artifact).read_text())
            components = provenance["components"]
            assert components["dataset_factory"] == dataset
            assert components["collator"] == (collator if collators is None else collators[stage])
            if stage == "export" and dynamic_axes is not None:
                assert provenance["configuration"]["dynamic_axes"] == dynamic_axes

    inference_run_id = result["inference_run_id"]
    assert isinstance(inference_run_id, str)
    artifact = client.download_artifacts(
        inference_run_id,
        "outputs/submission.json",
        str(destination / "submission"),
    )
    submission = json.loads(Path(artifact).read_text())
    assert submission["digest"] == result["submission_digest"]
