from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch


def test_calibration_selects_measured_throughput_without_changing_effective_batch() -> None:
    from dsio.experimental.execution import calibrate_execution

    measured: list[dict[str, Any]] = []

    def benchmark(candidate: Mapping[str, Any]) -> Mapping[str, float | int]:
        measured.append(dict(candidate))
        throughput = {16: 120.0, 64: 410.0, 256: 390.0}[candidate["batch_size"]]
        return {
            "examples_per_second": throughput,
            "peak_device_memory_bytes": candidate["batch_size"] * 1_000,
        }

    evidence = calibrate_execution(
        benchmark,
        target_effective_batch_size=256,
        candidates=[
            {"batch_size": 16, "num_workers": 2, "pin_memory": True},
            {"batch_size": 64, "num_workers": 4, "pin_memory": True},
            {"batch_size": 256, "num_workers": 8, "pin_memory": True},
        ],
    )

    assert [candidate["accumulate_grad_batches"] for candidate in measured] == [16, 4, 1]
    assert all(
        candidate["batch_size"] * candidate["accumulate_grad_batches"] == 256
        for candidate in measured
    )
    assert evidence["selected"] == measured[1]
    assert [trial["measurement"]["examples_per_second"] for trial in evidence["trials"]] == [
        120.0,
        410.0,
        390.0,
    ]
    assert evidence["policy"] == {
        "objective": "max_examples_per_second",
        "objective_tolerance_fraction": 0.02,
        "target_effective_batch_size": 256,
        "max_device_memory_bytes": None,
        "max_host_memory_bytes": None,
    }
    assert evidence["environment"]["torch_version"] == torch.__version__
    assert evidence["environment"]["device_type"] in {"cpu", "cuda"}
    assert len(evidence["environment_digest"]) == 64


def test_calibration_rejects_candidates_that_change_scientific_configuration() -> None:
    from dsio.experimental.execution import calibrate_execution

    with pytest.raises(ValueError, match="execution-only.*learning_rate"):
        calibrate_execution(
            lambda _candidate: {
                "examples_per_second": 1.0,
                "peak_device_memory_bytes": 1,
            },
            target_effective_batch_size=32,
            candidates=[{"batch_size": 16, "learning_rate": 0.1}],
        )


def test_calibration_rejects_a_micro_batch_that_changes_the_effective_batch() -> None:
    from dsio.experimental.execution import calibrate_execution

    with pytest.raises(ValueError, match="batch_size 24.*effective batch 64"):
        calibrate_execution(
            lambda _candidate: {
                "examples_per_second": 1.0,
                "peak_device_memory_bytes": 1,
            },
            target_effective_batch_size=64,
            candidates=[{"batch_size": 24}],
        )


def test_calibration_rejects_fast_candidates_outside_the_memory_budget() -> None:
    from dsio.experimental.execution import calibrate_execution

    def benchmark(candidate: Mapping[str, Any]) -> Mapping[str, float | int]:
        batch_size = candidate["batch_size"]
        return {
            "examples_per_second": float(batch_size),
            "peak_device_memory_bytes": batch_size * 100,
        }

    evidence = calibrate_execution(
        benchmark,
        target_effective_batch_size=64,
        candidates=[{"batch_size": 16}, {"batch_size": 64}],
        max_device_memory_bytes=2_000,
    )

    assert evidence["selected"]["batch_size"] == 16
    assert [trial["status"] for trial in evidence["trials"]] == ["admissible", "rejected"]
    assert evidence["trials"][1]["reason"] == "device_memory_budget"


def test_calibration_records_cuda_oom_and_continues_with_smaller_candidates() -> None:
    from dsio.experimental.execution import calibrate_execution

    def benchmark(candidate: Mapping[str, Any]) -> Mapping[str, float | int]:
        if candidate["batch_size"] == 64:
            raise torch.cuda.OutOfMemoryError("candidate does not fit")
        return {"examples_per_second": 10.0, "peak_device_memory_bytes": 100}

    evidence = calibrate_execution(
        benchmark,
        target_effective_batch_size=64,
        candidates=[{"batch_size": 64}, {"batch_size": 16}],
    )

    assert evidence["selected"]["batch_size"] == 16
    assert evidence["trials"][0]["status"] == "rejected"
    assert evidence["trials"][0]["reason"] == "cuda_out_of_memory"


def test_execution_calibration_is_admissible_for_consumer_projects() -> None:
    from dsio.experimental import require_admissible_component

    require_admissible_component(
        "dsio.experimental.execution:calibrate_execution",
        project_names=("parkinsons_fog", "child_mind"),
    )


def test_calibration_can_optimize_projected_time_instead_of_warm_throughput() -> None:
    from dsio.experimental.execution import calibrate_execution

    def benchmark(candidate: Mapping[str, Any]) -> Mapping[str, float | int]:
        if candidate["num_workers"] == 8:
            return {
                "examples_per_second": 900.0,
                "projected_examples_per_second": 100.0,
                "peak_device_memory_bytes": 10,
            }
        return {
            "examples_per_second": 500.0,
            "projected_examples_per_second": 400.0,
            "peak_device_memory_bytes": 10,
        }

    evidence = calibrate_execution(
        benchmark,
        target_effective_batch_size=64,
        candidates=[
            {"batch_size": 64, "num_workers": 8},
            {"batch_size": 64, "num_workers": 2},
        ],
        objective_metric="projected_examples_per_second",
    )

    assert evidence["selected"]["num_workers"] == 2
    assert evidence["policy"]["objective"] == "max_projected_examples_per_second"


def test_calibration_prefers_simpler_execution_when_measurements_are_effectively_tied() -> None:
    from dsio.experimental.execution import calibrate_execution

    def benchmark(candidate: Mapping[str, Any]) -> Mapping[str, float | int]:
        throughput = {256: 15_726.0, 1024: 15_701.0}[candidate["batch_size"]]
        return {
            "examples_per_second": throughput,
            "peak_device_memory_bytes": candidate["batch_size"] * 1_000,
        }

    evidence = calibrate_execution(
        benchmark,
        target_effective_batch_size=1024,
        candidates=[
            {"batch_size": 256, "num_workers": 0, "pin_memory": True},
            {"batch_size": 1024, "num_workers": 0, "pin_memory": True},
        ],
    )

    assert evidence["selected"]["batch_size"] == 1024


def test_calibration_fingerprints_the_measured_device_not_cuda_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dsio.experimental.execution import calibrate_execution

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    evidence = calibrate_execution(
        lambda _candidate: {
            "examples_per_second": 1.0,
            "peak_device_memory_bytes": 0,
        },
        target_effective_batch_size=8,
        candidates=[{"batch_size": 8}],
        device="cpu",
    )

    assert evidence["environment"]["device_type"] == "cpu"
    assert "device_name" not in evidence["environment"]


def test_calibration_fails_cleanly_when_every_candidate_exceeds_a_budget() -> None:
    from dsio.experimental.execution import calibrate_execution

    with pytest.raises(ValueError, match="no admissible execution candidate"):
        calibrate_execution(
            lambda _candidate: {
                "examples_per_second": 1.0,
                "peak_device_memory_bytes": 100,
            },
            target_effective_batch_size=8,
            candidates=[{"batch_size": 8}],
            max_device_memory_bytes=10,
        )
