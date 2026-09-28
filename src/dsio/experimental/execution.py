"""Measured selection among explicitly declared execution configurations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

Benchmark = Callable[[Mapping[str, Any]], Mapping[str, float | int]]
_EXECUTION_KEYS = frozenset({"batch_size", "num_workers", "pin_memory", "prefetch_factor"})


def calibrate_execution(
    benchmark: Benchmark,
    *,
    target_effective_batch_size: int,
    candidates: Sequence[Mapping[str, Any]],
    max_device_memory_bytes: int | None = None,
    max_host_memory_bytes: int | None = None,
    objective_metric: str = "examples_per_second",
    objective_tolerance_fraction: float = 0.02,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Measure execution-only candidates and return the fastest admissible one.

    Each micro-batch divides one declared nominal effective batch, and accumulation is
    derived rather than tuned. Different micro-batches can still change floating-point
    numerics or batch-dependent model behavior, so the selected configuration remains
    explicit provenance rather than being treated as an equivalent replay.
    """
    if not callable(benchmark):
        raise ValueError("benchmark must be callable")
    if (
        isinstance(target_effective_batch_size, bool)
        or not isinstance(target_effective_batch_size, int)
        or target_effective_batch_size <= 0
    ):
        raise ValueError("target_effective_batch_size must be a positive integer")
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError("candidates must be a non-empty sequence of mappings")
    if not candidates:
        raise ValueError("candidates must be a non-empty sequence of mappings")
    _optional_positive_integer("max_device_memory_bytes", max_device_memory_bytes)
    _optional_positive_integer("max_host_memory_bytes", max_host_memory_bytes)
    if not isinstance(objective_metric, str) or not objective_metric:
        raise ValueError("objective_metric must be a non-empty string")
    if (
        isinstance(objective_tolerance_fraction, bool)
        or not isinstance(objective_tolerance_fraction, int | float)
        or not 0 <= objective_tolerance_fraction < 1
    ):
        raise ValueError("objective_tolerance_fraction must be a number in [0, 1)")
    trials: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("each calibration candidate must be a mapping")
        unsupported = sorted(set(candidate) - _EXECUTION_KEYS)
        if unsupported:
            raise ValueError(
                "calibration candidates accept execution-only settings; unsupported: "
                + ", ".join(unsupported)
            )
        resolved = {
            "batch_size": candidate.get("batch_size"),
            "num_workers": candidate.get("num_workers", 0),
            "pin_memory": candidate.get("pin_memory", False),
            "prefetch_factor": candidate.get("prefetch_factor", 2),
        }
        batch_size = resolved.get("batch_size")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("candidate batch_size must be a positive integer")
        if target_effective_batch_size % batch_size:
            raise ValueError(
                f"candidate batch_size {batch_size} does not divide effective batch "
                f"{target_effective_batch_size}"
            )
        num_workers = resolved["num_workers"]
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers < 0
        ):
            raise ValueError("candidate num_workers must be a non-negative integer")
        if not isinstance(resolved["pin_memory"], bool):
            raise ValueError("candidate pin_memory must be bool")
        prefetch_factor = resolved["prefetch_factor"]
        if (
            isinstance(prefetch_factor, bool)
            or not isinstance(prefetch_factor, int)
            or prefetch_factor <= 0
        ):
            raise ValueError("candidate prefetch_factor must be a positive integer")
        resolved["accumulate_grad_batches"] = target_effective_batch_size // batch_size
        try:
            measurement = dict(benchmark(resolved))
        except torch.cuda.OutOfMemoryError as error:
            trials.append(
                {
                    "candidate": resolved,
                    "status": "rejected",
                    "reason": "cuda_out_of_memory",
                    "error": str(error),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        _positive_measurement(measurement, objective_metric)
        _nonnegative_measurement(measurement, "peak_device_memory_bytes")
        if max_host_memory_bytes is not None:
            _nonnegative_measurement(measurement, "peak_process_tree_rss_bytes")
        exceeds_memory = (
            max_device_memory_bytes is not None
            and measurement["peak_device_memory_bytes"] > max_device_memory_bytes
        )
        exceeds_host_memory = (
            max_host_memory_bytes is not None
            and measurement["peak_process_tree_rss_bytes"] > max_host_memory_bytes
        )
        trial = {
            "candidate": resolved,
            "measurement": measurement,
            "status": "rejected" if exceeds_memory or exceeds_host_memory else "admissible",
        }
        if exceeds_memory:
            trial["reason"] = "device_memory_budget"
        elif exceeds_host_memory:
            trial["reason"] = "host_memory_budget"
        trials.append(trial)

    admissible = [trial for trial in trials if trial["status"] == "admissible"]
    if not admissible:
        reasons = sorted({str(trial.get("reason", "unknown")) for trial in trials})
        raise ValueError(
            "no admissible execution candidate; rejected by: " + ", ".join(reasons)
        )
    best_objective = max(
        trial["measurement"][objective_metric] for trial in admissible
    )
    effectively_tied = [
        trial
        for trial in admissible
        if trial["measurement"][objective_metric]
        >= best_objective * (1 - objective_tolerance_fraction)
    ]
    selected = min(effectively_tied, key=_execution_complexity)["candidate"]
    environment = _environment(device)
    return {
        "policy": {
            "objective": f"max_{objective_metric}",
            "objective_tolerance_fraction": float(objective_tolerance_fraction),
            "target_effective_batch_size": target_effective_batch_size,
            "max_device_memory_bytes": max_device_memory_bytes,
            "max_host_memory_bytes": max_host_memory_bytes,
        },
        "selected": selected,
        "trials": trials,
        "environment": environment,
        "environment_digest": hashlib.sha256(
            json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _execution_complexity(trial: Mapping[str, Any]) -> tuple[int, int, int, bool]:
    candidate = trial["candidate"]
    return (
        candidate["accumulate_grad_batches"],
        candidate["num_workers"],
        candidate["prefetch_factor"],
        candidate["pin_memory"],
    )


def _environment(device: str | torch.device | None) -> dict[str, Any]:
    resolved = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError(f"calibration device must be CPU or CUDA, got {resolved.type!r}")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA calibration device is unavailable; DSIO will not fall back to CPU")
    environment: dict[str, Any] = {
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device_type": resolved.type,
        "host_memory_bytes": int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")),
    }
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        environment.update(
            device_index=index,
            device_name=properties.name,
            device_total_memory_bytes=properties.total_memory,
            device_capability=[properties.major, properties.minor],
            device_uuid=str(properties.uuid),
        )
    return environment


def _optional_positive_integer(name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")


def _positive_measurement(measurement: Mapping[str, Any], name: str) -> None:
    value = measurement.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"benchmark measurement {name!r} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"benchmark measurement {name!r} must be a finite positive number")


def _nonnegative_measurement(measurement: Mapping[str, Any], name: str) -> None:
    value = measurement.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"benchmark measurement {name!r} must be a finite non-negative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"benchmark measurement {name!r} must be a finite non-negative number")


__all__ = ["calibrate_execution"]
