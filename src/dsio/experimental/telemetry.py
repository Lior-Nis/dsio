"""Linux/CUDA resource evidence shared by measured training experiments."""

from __future__ import annotations

import math
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


@contextmanager
def measure_phase(
    phase: str,
    *,
    cuda_device: int | None = None,
    sample_interval_seconds: float = 0.1,
    proc_root: str | Path = "/proc",
) -> Iterator[dict[str, Any]]:
    """Measure one phase without introducing a DSIO telemetry abstraction."""
    if not phase:
        raise ValueError("phase must be non-empty")
    if (
        isinstance(sample_interval_seconds, bool)
        or not isinstance(sample_interval_seconds, (int, float))
        or not math.isfinite(sample_interval_seconds)
        or sample_interval_seconds <= 0
    ):
        raise ValueError("sample_interval_seconds must be a finite positive number")
    if cuda_device is not None and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; scale training will not fall back"
        )
    if cuda_device is not None and (
        isinstance(cuda_device, bool) or not isinstance(cuda_device, int) or cuda_device < 0
    ):
        raise ValueError(f"cuda_device must be a non-negative integer, got {cuda_device!r}")

    cuda_requested = cuda_device is not None
    cuda_uuid = _resolve_cuda_uuid(cuda_device) if cuda_device is not None else None
    evidence: dict[str, Any] = {
        "phase": phase,
        "cache_state": "uncontrolled-os-page-cache",
        "measurement_method": {
            "elapsed": "time.perf_counter",
            "process_tree_rss": "linux-proc-status-vmrss-summed-all-thread-children",
            "cuda_memory": "torch.cuda.max_memory_allocated-and-reserved"
            if cuda_requested
            else "not_applicable",
            "gpu_utilization": "nvidia-smi-device-sampling"
            if cuda_requested
            else "not_applicable",
        },
        "cuda": {
            "requested": cuda_requested,
            "device": cuda_device,
            "uuid": cuda_uuid,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "utilization_samples": 0,
            "utilization_mean_percent": None,
            "utilization_peak_percent": None,
        },
    }
    if cuda_device is not None:
        with torch.cuda.device(cuda_device):
            torch.cuda.reset_peak_memory_stats()

    proc_root = Path(proc_root)
    samples: list[int] = [_process_tree_rss(proc_root)]
    stop = threading.Event()
    utilization: list[float] = (
        [_gpu_utilization(cuda_uuid)] if cuda_uuid is not None else []
    )
    sampler_errors: list[Exception] = []
    sampler = threading.Thread(
        target=_sample_resources,
        args=(
            stop,
            sample_interval_seconds,
            cuda_uuid,
            proc_root,
            samples,
            utilization,
            sampler_errors,
        ),
        name=f"dsio-{phase}-telemetry",
        daemon=False,
    )
    started = time.perf_counter()
    sampler.start()
    body_error: BaseException | None = None
    try:
        yield evidence
    except BaseException as error:
        body_error = error
        raise
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        while sampler.is_alive():
            sampler.join(timeout=0.1)
        try:
            samples.append(_process_tree_rss(proc_root))
        except RuntimeError:
            if body_error is None:
                raise
        evidence["elapsed_seconds"] = elapsed
        evidence["peak_process_tree_rss_bytes"] = max(samples)
        if cuda_device is not None:
            with torch.cuda.device(cuda_device):
                evidence["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                evidence["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            evidence["cuda"]["utilization_samples"] = len(utilization)
            if utilization:
                evidence["cuda"]["utilization_mean_percent"] = sum(utilization) / len(
                    utilization
                )
                evidence["cuda"]["utilization_peak_percent"] = max(utilization)
        if sampler_errors and body_error is None:
            raise RuntimeError("CUDA telemetry sampling failed") from sampler_errors[0]
        if cuda_requested and not utilization and body_error is None:
            raise RuntimeError("CUDA telemetry produced zero utilization samples")


def log_phase_evidence(
    run_id: str,
    namespace: str,
    evidence: dict[str, Any],
    measurements: dict[str, int | float],
) -> None:
    """Log one phase's consumer-owned resource artifact and native MLflow metrics."""
    from mlflow import MlflowClient

    payload = {"resources": evidence, "measurements": measurements}
    client = MlflowClient()
    client.log_dict(run_id, payload, f"outputs/{namespace}-telemetry.json")
    metrics: dict[str, float] = {
        "elapsed_seconds": float(evidence["elapsed_seconds"]),
        "peak_process_tree_rss_bytes": float(evidence["peak_process_tree_rss_bytes"]),
        **{name: float(value) for name, value in measurements.items()},
    }
    cuda = evidence["cuda"]
    for name in (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "utilization_samples",
        "utilization_mean_percent",
        "utilization_peak_percent",
    ):
        value = cuda[name]
        if value is not None:
            metrics[f"cuda.{name}"] = float(value)
    for name, value in metrics.items():
        client.log_metric(run_id, f"scale.{namespace}.{name}", value)


def _sample_resources(
    stop: threading.Event,
    interval: float,
    cuda_uuid: str | None,
    proc_root: Path,
    rss_samples: list[int],
    utilization_samples: list[float],
    errors: list[Exception],
) -> None:
    try:
        while True:
            rss_samples.append(_process_tree_rss(proc_root))
            if cuda_uuid is not None:
                utilization_samples.append(_gpu_utilization(cuda_uuid))
            if stop.wait(interval):
                return
    except RuntimeError as error:
        errors.append(error)
        stop.set()


def _process_tree_rss(proc_root: Path) -> int:
    if not proc_root.is_dir():
        raise RuntimeError(f"process telemetry is unavailable: {proc_root} is not a directory")
    root_pid = _pid()
    pending = [proc_root / str(root_pid)]
    observed: set[int] = set()
    total = 0
    while pending:
        process = pending.pop()
        try:
            pid = int(process.name)
        except ValueError:
            continue
        if pid in observed:
            continue
        observed.add(pid)
        total += _rss(process / "status", required=pid == root_pid)
        children = _child_pids(process, required=pid == root_pid)
        pending.extend(proc_root / child for child in children)
    if total <= 0:
        raise RuntimeError("process telemetry is unavailable: sampled process-tree RSS was zero")
    return total


def _child_pids(process: Path, *, required: bool) -> list[str]:
    task_directory = process / "task"
    try:
        threads = list(task_directory.iterdir())
    except OSError as error:
        if required:
            raise RuntimeError(
                f"process telemetry is unavailable: cannot read {task_directory}"
            ) from error
        return []
    children: list[str] = []
    readable = False
    for thread in threads:
        try:
            children.extend((thread / "children").read_text().split())
            readable = True
        except OSError:
            continue
    if required and not readable:
        raise RuntimeError(
            f"process telemetry is unavailable: no thread children file was readable in "
            f"{task_directory}"
        )
    return children


def _pid() -> int:
    import os

    return os.getpid()


def _rss(status: Path, *, required: bool) -> int:
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError) as error:
        if required:
            raise RuntimeError(f"process telemetry is unavailable: cannot read {status}") from error
        return 0
    if required:
        raise RuntimeError(f"process telemetry is unavailable: VmRSS is absent from {status}")
    return 0


def _resolve_cuda_uuid(device: int) -> str:
    raw_uuid = str(torch.cuda.get_device_properties(device).uuid)
    selector = raw_uuid if raw_uuid.startswith(("GPU-", "MIG-")) else f"GPU-{raw_uuid}"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={selector}",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        uuid = completed.stdout.strip().splitlines()[0]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"CUDA telemetry could not query nvidia-smi for logical device {device}"
        ) from error
    if not uuid:
        raise RuntimeError(f"CUDA telemetry received an empty UUID for logical device {device}")
    return uuid


def _gpu_utilization(uuid: str) -> float:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return float(completed.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"CUDA telemetry failed to sample nvidia-smi for {uuid}") from error
