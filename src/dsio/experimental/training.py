"""Measured execution calibration for a real DSIO training workload."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence, Sized
from contextlib import suppress
from math import ceil
from pathlib import Path
from typing import Any

import torch
from lightning import seed_everything
from mlflow import MlflowClient

from dsio.data.loading import DsioDataModule
from dsio.experimental.execution import calibrate_execution
from dsio.experimental.telemetry import measure_phase
from dsio.model.module import DsioModule

ModuleFactory = Callable[[], DsioModule]
DataModuleFactory = Callable[[Mapping[str, Any]], DsioDataModule]


def calibrate_training_execution(
    module_factory: ModuleFactory,
    data_module_factory: DataModuleFactory,
    *,
    seed: int,
    accelerator: str,
    expected_epochs: int,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Calibrate the real loader/model/objective while keeping the effective batch fixed.

    Consumes:
        ``module_factory()`` returning a fresh ``DsioModule`` whose ``configure_optimizers``
        returns a bare optimizer, and ``data_module_factory(candidate)`` taking one candidate
        mapping (``batch_size``, ``num_workers``, ``pin_memory``, ``prefetch_factor``) and
        returning an object with ``setup(stage)`` and a ``train_dataloader()`` over a sized
        dataset whose batches carry ``sample_id``: the project's real model, objective and
        data.

    Produces:
        The :func:`~dsio.experimental.execution.calibrate_execution` evidence, measured by
        short warm-up and timed optimizer steps on the real training path; each trial's
        measurement adds ``projected_examples_per_second``, ``startup_seconds``,
        ``peak_process_tree_rss_bytes`` and GPU utilization.

    Parameters:
        ``seed``; ``accelerator`` (``"cpu"``, ``"cuda"`` or ``"auto"``);
        ``expected_epochs``; ``configuration`` with ``target_effective_batch_size`` and
        optional ``warmup_effective_batches`` (1), ``measure_effective_batches`` (10),
        ``candidates`` (derived from the target when omitted),
        ``max_device_memory_fraction`` (0.8) and ``max_host_memory_fraction`` (0.5).

    Devices:
        CPU or CUDA; fails closed when CUDA is requested but unavailable.

    Limitations:
        Requires an optimizer without a scheduler; loader workers import the full training
        stack, so worker count trades host memory against throughput; benchmarks are
        seconds long, not full epochs, and selection is timing-sensitive within the
        tolerance band.

    Example:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> from dsio.experimental.model import LossObjective
        >>> from dsio.model.module import DsioModule
        >>> rows = [
        ...     {"sample_id": f"s{i}", "x": torch.ones(4), "y": torch.ones(1)} for i in range(32)
        ... ]
        >>> class Rows:
        ...     def __init__(self, candidate):
        ...         self.candidate = candidate
        ...     def setup(self, stage):
        ...         pass
        ...     def train_dataloader(self):
        ...         return DataLoader(rows, batch_size=self.candidate["batch_size"])
        >>> def build_module():
        ...     return DsioModule(
        ...         model=torch.nn.Linear(4, 1),
        ...         objective=LossObjective(torch.nn.MSELoss()),
        ...         optimizer_factory=torch.optim.SGD,
        ...         optimizer_parameters={"lr": 0.01},
        ...     )
        >>> evidence = calibrate_training_execution(
        ...     build_module,
        ...     Rows,
        ...     seed=7,
        ...     accelerator="cpu",
        ...     expected_epochs=1,
        ...     configuration={
        ...         "target_effective_batch_size": 8,
        ...         "measure_effective_batches": 2,
        ...         "candidates": [{"batch_size": 8}, {"batch_size": 4}],
        ...     },
        ... )
        >>> selected = evidence["selected"]
        >>> selected["batch_size"] * selected["accumulate_grad_batches"]
        8
    """
    target = _positive_integer(
        "target_effective_batch_size", configuration.get("target_effective_batch_size")
    )
    warmup = _positive_integer(
        "warmup_effective_batches", configuration.get("warmup_effective_batches", 1)
    )
    measured = _positive_integer(
        "measure_effective_batches", configuration.get("measure_effective_batches", 10)
    )
    expected_epochs = _positive_integer("expected_epochs", expected_epochs)
    device = _device(accelerator)
    candidates = configuration.get("candidates")
    if candidates is None:
        candidates = _candidates(target, device.type)
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError("execution calibration candidates must be a sequence of mappings")
    memory_fraction = configuration.get("max_device_memory_fraction", 0.8)
    if isinstance(memory_fraction, bool) or not isinstance(memory_fraction, int | float):
        raise ValueError("max_device_memory_fraction must be a number in (0, 1]")
    if not 0 < memory_fraction <= 1:
        raise ValueError("max_device_memory_fraction must be a number in (0, 1]")
    host_memory_fraction = configuration.get("max_host_memory_fraction", 0.5)
    if isinstance(host_memory_fraction, bool) or not isinstance(host_memory_fraction, int | float):
        raise ValueError("max_host_memory_fraction must be a number in (0, 1]")
    if not 0 < host_memory_fraction <= 1:
        raise ValueError("max_host_memory_fraction must be a number in (0, 1]")
    max_memory = None
    if device.type == "cuda":
        free_device_memory, _ = torch.cuda.mem_get_info(device)
        max_memory = int(free_device_memory * memory_fraction)
    max_host_memory = int(_available_host_memory_bytes() * host_memory_fraction)

    def benchmark(candidate: Mapping[str, Any]) -> Mapping[str, float | int]:
        return _benchmark(
            module_factory,
            data_module_factory,
            seed=seed,
            device=device,
            candidate=candidate,
            expected_epochs=expected_epochs,
            warmup_effective_batches=warmup,
            measure_effective_batches=measured,
        )

    evidence = calibrate_execution(
        benchmark,
        target_effective_batch_size=target,
        candidates=candidates,
        max_device_memory_bytes=max_memory,
        max_host_memory_bytes=max_host_memory,
        objective_metric="projected_examples_per_second",
        device=device,
    )
    evidence["policy"].update(
        warmup_effective_batches=warmup,
        measure_effective_batches=measured,
        max_device_memory_fraction=float(memory_fraction),
        max_host_memory_fraction=float(host_memory_fraction),
        expected_epochs=expected_epochs,
        benchmark_scope="data-forward-backward-optimizer",
        micro_batch_semantics="explicit-provenance-not-equivalent-replay",
    )
    return evidence


def _benchmark(
    module_factory: ModuleFactory,
    data_module_factory: DataModuleFactory,
    *,
    seed: int,
    device: torch.device,
    candidate: Mapping[str, Any],
    expected_epochs: int,
    warmup_effective_batches: int,
    measure_effective_batches: int,
) -> Mapping[str, float | int]:
    seed_everything(seed, workers=True, verbose=False)
    data_module = data_module_factory(candidate)
    data_module.setup("fit")
    loader = data_module.train_dataloader()
    module = module_factory().to(device).train()
    optimizer = module.configure_optimizers()
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("execution calibration currently requires an optimizer without scheduler")
    accumulation = int(candidate["accumulate_grad_batches"])
    iterator: Iterator[Mapping[str, Any]] | None = None
    training_step = 0

    def effective_batch() -> int:
        nonlocal iterator, training_step
        examples_seen = 0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            if iterator is None:
                iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            transferred = {
                key: (
                    value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in batch.items()
            }
            if module.training_augmentation is not None:
                transferred = module.training_augmentation(
                    transferred,
                    seed=module.augmentation_seed,
                    epoch=0,
                    step=training_step,
                    identity=module.augmentation_identity,
                )
            values = module.validate_objective_result(
                module.objective(module.model, transferred, "train")
            )
            loss = values["loss"]
            if not isinstance(loss, torch.Tensor):
                raise TypeError("calibration objective loss must be a tensor")
            loss = loss / accumulation
            loss.backward()
            examples_seen += len(batch["sample_id"])
            training_step += 1
        optimizer.step()
        return examples_seen

    try:
        startup_started = time.perf_counter()
        iterator = iter(loader)
        warmup_examples = sum(effective_batch() for _ in range(warmup_effective_batches))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        startup_seconds = time.perf_counter() - startup_started
        prefetched_micro_batches = int(candidate.get("num_workers", 0)) * int(
            candidate.get("prefetch_factor", 2)
        )
        measured_batches = max(
            measure_effective_batches,
            ceil((prefetched_micro_batches + 2) / accumulation),
        )
        cuda_index = device.index if device.type == "cuda" else None
        with measure_phase("calibration", cuda_device=cuda_index) as telemetry:
            examples_seen = sum(effective_batch() for _ in range(measured_batches))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            with suppress(Exception):
                shutdown()

    elapsed = float(telemetry["elapsed_seconds"])
    cuda = telemetry["cuda"]
    steady_throughput = examples_seen / elapsed
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("execution calibration requires a sized training dataset")
    projected_examples = len(dataset) * expected_epochs
    projected_seconds = (
        startup_seconds * expected_epochs
        + max(projected_examples - warmup_examples, 0) / steady_throughput
    )
    peak_allocated = int(cuda["peak_allocated_bytes"] or 0)
    peak_reserved = int(cuda["peak_reserved_bytes"] or 0)
    result: dict[str, float | int] = {
        "examples_per_second": steady_throughput,
        "projected_examples_per_second": projected_examples / projected_seconds,
        "projected_total_seconds": projected_seconds,
        "projected_examples": projected_examples,
        "startup_seconds": startup_seconds,
        "elapsed_seconds": elapsed,
        "examples": examples_seen,
        "optimizer_steps": measured_batches,
        "micro_batches": measured_batches * accumulation,
        "peak_process_tree_rss_bytes": int(telemetry["peak_process_tree_rss_bytes"]),
        "peak_device_memory_bytes": max(peak_allocated, peak_reserved),
        "peak_device_allocated_bytes": peak_allocated,
        "peak_device_reserved_bytes": peak_reserved,
        "gpu_utilization_mean_percent": float(cuda["utilization_mean_percent"] or 0.0),
        "gpu_utilization_peak_percent": float(cuda["utilization_peak_percent"] or 0.0),
    }
    return result


def log_calibration(run_id: str, evidence: Mapping[str, Any]) -> None:
    """Record the complete calibration artifact and dashboard-friendly MLflow series.

    Consumes:
        An active MLflow run ID and the evidence returned by
        :func:`calibrate_training_execution`, whose trial measurements carry
        ``examples_per_second``, ``projected_examples_per_second``,
        ``peak_device_memory_bytes``, ``peak_process_tree_rss_bytes`` and
        ``gpu_utilization_mean_percent``.

    Produces:
        ``execution/calibration.json``, ``calibration.selected.*`` and
        ``calibration.environment_digest`` params, and one metric step per trial.

    Parameters:
        ``run_id``; ``evidence``.

    Devices:
        Device-independent.

    Limitations:
        Requires a writable MLflow run; it logs, it does not select.

    Example:
        >>> from mlflow import MlflowClient
        >>> from dsio.experimental.execution import calibrate_execution
        >>> client = MlflowClient()
        >>> run = client.create_run(client.create_experiment("calibration-example"))
        >>> def benchmark(candidate):  # the measurement shape training calibration records
        ...     return {
        ...         "examples_per_second": float(candidate["batch_size"]),
        ...         "projected_examples_per_second": float(candidate["batch_size"]),
        ...         "peak_device_memory_bytes": 0,
        ...         "peak_process_tree_rss_bytes": 0,
        ...         "gpu_utilization_mean_percent": 0.0,
        ...     }
        >>> evidence = calibrate_execution(
        ...     benchmark,
        ...     target_effective_batch_size=8,
        ...     candidates=[{"batch_size": 8}],
        ...     device="cpu",
        ... )
        >>> log_calibration(run.info.run_id, evidence)
        >>> client.get_run(run.info.run_id).data.params["calibration.selected.batch_size"]
        '8'
    """
    client = MlflowClient()
    client.log_dict(run_id, dict(evidence), "execution/calibration.json")
    selected = evidence["selected"]
    for name, value in selected.items():
        client.log_param(run_id, f"calibration.selected.{name}", value)
    client.log_param(run_id, "calibration.environment_digest", evidence["environment_digest"])
    selected_measurement = None
    for step, trial in enumerate(evidence["trials"]):
        measurement = trial.get("measurement")
        if measurement is None:
            continue
        candidate = trial["candidate"]
        for name in ("batch_size", "num_workers"):
            client.log_metric(
                run_id,
                f"calibration.trial.{name}",
                candidate[name],
                step=step,
            )
        for metric in (
            "examples_per_second",
            "projected_examples_per_second",
            "peak_device_memory_bytes",
            "peak_process_tree_rss_bytes",
            "gpu_utilization_mean_percent",
        ):
            client.log_metric(
                run_id,
                f"calibration.trial.{metric}",
                measurement[metric],
                step=step,
            )
        if candidate == selected:
            selected_measurement = measurement
    if selected_measurement is None:
        raise RuntimeError("selected calibration candidate has no measurement evidence")
    client.log_metric(
        run_id,
        "calibration.selected.projected_examples_per_second",
        selected_measurement["projected_examples_per_second"],
    )


def _candidates(target: int, device_type: str) -> list[dict[str, Any]]:
    batches = [value for value in (16, 64, 256, 1024) if value <= target and target % value == 0]
    if target not in batches:
        batches.append(target)
    cpu_count = os.cpu_count() or 1
    two_workers, eight_workers = min(2, cpu_count), min(8, cpu_count)
    pin_memory = device_type == "cuda"
    candidates = {
        (batch_size, num_workers, pin_memory, 2)
        for batch_size in batches
        for num_workers in {0, two_workers, eight_workers}
    }
    candidates.update(
        {
            (target, 0, False, 2),
            (target, two_workers, False, 2),
            (target, two_workers, pin_memory, 4),
            (target, eight_workers, False, 2),
            (target, eight_workers, pin_memory, 4),
        }
    )
    return [
        {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pinned,
            "prefetch_factor": prefetch_factor,
        }
        for batch_size, num_workers, pinned, prefetch_factor in sorted(candidates)
    ]


def _device(accelerator: str) -> torch.device:
    if accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA calibration requested but CUDA is unavailable")
        return torch.device("cuda", 0)
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "auto":
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    raise ValueError(f"execution calibration does not support accelerator {accelerator!r}")


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _available_host_memory_bytes() -> int:
    fields: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            fields[name] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    available = fields.get("MemAvailable")
    if available is None:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    return available


__all__ = ["calibrate_training_execution", "log_calibration"]
