"""Run equivalent storage-candidate workloads and emit raw JSON evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.storage.candidates import (
    CANDIDATES,
    build_candidate,
    candidate_path,
    measure_reads,
    recovery_observation,
)

PROFILES = ("smooth-signal", "fixed-items")


def synthetic_array(profile: str, *, rows: int, channels: int, seed: int) -> np.ndarray:
    """Create a deterministic workload with the declared shape and float32 dtype."""
    if rows < 1 or channels < 1:
        raise ValueError("rows and channels must be positive")
    rng = np.random.default_rng(seed)
    if profile == "smooth-signal":
        time = np.linspace(0, 80, rows, dtype=np.float32)[:, None]
        frequencies = np.arange(1, channels + 1, dtype=np.float32)[None, :]
        return np.asarray(
            np.sin(time * frequencies) + rng.normal(0, 0.03, (rows, channels)),
            dtype=np.float32,
            order="C",
        )
    if profile == "fixed-items":
        return np.asarray(
            rng.integers(-2048, 2048, size=(rows, channels), dtype=np.int16),
            dtype=np.float32,
            order="C",
        )
    raise ValueError(f"unknown synthetic profile {profile!r}; expected {', '.join(PROFILES)}")


def run_benchmark(
    work_root: Path,
    *,
    profiles: tuple[str, ...] = PROFILES,
    candidates: tuple[str, ...] = CANDIDATES,
    rows: int = 2_000_000,
    channels: int = 3,
    window: int = 500,
    reads: int = 2_000,
    workers: tuple[int, ...] = (1, 2, 4),
    seed: int = 42,
    real_sources: tuple[tuple[str, np.ndarray, dict[str, Any]], ...] = (),
) -> dict[str, Any]:
    """Run declared workloads and return a JSON-serializable evidence document."""
    _validate_inputs(profiles, candidates, rows, channels, window, reads, workers)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    workloads: list[dict[str, Any]] = []
    sources: list[tuple[str, str, np.ndarray, dict[str, Any]]] = [
        (
            profile,
            "synthetic",
            synthetic_array(profile, rows=rows, channels=channels, seed=seed),
            {"profile": profile},
        )
        for profile in profiles
    ]
    sources.extend(
        (name, "real", np.asarray(array, dtype=np.float32), details)
        for name, array, details in real_sources
    )

    for name, kind, source, origin in sources:
        if source.ndim != 2:
            raise ValueError(f"workload {name!r} has shape {source.shape}; expected a 2-D array")
        if len(source) <= window:
            raise ValueError(
                f"workload {name!r} has {len(source)} rows; it needs more than window={window}"
            )
        source = np.ascontiguousarray(source, dtype=np.float32)
        workload_root = work_root / _safe_name(name)
        workload_root.mkdir(parents=True, exist_ok=True)
        source_path = workload_root / "source.npy"
        np.save(source_path, source, allow_pickle=False)
        sequential = _sequential_starts(len(source), window, reads)
        random = _random_starts(len(source), window, reads, seed)
        results = [
            _run_candidate(
                candidate,
                source,
                source_path,
                workload_root,
                sequential,
                random,
                window,
                workers,
                seed,
            )
            for candidate in candidates
        ]
        _assert_equivalent(name, results)
        workloads.append(
            {
                "name": name,
                "kind": kind,
                "source": {
                    "shape": list(source.shape),
                    "dtype": str(source.dtype),
                    "bytes": source.nbytes,
                    "sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                    "origin": origin,
                },
                "candidates": results,
            }
        )

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "profiles": list(profiles),
            "candidates": list(candidates),
            "rows": rows,
            "channels": channels,
            "window": window,
            "reads": reads,
            "workers": list(workers),
            "seed": seed,
            "real_sources": [details for _, _, details in real_sources],
        },
        "environment": _environment(),
        "versions": _versions(candidates),
        "workloads": workloads,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _run_candidate(
    candidate: str,
    source: np.ndarray,
    source_path: Path,
    workload_root: Path,
    sequential: list[int],
    random: list[int],
    window: int,
    workers: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    candidate_root = workload_root / candidate
    candidate_root.mkdir(parents=True, exist_ok=True)
    path = candidate_root / candidate_path(candidate)
    build = _isolated(build_candidate, candidate, source_path, path)
    storage = build.pop("storage")
    configuration = build.pop("configuration")
    dependencies = build.pop("dependencies")
    shape = (int(source.shape[0]), int(source.shape[1]))
    dtype = str(source.dtype)
    sequential_result = _isolated(
        measure_reads,
        candidate,
        path,
        shape=shape,
        dtype=dtype,
        starts=sequential,
        window=window,
    )
    random_result = _isolated(
        measure_reads,
        candidate,
        path,
        shape=shape,
        dtype=dtype,
        starts=random,
        window=window,
    )
    multi_worker = [
        _measure_workers(
            candidate,
            path,
            shape=shape,
            dtype=dtype,
            window=window,
            reads=len(random),
            workers=count,
            seed=seed,
        )
        for count in workers
    ]
    recovery_rows = min(len(source), max(window + 1, 4096))
    recovery = recovery_observation(
        candidate,
        source[:recovery_rows],
        candidate_root / "recovery",
    )
    return {
        "candidate": candidate,
        "configuration": configuration,
        "dependencies": dependencies,
        "build": build,
        "storage": storage,
        "sequential": sequential_result,
        "random": random_result,
        "multi_worker": multi_worker,
        "recovery": recovery,
    }


def _measure_workers(
    candidate: str,
    path: Path,
    *,
    shape: tuple[int, int],
    dtype: str,
    window: int,
    reads: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    jobs = [
        _random_starts(shape[0], window, reads, seed + worker)
        for worker in range(workers)
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [
            pool.submit(
                measure_reads,
                candidate,
                path,
                shape=shape,
                dtype=dtype,
                starts=starts,
                window=window,
            )
            for starts in jobs
        ]
        raw = [future.result() for future in futures]
    elapsed = max(item["elapsed_seconds"] for item in raw)
    logical_bytes = sum(item["logical_bytes"] for item in raw)
    return {
        "workers": workers,
        "elapsed_seconds": elapsed,
        "windows": reads * workers,
        "logical_bytes": logical_bytes,
        "windows_per_second": reads * workers / elapsed,
        "throughput_bytes_per_second": logical_bytes / elapsed,
        "peak_rss_bytes_sum": sum(item["peak_rss_bytes"] for item in raw),
        "raw_workers": raw,
    }


def _isolated(function: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return pool.submit(function, *args, **kwargs).result()


def _sequential_starts(rows: int, window: int, reads: int) -> list[int]:
    if reads * window > rows:
        raise ValueError(
            f"sequential reads need {reads * window} rows, but the workload has {rows}"
        )
    return [int(value) for value in np.arange(reads, dtype=np.int64) * window]


def _random_starts(rows: int, window: int, reads: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, rows - window + 1, size=reads)]


def _assert_equivalent(workload: str, results: list[dict[str, Any]]) -> None:
    for operation in ("sequential", "random"):
        checksums = {result[operation]["checksum"] for result in results}
        if len(checksums) != 1:
            raise RuntimeError(
                f"candidate outputs differ for {workload!r} {operation} reads: {checksums}"
            )
    worker_counts = [item["workers"] for item in results[0]["multi_worker"]]
    for index, workers in enumerate(worker_counts):
        checksums = {
            tuple(worker["checksum"] for worker in result["multi_worker"][index]["raw_workers"])
            for result in results
        }
        if len(checksums) != 1:
            raise RuntimeError(
                f"candidate outputs differ for {workload!r} at {workers} workers"
            )


def _validate_inputs(
    profiles: tuple[str, ...],
    candidates: tuple[str, ...],
    rows: int,
    channels: int,
    window: int,
    reads: int,
    workers: tuple[int, ...],
) -> None:
    unknown_profiles = sorted(set(profiles) - set(PROFILES))
    unknown_candidates = sorted(set(candidates) - set(CANDIDATES))
    if unknown_profiles:
        raise ValueError(f"unknown synthetic profiles: {', '.join(unknown_profiles)}")
    if unknown_candidates:
        raise ValueError(f"unknown candidates: {', '.join(unknown_candidates)}")
    if min(rows, channels, window, reads, *workers) < 1:
        raise ValueError("rows, channels, window, reads, and worker counts must be positive")
    if rows <= window:
        raise ValueError("rows must be greater than window")


def _versions(candidates: tuple[str, ...]) -> dict[str, str]:
    packages = {"numpy"}
    if "arrow-ipc" in candidates:
        packages.add("pyarrow")
    if "zarr-v3" in candidates:
        packages.update(("numcodecs", "zarr"))
    return {name: importlib.metadata.version(name) for name in sorted(packages)}


def _environment() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_commit": commit,
        "git_dirty": dirty,
        "git_dirty_scope": "tracked files only",
        "rss_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "cache_state": "uncontrolled; reads may use the operating-system page cache",
    }


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    )
