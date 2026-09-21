from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dsio.data.store import SignalStore


def test_synthetic_profiles_are_seeded_and_distinct() -> None:
    code = """
import hashlib
import json
from benchmarks.storage.benchmark import synthetic_array

def digest(profile):
    value = synthetic_array(profile, rows=64, channels=3, seed=7)
    return [str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]

print(json.dumps([digest('smooth-signal'), digest('fixed-items'), digest('smooth-signal')]))
"""
    values = json.loads(
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert values[0] == values[2]
    assert values[0][:2] == ["float32", [64, 3]]
    assert values[0][2] != values[1][2]


def test_report_records_inputs_environment_raw_measurements_and_recovery(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.storage",
            "--quick",
            "--candidates",
            "flat-binary",
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads(output.read_text())

    assert report["schema_version"] == 1
    assert report["inputs"] == {
        "profiles": ["smooth-signal", "fixed-items"],
        "candidates": ["flat-binary"],
        "rows": 4096,
        "channels": 3,
        "window": 64,
        "reads": 16,
        "workers": [1, 2],
        "seed": 42,
        "real_sources": [],
    }
    assert report["environment"]["python"]
    assert report["environment"]["platform"]
    assert report["environment"]["cpu_count"]
    assert report["versions"]["numpy"]

    for workload in report["workloads"]:
        assert workload["source"]["sha256"]
        result = workload["candidates"][0]
        assert result["candidate"] == "flat-binary"
        assert result["build"]["elapsed_seconds"] >= 0
        assert result["build"]["throughput_bytes_per_second"] > 0
        assert result["storage"]["bytes"] > 0
        assert result["storage"]["files"] == 1
        assert result["sequential"]["checksum"] == workload["expected_checksums"][
            "sequential"
        ]
        assert result["random"]["checksum"] == workload["expected_checksums"]["random"]
        assert {item["workers"] for item in result["multi_worker"]} == {1, 2}
        for measurement in result["multi_worker"]:
            assert len(set(measurement["pids"])) == measurement["workers"]
            assert measurement["elapsed_seconds"] > 0
        assert result["recovery"]["published"] is False
        assert result["recovery"]["partial_accepted_as_complete"] is False
        assert result["recovery"]["retry_succeeded"] is True
def test_production_store_has_no_backend_selector() -> None:
    assert "backend" not in inspect.signature(SignalStore).parameters
    assert "backend" not in inspect.signature(SignalStore.open).parameters


def test_all_candidates_return_identical_logical_reads(tmp_path: Path) -> None:
    output = tmp_path / "all-candidates.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.storage",
            "--quick",
            "--profiles",
            "smooth-signal",
            "--output",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text())
    assert set(report["versions"]) == {"numcodecs", "numpy", "pyarrow", "zarr"}
    results = report["workloads"][0]["candidates"]
    expected = report["workloads"][0]["expected_checksums"]
    assert {result["sequential"]["checksum"] for result in results} == {
        expected["sequential"]
    }
    assert {result["random"]["checksum"] for result in results} == {expected["random"]}
    recovery = {result["candidate"]: result["recovery"] for result in results}
    assert recovery["flat-binary"]["partial_opened"] is False
    assert recovery["arrow-ipc"]["partial_opened"] is False
    assert recovery["zarr-v3"]["partial_opened"] is True
    assert all(
        observation["partial_accepted_as_complete"] is False
        for observation in recovery.values()
    )


def test_two_dimensional_real_source_is_limited_before_benchmarking(tmp_path: Path) -> None:
    import zarr

    source = tmp_path / "source.zarr"
    zarr.create_array(
        store=str(source),
        name="accs",
        data=np.arange(128 * 3, dtype=np.float32).reshape(128, 3),
        zarr_format=3,
    )
    output = tmp_path / "real.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.storage",
            "--real-only",
            "--real-zarr",
            str(source),
            "--real-row-limit",
            "32",
            "--window",
            "8",
            "--reads",
            "4",
            "--workers",
            "1",
            "--candidates",
            "flat-binary",
            "--output",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["workloads"][0]["source"]["shape"] == [32, 3]
    assert report["inputs"]["real_sources"][0]["row_limit"] == 32


def test_reused_work_root_fails_with_an_actionable_error(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    work_root.mkdir()
    (work_root / "existing").write_text("do not overwrite")
    output = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.storage",
            "--quick",
            "--candidates",
            "flat-binary",
            "--work-root",
            str(work_root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "work root must be empty" in completed.stderr
    assert (work_root / "existing").read_text() == "do not overwrite"


def test_zarr_recovery_handles_fill_only_arrays(tmp_path: Path) -> None:
    import zarr

    source = tmp_path / "zeros.zarr"
    zarr.create_array(
        store=str(source),
        name="accs",
        data=np.zeros((128, 3), dtype=np.float32),
        zarr_format=3,
    )
    output = tmp_path / "zeros.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.storage",
            "--real-only",
            "--real-zarr",
            str(source),
            "--window",
            "8",
            "--reads",
            "4",
            "--workers",
            "1",
            "--candidates",
            "zarr-v3",
            "--output",
            str(output),
        ],
        check=True,
    )

    recovery = json.loads(output.read_text())["workloads"][0]["candidates"][0]["recovery"]
    assert recovery["partial_opened"] is False
    assert recovery["partial_accepted_as_complete"] is False
    assert recovery["retry_succeeded"] is True
