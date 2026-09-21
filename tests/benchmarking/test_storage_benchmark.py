from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

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
        assert result["sequential"]["checksum"]
        assert result["random"]["checksum"]
        assert {item["workers"] for item in result["multi_worker"]} == {1, 2}
        assert result["recovery"]["published"] is False
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
    assert set(report["versions"]) == {"numpy", "pyarrow", "zarr"}
    results = report["workloads"][0]["candidates"]
    assert len({result["sequential"]["checksum"] for result in results}) == 1
    assert len({result["random"]["checksum"] for result in results}) == 1
