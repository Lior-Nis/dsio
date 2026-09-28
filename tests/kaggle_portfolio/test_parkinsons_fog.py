from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from mlflow import MlflowClient
from tests.kaggle_portfolio.assertions import (
    assert_downstream_evidence,
    assert_execution_evidence,
    assert_replay_run_ids_differ,
)

from dsio.contracts import sha256_of_bytes
from dsio.data.loading import DsioDataModule
from dsio.model.module import DsioModule


def test_parkinsons_scale_inventory_resolves_official_ids_and_rejects_gaps(
    parkinsons_fog_csvs: Path,
    tmp_path: Path,
) -> None:
    from reference_projects.kaggle.parkinsons_fog.scale.inventory import (
        resolve_official_inventory,
    )

    labelled = tmp_path / "central-labelled"
    defog = labelled / "defog" / "sessions"
    tdcsfog = labelled / "tdcsfog" / "sessions"
    notype = labelled / "notype" / "sessions"
    for directory in (defog, tdcsfog, notype):
        directory.mkdir(parents=True)

    for source in sorted((parkinsons_fog_csvs / "train" / "defog").glob("*.csv")):
        shutil.copyfile(source, defog / source.name)
    for source in sorted((parkinsons_fog_csvs / "train" / "tdcsfog").glob("*.csv")):
        shutil.copyfile(source, tdcsfog / source.name)
    shutil.copyfile(
        parkinsons_fog_csvs / "test" / "defog" / "test-defog.csv",
        notype / "test-defog.csv",
    )
    shutil.copyfile(next(defog.glob("*.csv")), defog / "unrelated-extra.csv")
    shutil.copyfile(next(tdcsfog.glob("*.csv")), tdcsfog / "unrelated-extra.csv")

    daily = tmp_path / "daily"
    daily.mkdir()
    (daily / "daily0.parquet").write_bytes(b"parquet-zero")
    (daily / "daily1.parquet").write_bytes(b"parquet-one")
    (parkinsons_fog_csvs / "daily_metadata.csv").write_text(
        "Id,Subject,Visit,Beginning of recording [00:00-23:59]\n"
        "daily0,daily-subject-0,1,08:00\n"
        "daily1,daily-subject-1,1,09:00\n"
    )

    inventory = resolve_official_inventory(
        parkinsons_fog_csvs,
        labelled,
        daily,
        expected_counts={
            "train_defog": 4,
            "train_tdcsfog": 4,
            "notype": 1,
            "daily": 2,
            "test": 2,
        },
    )

    assert [item["recording_id"] for item in inventory["train_defog"]] == [
        "defog0",
        "defog1",
        "defog2",
        "defog3",
    ]
    assert len(inventory["train_tdcsfog"]) == 4
    assert [item["recording_id"] for item in inventory["notype"]] == ["test-defog"]
    assert [item["recording_id"] for item in inventory["daily"]] == ["daily0", "daily1"]
    assert len(inventory["test"]) == 2
    assert inventory["extras"] == {
        "train_defog": ["unrelated-extra"],
        "train_tdcsfog": ["unrelated-extra"],
        "notype": [],
        "daily": [],
    }
    assert inventory["manifest_digest"]
    assert all(
        isinstance(item["mtime_ns"], int)
        for lane in ("train_defog", "train_tdcsfog", "notype", "daily", "test")
        for item in inventory[lane]
    )

    with pytest.raises(ValueError, match="expected_counts.*exactly"):
        resolve_official_inventory(
            parkinsons_fog_csvs,
            labelled,
            daily,
            expected_counts={
                "train_defog": 4,
                "train_tdcsfog": 4,
                "notype": 1,
                "daily": 2,
            },
        )
    with pytest.raises(ValueError, match="expected_counts.*positive integer.*daily"):
        resolve_official_inventory(
            parkinsons_fog_csvs,
            labelled,
            daily,
            expected_counts={
                "train_defog": 4,
                "train_tdcsfog": 4,
                "notype": 1,
                "daily": True,
                "test": 2,
            },
        )

    from reference_projects.kaggle.parkinsons_fog.data import (
        iter_recording_windows,
        load_competition_data,
    )

    loaded = load_competition_data(parkinsons_fog_csvs, source_inventory=inventory)
    assert len(loaded["train"]) == 8
    assert all(str(labelled) in recording["path"] for recording in loaded["train"])
    assert len(loaded["test"]) == 2
    changed_recording = loaded["train"][0]
    changed_path = Path(changed_recording["path"])
    changed_stat = changed_path.stat()
    os.utime(
        changed_path,
        ns=(changed_stat.st_atime_ns, changed_stat.st_mtime_ns + 1_000_000_000),
    )
    with pytest.raises(ValueError, match="changed since inventory"):
        next(iter_recording_windows(changed_recording))

    shutil.copyfile(next(defog.glob("*.csv")), notype / "defog0.csv")
    with pytest.raises(ValueError, match="ambiguous.*defog0"):
        resolve_official_inventory(
            parkinsons_fog_csvs,
            labelled,
            daily,
            expected_counts={
                "train_defog": 4,
                "train_tdcsfog": 4,
                "notype": 1,
                "daily": 2,
                "test": 2,
            },
        )
    (notype / "defog0.csv").unlink()

    metadata = parkinsons_fog_csvs / "daily_metadata.csv"
    metadata.write_text(metadata.read_text() + "daily0,daily-subject-0,1,08:00\n")
    with pytest.raises(ValueError, match="daily_metadata.csv.*duplicate identity"):
        resolve_official_inventory(
            parkinsons_fog_csvs,
            labelled,
            daily,
            expected_counts={
                "train_defog": 4,
                "train_tdcsfog": 4,
                "notype": 1,
                "daily": 2,
                "test": 2,
            },
        )
    metadata.write_text(
        "Id,Subject,Visit,Beginning of recording [00:00-23:59]\n"
        "daily0,daily-subject-0,1,08:00\n"
        "daily1,daily-subject-1,1,09:00\n"
    )
    metadata.write_text(
        "Id,Subject,Visit,Beginning of recording [00:00-23:59]\n"
        "daily0, ,1,08:00\n"
        "daily1,daily-subject-1,1,09:00\n"
    )
    with pytest.raises(ValueError, match="daily_metadata.csv.*row 2.*Subject"):
        resolve_official_inventory(
            parkinsons_fog_csvs,
            labelled,
            daily,
            expected_counts={
                "train_defog": 4,
                "train_tdcsfog": 4,
                "notype": 1,
                "daily": 2,
                "test": 2,
            },
        )
    metadata.write_text(
        "Id,Subject,Visit,Beginning of recording [00:00-23:59]\n"
        "daily0,daily-subject-0,1,08:00\n"
        "daily1,daily-subject-1,1,09:00\n"
    )
    metadata.write_text(
        "Id,Subject,Visit,Beginning of recording [00:00-23:59]\n"
        "daily0,daily-subject-0,1\n"
        "daily1,daily-subject-1,1,09:00\n"
    )
    with pytest.raises(ValueError, match="daily_metadata.csv.*row 2.*ragged"):
        resolve_official_inventory(
            parkinsons_fog_csvs,
            labelled,
            daily,
            expected_counts={
                "train_defog": 4,
                "train_tdcsfog": 4,
                "notype": 1,
                "daily": 2,
                "test": 2,
            },
        )
    metadata.write_text(
        "Id,Subject,Visit,Beginning of recording [00:00-23:59]\n"
        "daily0,daily-subject-0,1,08:00\n"
        "daily1,daily-subject-1,1,09:00\n"
    )

    tdcs_test = parkinsons_fog_csvs / "test" / "tdcsfog" / "shared.csv"
    tdcs_test_bytes = tdcs_test.read_bytes()
    tdcs_test.unlink()
    second_defog_test = parkinsons_fog_csvs / "test" / "defog" / "defog0.csv"
    shutil.copyfile(
        parkinsons_fog_csvs / "test" / "defog" / "test-defog.csv",
        second_defog_test,
    )
    with pytest.raises(ValueError, match="test.*kinds.*tdcsfog"):
        resolve_official_inventory(
            parkinsons_fog_csvs,
            labelled,
            daily,
            expected_counts={
                "train_defog": 4,
                "train_tdcsfog": 4,
                "notype": 1,
                "daily": 2,
                "test": 2,
            },
        )
    second_defog_test.unlink()
    tdcs_test.write_bytes(tdcs_test_bytes)

    (daily / "daily1.parquet").unlink()
    with pytest.raises(ValueError, match="daily.*daily1.*missing"):
        resolve_official_inventory(
            parkinsons_fog_csvs,
            labelled,
            daily,
            expected_counts={
                "train_defog": 4,
                "train_tdcsfog": 4,
                "notype": 1,
                "daily": 2,
                "test": 2,
            },
        )


def test_parkinsons_scale_scan_is_bounded_deterministic_and_strict(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.parkinsons_fog.scale.scanning import scan_non_supervised

    notype = tmp_path / "notype.csv"
    notype.write_text(
        "Time,AccV,AccML,AccAP,Event,Valid,Task,StartHesitation,Turn,Walking\n"
        "0,1.0,2.0,3.0,0,False,False,0,0,0\n"
        "1,1.5,2.5,3.5,1,True,True,1,0,0\n"
        "2,2.0,3.0,4.0,0,True,True,0,1,0\n"
    )
    daily = tmp_path / "daily.parquet"
    pq.write_table(
        pa.table(
            {
                "Time": pa.array([0, 1, 2, 3, 4], type=pa.int64()),
                "AccV": pa.array([1.0, 1.1, 1.2, 1.3, 1.4], type=pa.float64()),
                "AccML": pa.array([2.0, 2.1, 2.2, 2.3, 2.4], type=pa.float64()),
                "AccAP": pa.array([3.0, 3.1, 3.2, 3.3, 3.4], type=pa.float64()),
            }
        ),
        daily,
        row_group_size=3,
    )
    inventory = {
        "notype": [
            {
                "lane": "notype",
                "kind": "notype",
                "recording_id": "notype0",
                "path": str(notype),
                "bytes": notype.stat().st_size,
                "mtime_ns": notype.stat().st_mtime_ns,
            }
        ],
        "daily": [
            {
                "lane": "daily",
                "kind": "daily",
                "recording_id": "daily0",
                "path": str(daily),
                "bytes": daily.stat().st_size,
                "mtime_ns": daily.stat().st_mtime_ns,
            }
        ],
    }

    with pytest.raises(ValueError, match="notype.*must not be empty"):
        scan_non_supervised({**inventory, "notype": []}, batch_rows=2)
    with pytest.raises(ValueError, match="notype.*duplicate.*notype0"):
        scan_non_supervised(
            {**inventory, "notype": [*inventory["notype"], *inventory["notype"]]},
            batch_rows=2,
        )

    first = scan_non_supervised(inventory, batch_rows=2)
    second = scan_non_supervised(inventory, batch_rows=3)

    assert first["files"] == 2
    assert first["rows"] == 8
    assert first["source_bytes"] == notype.stat().st_size + daily.stat().st_size
    assert first["max_batch_rows"] == 2
    assert first["checksum"] == second["checksum"]
    assert first["lanes"] == {
        "notype": {"files": 1, "rows": 3, "source_bytes": notype.stat().st_size},
        "daily": {"files": 1, "rows": 5, "source_bytes": daily.stat().st_size},
    }

    notype.write_text("wrong,columns\n1,2\n")
    inventory["notype"][0]["bytes"] = notype.stat().st_size
    inventory["notype"][0]["mtime_ns"] = notype.stat().st_mtime_ns
    with pytest.raises(ValueError, match="notype0.*columns"):
        scan_non_supervised(inventory, batch_rows=2)
    stale_stat = notype.stat()
    os.utime(notype, ns=(stale_stat.st_atime_ns, stale_stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="notype.*notype0.*changed since inventory"):
        scan_non_supervised(inventory, batch_rows=2)


def test_parkinsons_scale_telemetry_records_resources_and_refuses_cuda_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dsio.experimental.telemetry import measure_phase

    for interval in (True, 0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="sample_interval_seconds.*finite positive number"):
            with measure_phase("scan", sample_interval_seconds=interval):
                pass
    with pytest.raises(RuntimeError, match="process telemetry.*unavailable"):
        with measure_phase("scan", proc_root=tmp_path / "missing-proc"):
            pass

    with measure_phase("scan", sample_interval_seconds=0.001) as evidence:
        values = np.ones((1024, 1024), dtype=np.float32)
        assert float(values.sum()) == 1024 * 1024

    assert evidence["phase"] == "scan"
    assert evidence["elapsed_seconds"] > 0
    assert evidence["peak_process_tree_rss_bytes"] > 0
    assert evidence["measurement_method"] == {
        "elapsed": "time.perf_counter",
        "process_tree_rss": "linux-proc-status-vmrss-summed-all-thread-children",
        "cuda_memory": "not_applicable",
        "gpu_utilization": "not_applicable",
    }
    assert evidence["cache_state"] == "uncontrolled-os-page-cache"
    assert evidence["cuda"] == {
        "requested": False,
        "device": None,
        "uuid": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "utilization_samples": 0,
        "utilization_mean_percent": None,
        "utilization_peak_percent": None,
    }

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable.*will not fall back"):
        with measure_phase("training", cuda_device=0):
            pass

    from contextlib import nullcontext
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(uuid="test-uuid"),
    )

    def missing_nvidia_smi(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing_nvidia_smi)
    with pytest.raises(RuntimeError, match="CUDA telemetry.*nvidia-smi"):
        with measure_phase("training", cuda_device=0, sample_interval_seconds=0.001):
            threading.Event().wait(0.01)
    assert not any(
        thread.name == "parkinsons-training-telemetry" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_parkinsons_scale_rejects_invalid_resource_configuration_before_work(
    tmp_path: Path,
    kaggle_services: None,
) -> None:
    del kaggle_services
    from reference_projects.kaggle.parkinsons_fog.tasks.data import scan_scale_sources

    from dsio.tracking import resolve_experiment

    experiment_id = resolve_experiment("parkinsons-invalid-scale-config").experiment_id
    missing = str(tmp_path / "must-not-be-read")
    with pytest.raises(ValueError, match="memory_budget_bytes.*positive integer"):
        scan_scale_sources.fn(
            missing,
            missing,
            missing,
            experiment_id,
            memory_budget_bytes=True,
        )


def test_parkinsons_telemetry_counts_children_spawned_by_non_main_threads() -> None:
    from dsio.experimental.telemetry import measure_phase

    baseline = 0
    for line in Path(f"/proc/{os.getpid()}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            baseline = int(line.split()[1]) * 1024
            break
    assert baseline > 0

    ready = threading.Event()
    child: dict[str, subprocess.Popen[str]] = {}

    def launch_child() -> None:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys; payload=bytearray(96*1024*1024); "
                    "[payload.__setitem__(i, 1) for i in range(0, len(payload), 4096)]; "
                    "print('ready', flush=True); sys.stdin.read(1)"
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        child["process"] = process
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        ready.set()
        process.wait(timeout=10)

    launcher = threading.Thread(target=launch_child, name="parkinsons-child-launcher")
    launcher.start()
    assert ready.wait(timeout=10)
    process = child["process"]
    try:
        with measure_phase("thread-child", sample_interval_seconds=0.01) as evidence:
            threading.Event().wait(0.1)
    finally:
        assert process.stdin is not None
        process.stdin.write("x")
        process.stdin.flush()
        launcher.join(timeout=10)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
    assert not launcher.is_alive()
    assert evidence["peak_process_tree_rss_bytes"] > baseline + 64 * 1024**2


def test_parkinsons_scale_flow_tracks_scan_and_explicit_training_configuration(
    parkinsons_fog_csvs: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    import pyarrow as pa
    import pyarrow.parquet as pq
    import reference_projects.kaggle.parkinsons_fog.scale.inventory as scale_inventory
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.parkinsons_fog.flow import parkinsons_fog_flow

    from dsio.train.trainer import TrainerConfig

    labelled = tmp_path / "central-labelled"
    for kind in ("defog", "tdcsfog", "notype"):
        (labelled / kind / "sessions").mkdir(parents=True)
    for kind in ("defog", "tdcsfog"):
        for source in (parkinsons_fog_csvs / "train" / kind).glob("*.csv"):
            shutil.copyfile(source, labelled / kind / "sessions" / source.name)
    notype = labelled / "notype" / "sessions" / "test-defog.csv"
    notype.write_text(
        "Time,AccV,AccML,AccAP,Event,Valid,Task,StartHesitation,Turn,Walking\n"
        "0,1.0,2.0,3.0,0,False,False,0,0,0\n"
        "1,1.5,2.5,3.5,1,True,True,1,0,0\n"
    )
    daily = tmp_path / "daily"
    daily.mkdir()
    pq.write_table(
        pa.table(
            {
                "Time": pa.array([0, 1, 2], type=pa.int64()),
                "AccV": pa.array([1.0, 1.1, 1.2], type=pa.float64()),
                "AccML": pa.array([2.0, 2.1, 2.2], type=pa.float64()),
                "AccAP": pa.array([3.0, 3.1, 3.2], type=pa.float64()),
            }
        ),
        daily / "daily0.parquet",
    )
    (parkinsons_fog_csvs / "daily_metadata.csv").write_text(
        "Id,Subject,Visit,Beginning of recording [00:00-23:59]\n"
        "daily0,daily-subject,1,08:00\n"
    )
    monkeypatch.setattr(
        scale_inventory,
        "OFFICIAL_COUNTS",
        {"train_defog": 4, "train_tdcsfog": 4, "notype": 1, "daily": 1, "test": 2},
    )
    trainer_config = TrainerConfig(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        deterministic=True,
        checkpoint=False,
        log_every_n_steps=1,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
    )

    with prefect_test_harness():
        result = parkinsons_fog_flow(
            str(parkinsons_fog_csvs),
            str(tmp_path / "work"),
            seed=17,
            labelled_root=str(labelled),
            daily_root=str(daily),
            trainer_config=trainer_config,
            execution_calibration={
                "target_effective_batch_size": 2,
                "candidates": [
                    {"batch_size": 1, "num_workers": 0, "pin_memory": False},
                    {"batch_size": 2, "num_workers": 0, "pin_memory": False},
                ],
                "warmup_effective_batches": 1,
                "measure_effective_batches": 1,
            },
            scan_batch_rows=2,
            scan_memory_budget_bytes=sys.maxsize,
        )

    assert result["scale"]["scan_run_id"]
    assert result["scale"]["manifest_digest"]
    assert result["scale"]["scan"]["files"] == 2
    assert result["scale"]["scan"]["rows"] == 5
    assert result["scale"]["scan"]["max_batch_rows"] == 2
    assert result["scale"]["scan_telemetry"]["peak_process_tree_rss_bytes"] > 0
    assert result["ingest_telemetry"]["elapsed_seconds"] > 0
    assert result["training_telemetry"]["elapsed_seconds"] > 0
    calibration = result["execution_calibration"]
    assert calibration["selected"]["batch_size"] in {1, 2}
    assert (
        calibration["selected"]["batch_size"]
        * calibration["selected"]["accumulate_grad_batches"]
        == 2
    )
    assert len(calibration["trials"]) == 2
    run = MlflowClient().get_run(result["scale"]["scan_run_id"])
    assert run.data.metrics["scale.scan.files"] == 2
    assert run.data.metrics["scale.scan.rows"] == 5
    train_run = MlflowClient().get_run(result["train_run_id"])
    assert int(train_run.data.params["calibration.selected.batch_size"]) in {1, 2}
    assert train_run.data.metrics["calibration.selected.projected_examples_per_second"] > 0
    assert train_run.data.metrics["calibration.trial.batch_size"] in {1, 2}
    assert any(
        artifact.path == "execution/calibration.json"
        for artifact in MlflowClient().list_artifacts(result["train_run_id"], "execution")
    )


def test_parkinsons_boundary_streams_windows_and_excludes_test_subjects(
    parkinsons_fog_csvs: Path,
) -> None:
    from reference_projects.kaggle.parkinsons_fog.data import (
        WINDOW_SIZE,
        iter_recording_windows,
        load_competition_data,
    )

    loaded = load_competition_data(parkinsons_fog_csvs)

    assert loaded["excluded_train_ids"] == ["same-subject", "shared"]
    assert loaded["test_subjects"] == ["held-out-subject", "test-subject"]
    assert loaded["submission_ids"][:2] == ["shared_0", "shared_1"]
    recording = next(value for value in loaded["train"] if value["recording_id"] == "defog0")
    windows = list(iter_recording_windows(recording, window_size=WINDOW_SIZE))
    assert sum(len(window["data"]) for window in windows) == 17
    assert max(len(window["data"]) for window in windows) <= WINDOW_SIZE
    mask = np.concatenate([window["data"][:, -1] for window in windows])
    assert mask.tolist() == [0.0, 0.0, *([1.0] * 13), 0.0, 0.0]

    (parkinsons_fog_csvs / "tdcsfog_metadata.csv").unlink()
    with pytest.raises(ValueError, match="metadata"):
        load_competition_data(parkinsons_fog_csvs)


def test_parkinsons_evaluation_rejects_malformed_dense_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import reference_projects.kaggle.parkinsons_fog.tasks.downstream as downstream
    from reference_projects.kaggle.parkinsons_fog.components import validate_fog_prediction
    from reference_projects.kaggle.parkinsons_fog.data import WINDOW_SIZE

    probability = torch.tensor([[[0.7, 0.2, 0.8]]])
    with pytest.raises(ValueError, match="thresholded"):
        validate_fog_prediction(
            {"prediction": torch.zeros_like(probability), "probability": probability}
        )

    class FakeStore:
        sample = np.zeros((1, 7), dtype=np.float32)

        def __init__(self, path: str) -> None:
            del path

        def read_sample(self, sample_id: str) -> dict[str, object]:
            del sample_id
            return {"data": self.sample}

    monkeypatch.setattr(downstream, "SignalStore", FakeStore)
    FakeStore.sample = np.zeros((1, 7), dtype=np.float32)
    FakeStore.sample[0, 3] = np.nan
    with pytest.raises(ValueError, match="labels.*finite binary"):
        downstream._targets_and_mask("unused", ["bad-label"])

    FakeStore.sample = np.zeros((1, 7), dtype=np.float32)
    FakeStore.sample[0, -1] = np.nan
    with pytest.raises(ValueError, match="mask.*zero or one"):
        downstream._targets_and_mask("unused", ["bad-mask"])

    FakeStore.sample = np.zeros((WINDOW_SIZE + 1, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="maximum"):
        downstream._targets_and_mask("unused", ["too-long"])


def test_parkinsons_flow_trains_masked_dense_prediction_with_subject_split(
    parkinsons_fog_csvs: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    from lightning import Trainer
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.parkinsons_fog.flow import parkinsons_fog_flow

    observed: list[tuple[type[object], type[object], bool]] = []
    original = Trainer.fit

    def record(trainer: Trainer, model: object, *args: object, **kwargs: object) -> object:
        datamodule = kwargs["datamodule"]
        result = original(trainer, model, *args, **kwargs)
        batch = next(iter(datamodule.train_dataloader()))
        observed.append((type(model), type(datamodule), bool((~batch["mask"]).any())))
        return result

    monkeypatch.setattr(Trainer, "fit", record)

    with prefect_test_harness():
        result = parkinsons_fog_flow(str(parkinsons_fog_csvs), str(tmp_path / "work"), seed=17)
        replay = parkinsons_fog_flow(str(parkinsons_fog_csvs), str(tmp_path / "work"), seed=17)

    assert result["split_algorithm"] == "group_shuffle"
    assert result["split_parameters"] == {"test_size": 0.34}
    assert set(result["split_groups"]["train"]).isdisjoint(result["split_groups"]["validate"])
    assigned = [
        sample_id for role in ("train", "validate") for sample_id in result["assignments"][role]
    ]
    assert not any("shared" in value or "same-subject" in value for value in assigned)
    assert observed == [
        (DsioModule, DsioDataModule, True),
        (DsioModule, DsioDataModule, True),
    ]
    assert result["prediction_count"] == 20
    assert np.isfinite(result["prediction"]).all()
    assert all(0 <= value <= 1 for row in result["prediction"] for value in row)
    assert set(result["metrics"]) == {
        "average_precision.StartHesitation",
        "average_precision.Turn",
        "average_precision.Walking",
        "average_precision.mean",
        "positive_rate.StartHesitation",
        "positive_rate.Turn",
        "positive_rate.Walking",
        "positive_rate.mean",
    }
    assert all(np.isfinite(value) for value in result["metrics"].values())
    assert result["metrics"]["average_precision.mean"] >= result["metrics"]["positive_rate.mean"]
    assert result["ignored_points"] > 0
    assert_replay_run_ids_differ(result, replay)
    assert result["split_digest"] == replay["split_digest"]
    assert result["identities"] == replay["identities"]
    assert result["metrics"] == replay["metrics"]
    assert result["prediction"] == replay["prediction"]
    assert result["submission_bytes"] == replay["submission_bytes"]
    lines = result["submission_bytes"].decode().splitlines()
    assert lines[0] == "Id,StartHesitation,Turn,Walking"
    assert [line.split(",", 1)[0] for line in lines[1:]] == [
        *[f"shared_{index}" for index in range(9)],
        *[f"test-defog_{index}" for index in range(11)],
    ]
    assert result["submission_digest"] == sha256_of_bytes(result["submission_bytes"])
    run = MlflowClient().get_run(result["evaluation_run_id"])
    for name, value in result["metrics"].items():
        assert run.data.metrics[name] == pytest.approx(value)
    assert run.data.params["evaluation.masked"] == "true"
    assert run.data.params["evaluation.target_names"] == ('["StartHesitation", "Turn", "Walking"]')
    assert_execution_evidence(
        result["train_run_id"],
        tmp_path / "parkinsons-provenance",
        optimizer="torch.optim:Adam",
        optimizer_parameters={"lr": 0.001},
        batch_size=16,
        num_workers=2,
    )
    assert_downstream_evidence(result, tmp_path / "parkinsons-downstream")
