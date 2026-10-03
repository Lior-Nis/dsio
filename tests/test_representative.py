"""The representative-tier runner records and compares parity faithfully."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tools import representative


def test_compare_is_exact_by_default_and_reports_moved_and_missing_metrics() -> None:
    baseline = {"accuracy": 0.5, "rmse": 1.0, "dropped": 2.0}
    observed = {"accuracy": 0.5, "rmse": 1.0 + 1e-12, "added": 3.0}

    moved = representative.compare(baseline, observed, tolerance=0.0)

    assert moved == {
        "added": (None, 3.0),
        "dropped": (2.0, None),
        "rmse": (1.0, 1.0 + 1e-12),
    }
    assert "rmse" not in representative.compare(baseline, observed, tolerance=1e-9)


def test_hardware_class_requires_the_recorded_device_and_thread_count() -> None:
    baseline = {"cpu": "AMD Ryzen", "cuda": None, "threads": 12}
    representative._require_hardware_class(baseline, dict(baseline))

    with pytest.raises(ValueError, match="hardware class"):
        representative._require_hardware_class(
            baseline, {"cpu": "AMD Ryzen", "cuda": None, "threads": 24}
        )
    with pytest.raises(ValueError, match="hardware class"):
        representative._require_hardware_class(
            baseline, {"cpu": "AMD Ryzen", "cuda": "GPU", "threads": 12}
        )


def test_ui_uri_is_explicit_and_cannot_persist_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_UI_URI", "https://mlflow.example.test/base/")
    assert representative._ui_uri("http://localhost:5000") == (
        "https://mlflow.example.test/base"
    )

    monkeypatch.setenv("MLFLOW_UI_URI", "https://user:secret@mlflow.example.test")
    with pytest.raises(ValueError, match="credentials"):
        representative._ui_uri("http://localhost:5000")


def test_single_and_multi_model_results_flatten_to_one_record() -> None:
    single = {"metrics": {"rmse": 2}, "train_run_id": "t", "evaluation_run_id": "e", "seed": 19}
    multi = {
        "data_run_id": "d",
        "models": {
            "fused": {"metrics": {"accuracy": 0.25}, "evaluation_run_id": "ef"},
            "tabular": {"metrics": {"accuracy": 0.5}, "evaluation_run_id": "et"},
        },
    }

    assert representative._metrics(single) == {"rmse": 2.0}
    assert representative._run_ids(single) == {
        "evaluation_run_id": "e",
        "train_run_id": "t",
    }
    assert representative._metrics(multi) == {"fused.accuracy": 0.25, "tabular.accuracy": 0.5}
    assert representative._run_ids(multi) == {
        "data_run_id": "d",
        "fused.evaluation_run_id": "ef",
        "tabular.evaluation_run_id": "et",
    }
    assert representative._evaluation_run_ids(
        {"runs": representative._run_ids(multi)}
    ) == {
        "fused.evaluation_run_id": "ef",
        "tabular.evaluation_run_id": "et",
    }


def test_tracking_requires_a_reachable_http_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        representative._require_live_tracking("file:///tmp/mlruns")
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        representative._require_live_tracking("")

    class UnreachableClient:
        def search_experiments(self, *, max_results: int) -> None:
            assert max_results == 1
            raise ConnectionError("refused")

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda _: UnreachableClient())
    with pytest.raises(RuntimeError, match="not reachable"):
        representative._require_live_tracking("https://mlflow.example.test")


def test_run_ids_become_immutable_server_uris() -> None:
    class Client:
        def get_run(self, run_id: str) -> SimpleNamespace:
            experiments = {"a" * 32: "41", "b" * 32: "42"}
            return SimpleNamespace(info=SimpleNamespace(experiment_id=experiments[run_id]))

    result = {
        "train_run_id": "a" * 32,
        "models": {"fused": {"evaluation_run_id": "b" * 32}},
    }

    runs = representative._runs(result, "https://mlflow.example.test/base/", Client())
    assert runs == {
        "fused.evaluation_run_id": "b" * 32,
        "fused.evaluation_run_uri": (
            "https://mlflow.example.test/base/#/experiments/42/runs/" + "b" * 32
        ),
        "train_run_id": "a" * 32,
        "train_run_uri": "https://mlflow.example.test/base/#/experiments/41/runs/" + "a" * 32,
    }
    report = representative.render(
        {
            "demo": {
                "commit": "deadbeef",
                "dirty": False,
                "hardware": {"cuda": None, "threads": 1},
                "metrics": {},
                "runs": runs,
                "seconds": 1.0,
            }
        }
    )
    assert runs["fused.evaluation_run_uri"] in report
    assert f"`{'b' * 32}`" not in report


def test_record_parity_logs_baseline_observed_and_signed_delta() -> None:
    class Client:
        def __init__(self) -> None:
            self.metrics: list[tuple[str, str, float]] = []
            self.terminated: tuple[str, str] | None = None

        def get_run(self, run_id: str) -> SimpleNamespace:
            assert run_id == "b" * 32
            return SimpleNamespace(info=SimpleNamespace(experiment_id="7"))

        def create_run(self, experiment_id: str, *, tags: dict[str, str]) -> SimpleNamespace:
            assert experiment_id == "7"
            assert tags == {
                "mlflow.runName": "parity-titanic",
                "dsio.parity.consumer": "titanic",
                "dsio.parity.baseline_commit": "before",
                "dsio.parity.baseline_run_id": "a" * 32,
                "dsio.parity.observed_commit": "after",
                "dsio.parity.observed_run_id": "b" * 32,
                "dsio.parity.status": "passed",
                "dsio.parity.tolerance": "0.0",
            }
            return SimpleNamespace(info=SimpleNamespace(run_id="c" * 32, experiment_id="7"))

        def log_metric(self, run_id: str, key: str, value: float) -> None:
            self.metrics.append((run_id, key, value))

        def set_terminated(self, run_id: str, *, status: str) -> None:
            self.terminated = (run_id, status)

    client = Client()
    parity = representative.record_parity(
        "titanic",
        {
            "commit": "before",
            "metrics": {"accuracy": 0.625},
            "runs": {"evaluation_run_id": "a" * 32},
        },
        {
            "commit": "after",
            "metrics": {"accuracy": 0.625},
            "runs": {"evaluation_run_id": "b" * 32},
        },
        tolerance=0.0,
        ui_uri="https://mlflow.example.test",
        client=client,
    )

    assert parity == {
        "moved": {},
        "run_id": "c" * 32,
        "run_uri": "https://mlflow.example.test/#/experiments/7/runs/" + "c" * 32,
    }
    assert client.metrics == [
        ("c" * 32, "baseline.accuracy", 0.625),
        ("c" * 32, "observed.accuracy", 0.625),
        ("c" * 32, "delta.accuracy", 0.0),
    ]
    assert client.terminated == ("c" * 32, "FINISHED")


def test_failed_parity_records_deltas_and_missing_metrics() -> None:
    class Client:
        def __init__(self) -> None:
            self.tags: dict[str, str] = {}
            self.metrics: dict[str, float] = {}
            self.status = ""

        def get_run(self, run_id: str) -> SimpleNamespace:
            return SimpleNamespace(info=SimpleNamespace(experiment_id="7"))

        def create_run(self, experiment_id: str, *, tags: dict[str, str]) -> SimpleNamespace:
            self.tags = tags
            return SimpleNamespace(info=SimpleNamespace(run_id="c" * 32, experiment_id="7"))

        def log_metric(self, run_id: str, key: str, value: float) -> None:
            self.metrics[key] = value

        def set_terminated(self, run_id: str, *, status: str) -> None:
            self.status = status

    client = Client()
    parity = representative.record_parity(
        "demo",
        {
            "commit": "before",
            "metrics": {"kept": 1.0, "dropped": 2.0},
            "runs": {"evaluation_run_id": "a" * 32},
        },
        {
            "commit": "after",
            "metrics": {"kept": 1.5, "added": 3.0},
            "runs": {"evaluation_run_id": "b" * 32},
        },
        tolerance=0.0,
        ui_uri="https://mlflow.example.test",
        client=client,
    )

    assert parity["moved"] == {
        "added": (None, 3.0),
        "dropped": (2.0, None),
        "kept": (1.0, 1.5),
    }
    assert client.metrics["delta.kept"] == 0.5
    assert client.tags["dsio.parity.missing_metrics"] == '["added", "dropped"]'
    assert client.tags["dsio.parity.status"] == "failed"
    assert client.status == "FAILED"


def test_consumer_runs_in_a_fresh_process_with_prefect_isolated_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setitem(
        representative.CONSUMERS,
        "isolated",
        representative.Consumer("module_that_must_not_import_in_parent:flow", data, {"x": 1}),
    )
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "https://mlflow.example.test")
    monkeypatch.setenv("PREFECT_API_URL", "https://shared-prefect.example.test")

    class Client:
        def search_experiments(self, *, max_results: int) -> list[object]:
            assert max_results == 1
            return []

        def get_run(self, run_id: str) -> SimpleNamespace:
            assert run_id == "a" * 32
            return SimpleNamespace(info=SimpleNamespace(experiment_id="7"))

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda _: Client())

    observed: dict[str, object] = {}

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            output = "deadbeef\n" if "rev-parse" in command else ""
            return subprocess.CompletedProcess(command, 0, stdout=output)
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        observed.update(
            command=command,
            kwargs=kwargs,
            prefect_home_existed=Path(environment["PREFECT_HOME"]).is_dir(),
        )
        result_path = Path(command[command.index("--result") + 1])
        result_path.write_text(
            json.dumps(
                {
                    "result": {"metrics": {"accuracy": 1.0}, "train_run_id": "a" * 32},
                    "hardware": {
                        "cpu": "test",
                        "threads": 1,
                        "torch": "x",
                        "cuda": None,
                        "python": "x",
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(representative.subprocess, "run", execute)
    record = representative.run("isolated", tmp_path / "runs")

    command = observed["command"]
    kwargs = observed["kwargs"]
    assert isinstance(command, list)
    assert command[:3] == [
        representative.sys.executable,
        str(Path(representative.__file__).resolve()),
        "_worker",
    ]
    assert "module_that_must_not_import_in_parent:flow" in command
    assert isinstance(kwargs, dict)
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert observed["prefect_home_existed"] is True
    assert "PREFECT_API_URL" not in environment
    assert environment["MLFLOW_TRACKING_URI"] == "https://mlflow.example.test"
    assert kwargs["check"] is True
    assert record["runs"] == {
        "train_run_id": "a" * 32,
        "train_run_uri": "https://mlflow.example.test/#/experiments/7/runs/" + "a" * 32,
    }
    assert "module_that_must_not_import_in_parent" not in representative.sys.modules


def test_worker_serializes_only_the_representative_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "result.json"
    flow_result = {
        "metrics": {"accuracy": 1.0},
        "train_run_id": "a" * 32,
        "submission_bytes": b"not JSON serializable",
        "models": {"fused": {"evaluation_run_id": "b" * 32}},
    }
    monkeypatch.setattr(
        representative.importlib,
        "import_module",
        lambda _: SimpleNamespace(flow=lambda *args, **kwargs: flow_result),
    )
    monkeypatch.setattr(representative, "_hardware", lambda: {"cpu": "test"})

    assert (
        representative._worker_main(
            [
                "--flow",
                "fake.module:flow",
                "--data",
                str(tmp_path / "data"),
                "--workspace",
                str(tmp_path / "workspace"),
                "--options",
                "{}",
                "--result",
                str(result_path),
            ]
        )
        == 0
    )
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "hardware": {"cpu": "test"},
        "result": {
            "fused.evaluation_run_id": "b" * 32,
            "metrics": {"accuracy": 1.0},
            "train_run_id": "a" * 32,
        },
    }


def test_compare_mode_never_rewrites_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.json"
    report_path = tmp_path / "baseline.md"
    baseline = {
        "demo": {
            "commit": "before",
            "hardware": {"cpu": "test", "cuda": None, "threads": 1},
            "metrics": {"accuracy": 1.0},
            "runs": {"evaluation_run_id": "a" * 32},
        }
    }
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    report_path.write_text("immutable baseline\n", encoding="utf-8")
    original_json = baseline_path.read_bytes()
    original_report = report_path.read_bytes()
    monkeypatch.setattr(representative, "BASELINE", baseline_path)
    monkeypatch.setattr(representative, "REPORT", report_path)
    monkeypatch.setattr(
        representative,
        "CONSUMERS",
        {"demo": representative.Consumer("fake:flow", tmp_path, {})},
    )
    monkeypatch.setattr(
        representative,
        "run",
        lambda name, root: {
            "commit": "after",
            "hardware": {"cpu": "test", "cuda": None, "threads": 1},
            "metrics": {"accuracy": 1.0},
            "runs": {"evaluation_run_id": "b" * 32},
            "seconds": 1.0,
            "tracking_uri": "http://localhost:5000",
            "ui_uri": "https://mlflow.example.test",
        },
    )
    recorded: list[dict[str, Any]] = []

    def record(*args: object, **kwargs: object) -> dict[str, Any]:
        recorded.append({"args": args, "kwargs": kwargs})
        return {
            "moved": {},
            "run_id": "c" * 32,
            "run_uri": "https://mlflow.example.test/#/experiments/7/runs/" + "c" * 32,
        }

    monkeypatch.setattr(representative, "record_parity", record)

    assert representative.main(["demo", "--compare"]) == 0
    assert len(recorded) == 1
    assert baseline_path.read_bytes() == original_json
    assert report_path.read_bytes() == original_report


def test_every_kaggle_consumer_has_a_representative_configuration() -> None:
    from tools import consumer_metrics

    kaggle = {
        row.consumer.removeprefix("reference_projects/kaggle/").replace("/", "_")
        for row in consumer_metrics.count_lines()
        if row.kaggle
    }
    assert kaggle == set(representative.CONSUMERS)


def test_committed_representative_evidence_uses_immutable_http_run_uris() -> None:
    baseline = json.loads(representative.BASELINE.read_text(encoding="utf-8"))

    for record in baseline.values():
        tracking_uri = record["tracking_uri"].rstrip("/")
        ui_uri = record.get("ui_uri", tracking_uri).rstrip("/")
        assert tracking_uri.startswith(("http://", "https://"))
        run_ids = {
            name: value for name, value in record["runs"].items() if name.endswith("_run_id")
        }
        run_uris = {
            name: value for name, value in record["runs"].items() if name.endswith("_run_uri")
        }
        assert {name.removesuffix("_run_id") for name in run_ids} == {
            name.removesuffix("_run_uri") for name in run_uris
        }
        for name, run_uri in run_uris.items():
            assert run_uri.startswith(f"{ui_uri}/#/experiments/")
            assert "/runs/" in run_uri
            assert run_uri.endswith(run_ids[f"{name.removesuffix('_run_uri')}_run_id"])
