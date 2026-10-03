"""The representative-tier runner records and compares parity faithfully."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
            assert run_uri.startswith(f"{tracking_uri}/#/experiments/")
            assert "/runs/" in run_uri
            assert run_uri.endswith(run_ids[f"{name.removesuffix('_run_uri')}_run_id"])
