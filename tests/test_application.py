"""Application-boundary orchestration stays ordered and policy-preserving."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import dsio.application as application
from dsio.application import CompletedRun, DryRunResult
from dsio.config.presets import StageHookFn
from dsio.config.registry import Registry
from dsio.config.schema import RunConfig
from dsio.splits.models import SplitError


def _stage_hooks(name: str, hook: StageHookFn) -> Registry[StageHookFn]:
    hooks: Registry[StageHookFn] = Registry("application-test-stage-hook")
    hooks.add(name, hook)
    return hooks


def test_dry_run_preflights_without_staging_or_starting(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("dry-run performed execution work")

    monkeypatch.setattr(application, "bootstrap", lambda: calls.append("bootstrap"))
    monkeypatch.setattr(
        application,
        "resolve",
        lambda name, overrides: calls.append(f"resolve:{name}:{overrides}") or config,
    )
    monkeypatch.setattr(application, "check", lambda resolved: calls.append("preflight"))
    monkeypatch.setattr(application, "seed_everything", unexpected)
    monkeypatch.setattr(application, "start_run", unexpected)
    monkeypatch.setattr(application, "execute", unexpected)
    monkeypatch.setattr(
        application,
        "STAGE_HOOKS",
        _stage_hooks("demo", unexpected),
    )

    result = application.execute_preset(
        "demo",
        ["seed=7"],
        dry_run=True,
        command=("dsio", "run", "demo"),
    )

    assert result == DryRunResult("demo", config)
    assert calls == ["bootstrap", "resolve:demo:['seed=7']", "preflight"]


def test_dry_run_reports_prerequisites_supplied_by_stage_hook(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stage(_config: RunConfig) -> None:
        raise AssertionError("dry-run invoked the stage hook")

    def missing_prerequisite(_config: RunConfig) -> None:
        raise SplitError("split is not staged")

    monkeypatch.setattr(application, "bootstrap", lambda: None)
    monkeypatch.setattr(application, "resolve", lambda *_args: config)
    monkeypatch.setattr(application, "STAGE_HOOKS", _stage_hooks("demo", stage))
    monkeypatch.setattr(application, "check", missing_prerequisite)

    result = application.execute_preset(
        "demo", [], dry_run=True, command=("dsio", "run", "demo")
    )

    assert isinstance(result, DryRunResult)
    assert result.pending_stage is not None
    assert "split is not staged" in result.pending_stage


def test_real_run_stages_preflights_seeds_starts_executes_and_cleans_up(
    config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    start_kwargs: dict[str, Any] = {}
    record = SimpleNamespace(run_id="run-1")
    active_run = SimpleNamespace(
        dir=tmp_path / "scratch",
        record=record,
        mlflow_run_id="mlflow-1",
    )

    def stage(resolved: RunConfig) -> None:
        assert resolved is config
        calls.append("stage")

    def start(**kwargs: Any) -> Any:
        calls.append("start")
        start_kwargs.update(kwargs)
        return active_run

    def run(resolved: RunConfig, active: Any) -> dict[str, float]:
        assert resolved is config and active is active_run
        calls.append("execute")
        return {"accuracy": 0.75}

    monkeypatch.setattr(application, "bootstrap", lambda: calls.append("bootstrap"))
    monkeypatch.setattr(application, "resolve", lambda *_args: calls.append("resolve") or config)
    monkeypatch.setattr(application, "STAGE_HOOKS", _stage_hooks("demo", stage))
    monkeypatch.setattr(application, "check", lambda resolved: calls.append("preflight"))
    monkeypatch.setattr(
        application,
        "seed_everything",
        lambda seed: calls.append("seed") or {"python": seed},
    )
    monkeypatch.setattr(application, "start_run", start)
    monkeypatch.setattr(application, "execute", run)
    monkeypatch.setattr(
        application.shutil,
        "rmtree",
        lambda path, **kwargs: calls.append(f"cleanup:{path.name}:{kwargs['ignore_errors']}"),
    )

    command = ("dsio", "run", "demo")
    result = application.execute_preset("demo", [], dry_run=False, command=command)

    assert result == CompletedRun(config, record, "mlflow-1", {"accuracy": 0.75})
    assert calls == [
        "bootstrap",
        "resolve",
        "stage",
        "preflight",
        "seed",
        "start",
        "execute",
        "cleanup:scratch:True",
    ]
    assert start_kwargs == {
        "name": config.name,
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "seed": config.seed,
        "seeds": {"python": config.seed},
        "tags": config.tags,
        "command": command,
        "fold": getattr(config.task, "fold", None),
    }


def test_failed_execution_still_cleans_scratch(
    config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    active_run = SimpleNamespace(
        dir=tmp_path / "failed-scratch",
        record=SimpleNamespace(),
        mlflow_run_id="mlflow-1",
    )

    def fail(_config: RunConfig, _run: Any) -> dict[str, float]:
        calls.append("execute")
        raise RuntimeError("training failed")

    monkeypatch.setattr(application, "bootstrap", lambda: None)
    monkeypatch.setattr(application, "resolve", lambda *_args: config)
    monkeypatch.setattr(application, "STAGE_HOOKS", Registry("empty-application-stage-hook"))
    monkeypatch.setattr(application, "check", lambda resolved: None)
    monkeypatch.setattr(application, "seed_everything", lambda seed: {})
    monkeypatch.setattr(application, "start_run", lambda **kwargs: active_run)
    monkeypatch.setattr(application, "execute", fail)
    monkeypatch.setattr(
        application.shutil,
        "rmtree",
        lambda path, **kwargs: calls.append(f"cleanup:{path.name}:{kwargs['ignore_errors']}"),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        application.execute_preset("demo", [], dry_run=False, command=("dsio", "run"))

    assert calls == ["execute", "cleanup:failed-scratch:True"]
