"""The agent runner end to end: same ledger, same folds, same artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsio.agents import ScriptedProvider
from dsio.agents.task import GRADERS, PROVIDERS, SUITES, TOOLSETS, AgentTask, Suite
from dsio.config.schema import RunConfig
from dsio.eval import read_report
from dsio.runs import RunLedger
from dsio.train import check, execute, load_runners

load_runners()

TASKS = {f"q{i}": f"question {i}" for i in range(12)}
ANSWERS = {f"q{i}": f"answer {i}" for i in range(12)}


@pytest.fixture(autouse=True)
def registered() -> None:
    if "demo" in SUITES:
        return

    @SUITES.register("demo")
    def _demo() -> Suite:
        return Suite(
            name="demo",
            prompts=TASKS,
            expected=ANSWERS,
            # Four templates of three variants each: the grouping that stops near-identical
            # generated tasks from being split across train and test.
            groups={key: f"template{i // 3}" for i, key in enumerate(sorted(TASKS))},
        )

    @PROVIDERS.register("perfect")
    def _perfect(**_: object) -> ScriptedProvider:
        return ScriptedProvider({TASKS[k]: ANSWERS[k] for k in TASKS})

    @PROVIDERS.register("flaky")
    def _flaky(**_: object) -> ScriptedProvider:
        return ScriptedProvider(
            {TASKS[k]: ANSWERS[k] for k in TASKS}, jitter=0.5
        )

    @TOOLSETS.register("none")
    def _none() -> dict:
        return {}


def run(task: AgentTask, tmp_path: Path, name: str = "agent"):  # type: ignore[no-untyped-def]
    config = RunConfig(name=name, seed=0, task=task)
    check(config)
    ledger = RunLedger(tmp_path / "runs")
    active = ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with active:
        metrics = execute(config, active)
    return active, metrics


# --- config refusals ------------------------------------------------------------------


def test_a_stochastic_configuration_measured_once_is_rejected() -> None:
    """A point estimate of a distribution, with no idea how far it moves."""
    with pytest.raises(ValueError, match="repeats"):
        AgentTask(suite="demo", provider="flaky", temperature=1.0, repeats=1)


def test_temperature_zero_needs_no_repeats() -> None:
    assert AgentTask(suite="demo", provider="perfect", temperature=0.0, repeats=1)


def test_preflight_resolves_every_name_before_spending_anything() -> None:
    """Cheaper here than anywhere else in dsio: what is avoided is a partially-completed
    run that already bought completions."""
    config = RunConfig(name="x", task=AgentTask(suite="demo", provider="nope"))
    with pytest.raises(KeyError, match="unknown provider"):
        check(config)


def test_an_unknown_grader_is_caught_up_front() -> None:
    config = RunConfig(name="x", task=AgentTask(suite="demo", provider="perfect", grader="??"))
    with pytest.raises(KeyError, match="unknown grader"):
        check(config)


# --- the run --------------------------------------------------------------------------


def test_the_runner_writes_the_same_artifact_contract(tmp_path: Path) -> None:
    """An agent run and a tabular run are read by exactly the same commands."""
    active, metrics = run(
        AgentTask(suite="demo", provider="perfect", cache_root=tmp_path / "cache"), tmp_path
    )
    report, oof = read_report(active.artifacts_dir)
    assert report.n_folds == 1
    assert oof is not None and len(oof) == 12
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["success_rate"] == pytest.approx(1.0)


def test_trajectories_are_the_artifact(tmp_path: Path) -> None:
    """A success rate cannot say whether failures were bad plans, bad tool calls or a tool
    that was down, and those have completely different fixes."""
    active, _ = run(
        AgentTask(suite="demo", provider="perfect", cache_root=tmp_path / "cache"), tmp_path
    )
    lines = (active.artifacts_dir / "trajectories.jsonl").read_text().strip().split("\n")
    assert len(lines) == 12
    first = json.loads(lines[0])
    assert first["steps"] and first["success"] is True
    assert "request" in first["steps"][0] and "response" in first["steps"][0]


def test_repeats_are_recorded_and_the_second_floor_reported(tmp_path: Path) -> None:
    active, metrics = run(
        AgentTask(
            suite="demo",
            provider="flaky",
            temperature=1.0,
            repeats=4,
            grader="contains",
            cache_root=tmp_path / "cache",
        ),
        tmp_path,
    )
    payload = json.loads((active.artifacts_dir / "repeats.json").read_text())
    assert payload["n_repeats"] == 4
    assert payload["n_examples"] == 12
    assert "run_std" in metrics and "agreement" in metrics
    lines = (active.artifacts_dir / "trajectories.jsonl").read_text().strip().split("\n")
    assert len(lines) == 48, "every task attempted every repeat"


def test_resource_metrics_are_reported_beside_the_score(tmp_path: Path) -> None:
    """A configuration two points better at four times the price is not obviously better,
    and a report that omits the price cannot say so."""
    _, metrics = run(
        AgentTask(suite="demo", provider="perfect", cache_root=tmp_path / "cache"), tmp_path
    )
    assert set(metrics) >= {
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "mean_steps",
        "tool_error_rate",
        "budget_exhausted",
        "cache_hit_rate",
    }


def test_a_rerun_is_served_from_the_response_cache(tmp_path: Path) -> None:
    """The expensive part of an agent experiment is the calls; changing a metric should not
    re-buy every completion."""
    task = AgentTask(suite="demo", provider="perfect", cache_root=tmp_path / "cache")
    run(task, tmp_path, name="first")
    _, metrics = run(task, tmp_path, name="second")
    assert metrics["cache_hit_rate"] == pytest.approx(1.0)
    assert metrics["cost_usd"] == 0.0


# --- grouping -------------------------------------------------------------------------


def test_tasks_split_by_their_template_group(tmp_path: Path) -> None:
    """Benchmark suites routinely contain families generated from one template, and
    splitting those across train and test is the same leak as splitting one subject."""
    from dsio.splits import SplitSpec, fold_paths, load_folds, write_splits

    suite = SUITES.get("demo")()
    root = tmp_path / "splits"
    write_splits(suite.examples, SplitSpec(scheme="kfold", k=2), name="k2", root=root)

    folds = load_folds(suite.examples, fold_paths(root, "k2"))
    groups = suite.examples.groups
    for fold in folds:
        assert not (set(groups[fold.train]) & set(groups[fold.test]))

    active, _ = run(
        AgentTask(
            suite="demo",
            provider="perfect",
            split="k2",
            splits_root=root,
            cache_root=tmp_path / "cache",
        ),
        tmp_path,
    )
    report, _ = read_report(active.artifacts_dir)
    assert report.n_folds == 2


def test_graders_are_registered_and_replaceable() -> None:
    assert {"exact_match", "contains"} <= set(GRADERS.names())
