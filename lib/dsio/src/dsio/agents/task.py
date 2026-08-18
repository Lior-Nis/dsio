"""The agent runner: the same ledger, the same folds, the same artifact contract.

An agent evaluation is a cross-validated experiment like any other, and it goes through the
same fold loop so that ``dsio eval verdict`` compares a prompt change the way it compares a
learning-rate change. What it adds is the second noise floor: the model's own sampling
variance, measured by repeating every task and reported next to the fold spread.

Tasks are a :class:`~dsio.data.adapters.KeyedExamples`, so grouping works the way it does
everywhere else. That matters more than it looks: benchmark suites routinely contain
families of near-identical tasks generated from one template, and splitting those across
train and test is the same leak as splitting one subject's recordings.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, model_validator

from dsio.agents.loop import Tool, contains, exact_match, run_agent
from dsio.agents.provider import CachedProvider, Provider, ResponseCache, cache_root
from dsio.agents.repeats import repeat_report
from dsio.agents.types import Trajectory
from dsio.config.registry import Registry
from dsio.config.schema import TASKS, TaskConfig
from dsio.data.adapters import KeyedExamples
from dsio.eval import Fold, FoldPrediction, cross_validate, write_report
from dsio.eval.metrics import METRICS
from dsio.train.runner import preflight, runner

if TYPE_CHECKING:
    from dsio.config.schema import RunConfig
    from dsio.runs.record import Run

TRAJECTORY_FILE = "trajectories.jsonl"
REPEATS_FILE = "repeats.json"

#: Task suites: name -> callable returning (examples, prompts, expected).
SUITES: Registry[Callable[..., Any]] = Registry("suite")

#: Providers: name -> callable returning a Provider.
PROVIDERS: Registry[Callable[..., Provider]] = Registry("provider")

#: Tool sets a suite can be run with.
TOOLSETS: Registry[Callable[..., Mapping[str, Tool]]] = Registry("toolset")

#: Graders: name -> callable turning a Trajectory into a pass/fail.
GRADERS: Registry[Callable[[Trajectory], bool]] = Registry("grader")

GRADERS.add("exact_match", exact_match)
GRADERS.add("contains", contains)


class Suite:
    """A set of tasks: keyed examples plus the prompt and expected answer for each."""

    def __init__(
        self,
        *,
        name: str,
        prompts: Mapping[str, str],
        expected: Mapping[str, str] | None = None,
        groups: Mapping[str, str] | None = None,
        attributes: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self.keys = sorted(prompts)
        self.prompts = dict(prompts)
        self.expected = dict(expected or {})
        per_key = attributes or {}
        names = sorted({key for values in per_key.values() for key in values})
        self.examples = KeyedExamples(
            name=name,
            keys=self.keys,
            groups=None if groups is None else [groups[key] for key in self.keys],
            attributes={
                attr: [per_key.get(key, {}).get(attr) for key in self.keys] for attr in names
            },
        )

    def __len__(self) -> int:
        return len(self.keys)


@TASKS.register("agent")
class AgentTask(TaskConfig):
    """Evaluate an agent configuration across folds and repeats."""

    kind: Literal["agent"] = "agent"

    suite: str = Field(description="Registered task suite.")
    provider: str = Field(description="Registered provider factory.")
    provider_params: dict[str, Any] = Field(default_factory=dict)
    toolset: str | None = None
    grader: str = "exact_match"

    model: str = "unspecified"
    system: str | None = None
    temperature: float = Field(default=0.0, ge=0.0)
    max_steps: int = Field(default=8, ge=1)

    repeats: int = Field(
        default=1,
        ge=1,
        description="Independent attempts per task. Below 2, sampling noise is unmeasured.",
    )
    split: str | None = Field(
        default=None, description="Committed split family. Absent means one fold over all tasks."
    )
    splits_root: Path = Path("splits")

    cache: bool = True
    cache_root: Path | None = None
    metrics: tuple[str, ...] = ("accuracy",)

    @model_validator(mode="after")
    def _check(self) -> AgentTask:
        if self.temperature > 0 and self.repeats < 2:
            raise ValueError(
                f"temperature is {self.temperature} but repeats is {self.repeats}: a "
                "stochastic configuration measured once reports a point estimate with no "
                "idea how far it moves. Set repeats >= 2, or temperature = 0."
            )
        return self


@preflight("agent")
def check_agent(config: RunConfig) -> None:
    """Resolve every name before a single request is issued.

    Cheaper here than anywhere else in dsio: the thing being avoided is not a slow data
    load but a partially-completed run that already spent money.
    """
    task = config.task
    assert isinstance(task, AgentTask)
    SUITES.get(task.suite)
    PROVIDERS.get(task.provider)
    GRADERS.get(task.grader)
    if task.toolset is not None:
        TOOLSETS.get(task.toolset)
    for name in task.metrics:
        METRICS.get(name)
    if task.split is not None:
        from dsio.splits import fold_paths

        fold_paths(task.splits_root, task.split)


@runner("agent")
def run_agent_task(config: RunConfig, run: Run) -> dict[str, float]:
    """Run every task, every repeat, in every fold — and report both noise floors."""
    task = config.task
    assert isinstance(task, AgentTask)

    suite: Suite = SUITES.get(task.suite)()
    provider: Provider = PROVIDERS.get(task.provider)(**task.provider_params)
    grader = GRADERS.get(task.grader)
    tools = TOOLSETS.get(task.toolset)() if task.toolset else {}

    cache: ResponseCache | None = None
    if task.cache:
        cache = ResponseCache(cache_root(task.cache_root))
        provider = CachedProvider(provider, cache)

    folds = _folds(task, suite)
    trajectories: list[Trajectory] = []
    outcomes: dict[str, list[float]] = {}

    def fit_predict(fold: Fold) -> FoldPrediction:
        """There is nothing to fit. Every held-out task is attempted `repeats` times.

        Keeping the shape of the fold loop rather than inventing a second one is what makes
        an agent result comparable to every other kind of result dsio produces — and a
        few-shot configuration that *does* select examples from the train part slots into
        exactly this seam.
        """
        truths: list[int] = []
        predictions: list[int] = []
        for position in fold.test.tolist():
            key = suite.keys[position]
            passes: list[float] = []
            for repeat in range(task.repeats):
                trajectory = run_agent(
                    provider,
                    example=key,
                    prompt=suite.prompts[key],
                    system=task.system,
                    tools=tools,
                    model=task.model,
                    temperature=task.temperature,
                    max_steps=task.max_steps,
                    repeat=repeat,
                    expected=suite.expected.get(key),
                )
                success = bool(grader(trajectory))
                trajectories.append(trajectory.model_copy(update={"success": success}))
                passes.append(float(success))
            outcomes[key] = passes
            truths.append(1)
            # The fold-level prediction is the majority verdict across repeats; the repeats
            # themselves drive the second floor and are reported separately.
            predictions.append(int(float(np.mean(passes)) >= 0.5))
        return FoldPrediction(
            y_true=np.array(truths), y_pred=np.array(predictions), y_score=None
        )

    report, oof = cross_validate(
        folds,
        fit_predict,
        metrics=list(task.metrics),
        n_rows=len(suite),
        on_fold=lambda fold: run.log_metrics(
            {f"fold/{name}": value for name, value in fold.metrics.items()}, step=fold.fold
        ),
    )
    write_report(run.artifacts_dir, report, oof)
    _write_trajectories(run.artifacts_dir / TRAJECTORY_FILE, trajectories)

    repeats = repeat_report(outcomes)
    (run.artifacts_dir / REPEATS_FILE).write_text(
        json.dumps(repeats.model_dump(mode="json"), indent=2, sort_keys=True)
    )

    metrics = dict(report.metrics)
    metrics["coverage"] = report.coverage
    metrics["success_rate"] = repeats.mean
    metrics["run_std"] = repeats.run_std
    metrics["agreement"] = repeats.agreement
    metrics.update(_resource_metrics(trajectories))
    if cache is not None:
        metrics["cache_hit_rate"] = float(cache.stats["hit_rate"])
    run.log_metrics(metrics)
    return metrics


def _folds(task: AgentTask, suite: Suite) -> list[Fold]:
    if task.split is None:
        # No split means one evaluation pass over everything. Nothing is fitted, so the
        # train part is genuinely empty and says so rather than being faked.
        return [
            Fold(
                index=0,
                train=np.array([], dtype=np.int64),
                test=np.arange(len(suite)),
                name="all",
                evaluation_only=True,
            )
        ]

    from dsio.splits import fold_paths, load_folds

    return load_folds(suite.examples, fold_paths(task.splits_root, task.split))


def _write_trajectories(path: Path, trajectories: list[Trajectory]) -> None:
    """One JSON object per line, so a long run streams and a failed one is still readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory.model_dump(mode="json"), sort_keys=True) + "\n")


def _resource_metrics(trajectories: list[Trajectory]) -> dict[str, float]:
    """Cost, latency and effort — the axes an agent result is actually traded off against.

    A configuration that is two points better and four times the price is not obviously
    better, and a report that omits the price cannot say so.
    """
    if not trajectories:
        return {}
    usages = [trajectory.usage for trajectory in trajectories]
    tool_calls = sum(len(trajectory.tool_calls) for trajectory in trajectories)
    failed = sum(trajectory.failed_tool_calls for trajectory in trajectories)
    return {
        "cost_usd": float(sum(usage.cost_usd for usage in usages)),
        "input_tokens": float(sum(usage.input_tokens for usage in usages)),
        "output_tokens": float(sum(usage.output_tokens for usage in usages)),
        "seconds": float(sum(usage.seconds for usage in usages)),
        "mean_steps": float(np.mean([t.n_steps for t in trajectories])),
        "tool_calls": float(tool_calls),
        "tool_error_rate": float(failed / tool_calls) if tool_calls else 0.0,
        "budget_exhausted": float(
            np.mean([1.0 if t.error else 0.0 for t in trajectories])
        ),
    }
