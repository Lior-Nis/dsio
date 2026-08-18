"""Adaptive search, driven straight through Optuna.

The `hydra-optuna-sweeper` plugin is unusable regardless of the Hydra decision: its stable
1.2.0 release is from 2022 and pins ``optuna<3.0``. Driving Optuna directly is both less
code and the only version-current option.

**A trial is an ordinary Run.** Same ledger, same config hash, same artifact contract, so a
searched result and a hand-run one are compared by the same machinery — and a trial that
turns out to matter can be promoted like any other run. Optuna keeps only what it needs to
choose the next point.

**Resume is the study, plus the ledger.** Optuna's own storage remembers which points were
tried, and the ledger remembers which configs completed. Pointing the study at a sqlite file
makes a killed search resumable in exactly the way a matrix is.

**A failed trial is pruned, not fatal.** A search whose fifth trial hits an out-of-memory
configuration should record that and carry on; a search that dies there has wasted the four
before it. Failures are surfaced in the report and the exit code, never swallowed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dsio.config.presets import resolve
from dsio.matrix.cells import MatrixError
from dsio.matrix.execute import completed_runs
from dsio.runs.record import RunLedger, RunStatus
from dsio.runs.seeding import seed_everything
from dsio.train.runner import check, execute

DISTRIBUTIONS = ("loguniform", "uniform", "int", "categorical")


@dataclass(frozen=True)
class SearchSpace:
    """One searchable parameter: a config path and the distribution to draw it from."""

    path: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()

    def suggest(self, trial: Any) -> Any:
        if self.kind == "loguniform":
            return trial.suggest_float(self.path, self.low, self.high, log=True)
        if self.kind == "uniform":
            return trial.suggest_float(self.path, self.low, self.high)
        if self.kind == "int":
            return trial.suggest_int(self.path, int(self.low or 0), int(self.high or 0))
        return trial.suggest_categorical(self.path, list(self.choices))


def parse_space(token: str) -> SearchSpace:
    """Parse ``path=loguniform(1e-5,1e-3)``, ``path=int(2,8)`` or ``path=categorical(a,b)``.

    The distribution is named explicitly rather than inferred from the values. A learning
    rate searched uniformly between 1e-5 and 1e-3 spends 90% of its trials above 1e-4, which
    is a silent and expensive mistake — and nothing in the numbers reveals which was meant.
    """
    path, _, rest = token.partition("=")
    if not _ or "(" not in rest or not rest.endswith(")"):
        raise MatrixError(
            f"search space {token!r} must be path=kind(args); "
            f"kinds are {', '.join(DISTRIBUTIONS)}"
        )
    kind, _, arguments = rest.partition("(")
    kind = kind.strip()
    if kind not in DISTRIBUTIONS:
        raise MatrixError(f"unknown distribution {kind!r}; use one of {', '.join(DISTRIBUTIONS)}")

    parts = [part.strip() for part in arguments[:-1].split(",") if part.strip()]
    if kind == "categorical":
        if not parts:
            raise MatrixError(f"categorical {path!r} has no choices")
        return SearchSpace(path=path, kind=kind, choices=tuple(parts))
    if len(parts) != 2:
        raise MatrixError(f"{kind} {path!r} needs exactly low and high, got {len(parts)}")
    low, high = float(parts[0]), float(parts[1])
    if low >= high:
        raise MatrixError(f"{kind} {path!r} has low {low} >= high {high}")
    return SearchSpace(path=path, kind=kind, low=low, high=high)


@dataclass
class TrialResult:
    number: int
    params: dict[str, Any]
    value: float | None
    config_hash: str
    run_id: str | None = None
    status: str = "completed"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "params": self.params,
            "value": self.value,
            "config_hash": self.config_hash[:12],
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class SearchReport:
    preset: str
    metric: str
    direction: str
    trials: list[TrialResult] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_value: float | None = None
    best_run_id: str | None = None

    @property
    def failed(self) -> list[TrialResult]:
        return [trial for trial in self.trials if trial.status == "failed"]

    @property
    def reused(self) -> int:
        return sum(1 for trial in self.trials if trial.status == "reused")

    @property
    def ok(self) -> bool:
        return self.best_value is not None and not self.failed

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "metric": self.metric,
            "direction": self.direction,
            "n_trials": len(self.trials),
            "reused": self.reused,
            "failed": len(self.failed),
            "best_value": self.best_value,
            "best_params": self.best_params,
            "best_run_id": self.best_run_id,
            "ok": self.ok,
            "trials": [trial.as_dict() for trial in self.trials],
        }


def run_search(
    preset: str,
    spaces: Sequence[SearchSpace],
    *,
    metric: str,
    direction: str = "maximize",
    n_trials: int = 20,
    seed: int = 42,
    base: Sequence[str] = (),
    ledger: RunLedger | None = None,
    storage: Path | str | None = None,
    study_name: str | None = None,
    reuse_completed: bool = True,
    on_trial: Callable[[TrialResult], None] | None = None,
    command: tuple[str, ...] = (),
) -> SearchReport:
    """Search ``spaces`` for the config that optimises ``metric``.

    ``storage`` points Optuna at a sqlite file, which is what makes a killed search
    resumable: the study remembers its trials, and the ledger remembers the runs.

    ``reuse_completed`` makes the search share the matrix's resume mechanism. A trial whose
    config hash already has a completed run reads that run's metric instead of retraining
    an identical configuration — so a search after a sweep costs only the points the sweep
    did not cover, and a re-invoked search does not repeat finished work. It is the same
    identity doing the same job, which is the point of content-addressing it.
    """
    try:
        import optuna
    except ModuleNotFoundError as error:  # pragma: no cover - depends on the install
        raise MatrixError(
            "search needs optuna, an optional extra; install dsio[search]"
        ) from error

    if not spaces:
        raise MatrixError("a search needs at least one parameter to vary")
    if direction not in ("maximize", "minimize"):
        raise MatrixError(f"direction must be maximize or minimize, got {direction!r}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ledger = ledger or RunLedger()
    report = SearchReport(preset=preset, metric=metric, direction=direction)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=None if storage is None else f"sqlite:///{storage}",
        study_name=study_name or f"dsio-{preset}",
        load_if_exists=storage is not None,
    )

    known = completed_runs(ledger) if reuse_completed else {}

    def objective(trial: Any) -> float:
        tokens = list(base) + [f"{space.path}={space.suggest(trial)}" for space in spaces]
        config = resolve(preset, tokens)
        result = TrialResult(
            number=trial.number,
            params=dict(trial.params),
            value=None,
            config_hash=config.config_hash,
        )

        previous = known.get(config.config_hash)
        if previous is not None and metric in previous.metrics:
            result.value = float(previous.metrics[metric])
            result.run_id = previous.run_id
            result.status = "reused"
            report.trials.append(result)
            if on_trial is not None:
                on_trial(result)
            return result.value

        try:
            check(config)
            seeds = seed_everything(config.seed)
            with ledger.start(
                name=config.name,
                config=config.to_dict(),
                config_hash=config.config_hash,
                seed=config.seed,
                seeds=seeds,
                tags=(*config.tags, "search"),
                command=command,
            ) as active:
                metrics = execute(config, active)
                active.finish(RunStatus.COMPLETED, metrics=metrics)
            if metric not in metrics:
                raise MatrixError(
                    f"trial {trial.number} produced no metric {metric!r}; it recorded "
                    f"{', '.join(sorted(metrics)) or 'nothing'}"
                )
            result.value = float(metrics[metric])
            result.run_id = active.run_id
            if reuse_completed:
                # Also guards within a single search: a sampler that revisits a point it
                # just evaluated would otherwise retrain it.
                known[config.config_hash] = active.record
        except Exception as error:
            result.status = "failed"
            result.error = f"{type(error).__name__}: {error}"
            report.trials.append(result)
            if on_trial is not None:
                on_trial(result)
            # Optuna records this as a pruned trial and continues sampling. The failure is
            # kept in the report and drives the exit code, so it is surfaced rather than
            # lost — but one bad configuration does not discard the trials before it.
            raise optuna.TrialPruned(result.error) from error

        report.trials.append(result)
        if on_trial is not None:
            on_trial(result)
        return result.value

    study.optimize(objective, n_trials=n_trials, catch=())

    successful = [
        trial
        for trial in report.trials
        if trial.status in ("completed", "reused") and trial.value is not None
    ]
    if successful:
        sign = 1.0 if direction == "maximize" else -1.0
        best = max(successful, key=lambda t: sign * float(t.value or 0.0))
        report.best_value = best.value
        report.best_params = best.params
        report.best_run_id = best.run_id
    return report
