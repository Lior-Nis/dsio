"""The tabular runner: cross-validated, grouped by default, on the shared fold loop.

What this runner owns is small and deliberate — load a dataset, build folds, hand a
``fit_predict`` to :func:`~dsio.eval.loop.cross_validate`, persist a model. It owns no
accumulation, no scoring, no artifact layout; those are framework code, identical here and
in the torch and forecast runners that follow.

Two things are not negotiable in this file:

**The sklearn ``Pipeline`` is the leakage boundary.** Fitting a scaler outside it, over all
rows, is the most common way a tabular result silently becomes wrong. Every estimator
factory here returns a ``Pipeline``, so preprocessing is refitted inside every fold.

**Grouping beats stratification when both apply.** If a dataset declares groups, folds are
grouped, and no amount of stratification convenience is allowed to override that. A
stratified fold that splits a subject across train and test is worse than an imbalanced one
that does not.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, model_validator

from dsio.config.registry import Registry
from dsio.config.schema import TASKS, TaskConfig
from dsio.eval import Fold, FoldPrediction, cross_validate, write_report
from dsio.eval.metrics import METRICS
from dsio.train.runner import preflight, runner

if TYPE_CHECKING:
    from dsio.config.schema import RunConfig
    from dsio.runs.record import Run

EstimatorFactory = Any
ESTIMATORS: Registry[EstimatorFactory] = Registry("estimator")

DatasetLoader = Any
DATASETS: Registry[DatasetLoader] = Registry("dataset")


class Dataset:
    """A tabular dataset plus the grouping key that makes its folds honest.

    ``groups`` is the leakage boundary — subject, machine, well, symbol. A loader that
    returns ``None`` is asserting that rows are genuinely independent, which for a real
    sensor or clinical corpus is almost never true.
    """

    def __init__(
        self,
        features: Any,
        target: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        name: str = "",
    ) -> None:
        self.features = features
        self.target = np.asarray(target)
        self.groups = None if groups is None else np.asarray(groups)
        self.name = name
        if self.groups is not None and self.groups.size != self.target.size:
            raise ValueError(
                f"dataset {name!r}: {self.groups.size} group labels for "
                f"{self.target.size} rows"
            )

    def __len__(self) -> int:
        return int(self.target.size)


@TASKS.register("tabular")
class TabularTask(TaskConfig):
    """Cross-validate an estimator on a tabular dataset."""

    kind: Literal["tabular"] = "tabular"
    dataset: str = Field(description="Name of a registered dataset loader.")
    estimator: str = "logreg"
    folds: int = Field(default=5, ge=1, description="1 means a single stratified holdout.")
    test_fraction: float = Field(
        default=0.25, gt=0.0, lt=1.0, description="Holdout size when folds=1."
    )
    metrics: tuple[str, ...] = ("accuracy", "f1_macro")
    params: dict[str, Any] = Field(default_factory=dict)
    keep_model: bool = Field(
        default=True, description="Refit on all rows and persist. Off for large sweeps."
    )

    @model_validator(mode="after")
    def _check(self) -> TabularTask:
        if not self.metrics:
            raise ValueError(
                "a run with no metrics produces predictions nobody can compare; "
                f"pick from {', '.join(METRICS.names())}"
            )
        return self


# --- datasets and estimators --------------------------------------------------------


@DATASETS.register("breast_cancer")
def _breast_cancer() -> Dataset:
    from sklearn.datasets import load_breast_cancer

    bundle = load_breast_cancer()
    return Dataset(bundle.data, bundle.target, name="breast_cancer")


@DATASETS.register("iris")
def _iris() -> Dataset:
    from sklearn.datasets import load_iris

    bundle = load_iris()
    return Dataset(bundle.data, bundle.target, name="iris")


@ESTIMATORS.register("logreg")
def _logreg(**params: Any) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, **params)),
        ]
    )


@ESTIMATORS.register("random_forest")
def _random_forest(**params: Any) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline([("model", RandomForestClassifier(**params))])


# --- folds --------------------------------------------------------------------------


def build_folds(dataset: Dataset, *, k: int, test_fraction: float, seed: int) -> list[Fold]:
    """Grouped where the dataset has groups, stratified where it does not.

    The choice is made from the data, not from config, because a config option to disable
    grouping is a config option to produce a wrong number. dsio's split files are the
    supported way to say something more specific; this is the convenience path for
    datasets that arrive as plain arrays.
    """
    from sklearn.model_selection import (
        GroupKFold,
        GroupShuffleSplit,
        StratifiedKFold,
        StratifiedShuffleSplit,
    )

    target, groups = dataset.target, dataset.groups
    placeholder = np.zeros(len(dataset))

    if k == 1:
        splitter: Any = (
            GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
            if groups is not None
            else StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        )
    elif groups is not None:
        # GroupKFold is deterministic and takes no seed; the shuffled variant needs one.
        splitter = GroupKFold(n_splits=k)
    else:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    return [
        Fold(index=i, train=train, test=test)
        for i, (train, test) in enumerate(
            splitter.split(placeholder, target, groups)
            if groups is not None
            else splitter.split(placeholder, target)
        )
    ]


# --- run ----------------------------------------------------------------------------


@preflight("tabular")
def check_tabular(config: RunConfig) -> None:
    """Resolve every name this task depends on before any data is read."""
    task = config.task
    assert isinstance(task, TabularTask)
    DATASETS.get(task.dataset)
    ESTIMATORS.get(task.estimator)
    for name in task.metrics:
        METRICS.get(name)


@runner("tabular")
def run_tabular(config: RunConfig, run: Run) -> dict[str, float]:
    """Cross-validate, write the artifact contract, and record every fold."""
    task = config.task
    assert isinstance(task, TabularTask)

    dataset = DATASETS.get(task.dataset)()
    if not isinstance(dataset, Dataset):  # a loader returning bare arrays
        dataset = Dataset(*dataset, name=task.dataset)

    folds = build_folds(
        dataset, k=task.folds, test_fraction=task.test_fraction, seed=config.seed
    )

    def fit_predict(fold: Fold) -> FoldPrediction:
        estimator = ESTIMATORS.get(task.estimator)(**task.params)
        estimator.fit(_rows(dataset.features, fold.train), dataset.target[fold.train])
        held_out = _rows(dataset.features, fold.test)
        return FoldPrediction(
            y_true=dataset.target[fold.test],
            y_pred=estimator.predict(held_out),
            y_score=_scores(estimator, held_out),
        )

    report, oof = cross_validate(
        folds,
        fit_predict,
        metrics=task.metrics,
        n_rows=len(dataset),
        on_fold=lambda fold: run.log_metrics(
            {f"fold/{name}": value for name, value in fold.metrics.items()}, step=fold.fold
        ),
    )
    write_report(run.artifacts_dir, report, oof)

    if task.keep_model:
        # Refit on everything: the cross-validated score estimates how well *this
        # procedure* generalises, and the model you ship should use all the evidence.
        final = ESTIMATORS.get(task.estimator)(**task.params)
        final.fit(dataset.features, dataset.target)
        (run.artifacts_dir / "model.pkl").write_bytes(pickle.dumps(final))

    metrics = dict(report.metrics)
    metrics["coverage"] = report.coverage
    metrics.update({f"{name}_fold_sd": value for name, value in report.per_fold_std.items()})
    run.log_metrics(metrics)
    return metrics


def _rows(features: Any, positions: np.ndarray) -> Any:
    """Positional row selection that works for numpy, polars and pandas alike."""
    if hasattr(features, "iloc"):
        return features.iloc[positions]
    if hasattr(features, "__getitem__") and hasattr(features, "shape"):
        return features[positions]
    return [features[int(i)] for i in positions]


def _scores(estimator: Any, features: Any) -> np.ndarray | None:
    """Continuous output where the estimator has one, else ``None``.

    Binary problems return the positive-class column so ranking metrics get a 1-D score;
    multiclass returns the full matrix, which those metrics then reject by name rather
    than silently scoring column 1 of a five-class problem.
    """
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(features))
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(features))
    return None
