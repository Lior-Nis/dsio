"""A minimal tabular runner, sufficient to prove the spine end to end.

Deliberately small. Splitting here is a single stratified holdout; the real splitter —
grouped by default, purged and embargoed for temporal data, with its assignment hashed
into the run record — arrives with ``dsio.splits`` and replaces this. The seam is
``_split``, so that swap touches one function.

The scikit-learn ``Pipeline`` is not decoration: it is the leakage boundary. Fitting a
scaler outside it, on all rows, is the most common way a tabular result becomes wrong.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from dsio.config.registry import Registry
from dsio.config.schema import TASKS, TaskConfig
from dsio.train.runner import preflight, runner

if TYPE_CHECKING:
    from dsio.config.schema import RunConfig
    from dsio.runs.record import Run

EstimatorFactory = Any
ESTIMATORS: Registry[EstimatorFactory] = Registry("estimator")


@TASKS.register("tabular")
class TabularTask(TaskConfig):
    """Fit an estimator on a tabular dataset and score it on a holdout."""

    kind: Literal["tabular"] = "tabular"
    dataset: str = Field(description="Name of a registered dataset loader.")
    estimator: str = "logreg"
    test_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


DatasetLoader = Any
DATASETS: Registry[DatasetLoader] = Registry("dataset")


@DATASETS.register("breast_cancer")
def _breast_cancer() -> tuple[Any, Any]:
    from sklearn.datasets import load_breast_cancer

    bundle = load_breast_cancer()
    return bundle.data, bundle.target


@DATASETS.register("iris")
def _iris() -> tuple[Any, Any]:
    from sklearn.datasets import load_iris

    bundle = load_iris()
    return bundle.data, bundle.target


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


def _split(features: Any, target: Any, *, test_fraction: float, seed: int) -> tuple[Any, ...]:
    """Stratified holdout. Replaced by ``dsio.splits`` in phase 3."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        features,
        target,
        test_size=test_fraction,
        random_state=seed,
        stratify=target,
    )


@preflight("tabular")
def check_tabular(config: RunConfig) -> None:
    """Resolve every name this task depends on before any data is read."""
    task = config.task
    assert isinstance(task, TabularTask)
    DATASETS.get(task.dataset)
    ESTIMATORS.get(task.estimator)


@runner("tabular")
def run_tabular(config: RunConfig, run: Run) -> dict[str, float]:
    """Fit, score, and persist a tabular model, recording everything through the run."""
    from sklearn.metrics import accuracy_score, f1_score

    task = config.task
    assert isinstance(task, TabularTask)

    features, target = DATASETS.get(task.dataset)()
    x_train, x_test, y_train, y_test = _split(
        features, target, test_fraction=task.test_fraction, seed=config.seed
    )

    estimator = ESTIMATORS.get(task.estimator)(**task.params)
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "n_train": float(len(y_train)),
        "n_test": float(len(y_test)),
    }
    run.log_metrics(metrics)

    artifact = run.artifacts_dir / "model.pkl"
    artifact.write_bytes(pickle.dumps(estimator))
    return metrics
