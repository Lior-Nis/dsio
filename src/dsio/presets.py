"""Presets: the runnable configurations this project defines.

A preset is a function returning a validated ``RunConfig``. Variants are *arguments*,
not files — `dsio run <preset> seed=7` needs nothing checked in, which is why this repo
has no tree of config files to keep in sync (ADR 0001).

This module is imported by discovery, so anything decorated with ``@preset`` here is
runnable from the CLI. Add your own below; upstream does not touch this file, so it will
not conflict when you merge.
"""

from __future__ import annotations

from dsio.config.presets import preset
from dsio.config.schema import RunConfig


@preset
def spine_baseline(
    dataset: str = "iris",
    estimator: str = "logreg",
    test_fraction: float = 0.25,
    seed: int = 42,
) -> RunConfig:
    """Starter baseline. Replace the task with your own once you have data staged."""
    # Imported here, not at module scope, so that enumerating presets does not pay for
    # importing a task. Bare `dsio run` lists presets and their parameters by
    # introspecting signatures; it never constructs a config, so it must not pull in
    # torch the day a built-in preset uses TorchTask.
    from dsio.train.tabular import TabularTask

    return RunConfig(
        name=f"{dataset}-{estimator}",
        seed=seed,
        tags=("baseline", "tabular"),
        task=TabularTask(
            dataset=dataset, estimator=estimator, test_fraction=test_fraction
        ),
    )
