"""`execute()` seeds every RNG it can touch, so calling it directly is reproducible.

`dsio run` (``src/dsio/cli/run_cmd.py``) already called ``seed_everything`` before
starting a run, but nothing seeded a *direct* call to ``execute()`` — which is what every
runner test, and any caller that is not going through the CLI, actually does. A
``RandomForestClassifier`` built with no explicit ``random_state`` draws from numpy's
global RNG, so two runs of the same config used to disagree unless something upstream had
happened to seed it. That silent gap is what ``execute`` seeding at its own entry closes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dsio.config.schema import RunConfig
from dsio.runs import RunLedger
from dsio.train import execute, load_runners
from dsio.train.tabular import TabularTask


@pytest.fixture(autouse=True)
def _runners() -> None:
    load_runners()


def _run_once(config: RunConfig, runs_root: Path) -> dict[str, float]:
    ledger = RunLedger(runs_root)
    run = ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with run:
        return execute(config, run)


def test_execute_is_deterministic_for_the_same_config_and_seed(tmp_path: Path) -> None:
    """The headline promise — same config, same seed, identical metrics — for a caller
    that never goes through `dsio run`. `random_forest` is the estimator that actually
    exercises the gap: it has internal randomness with no seed of its own, so it only
    reproduces if something upstream seeded numpy's global RNG."""
    config = RunConfig(
        name="det",
        seed=7,
        task=TabularTask(
            dataset="iris", estimator="random_forest", folds=3, keep_model=False
        ),
    )
    first = _run_once(config, tmp_path / "runs_a")
    second = _run_once(config, tmp_path / "runs_b")
    assert first == second
