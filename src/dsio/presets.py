"""Presets: the runnable configurations this project defines.

A preset is a function returning a validated ``RunConfig``. Variants are *arguments*,
not files — `dsio run <preset> seed=7` needs nothing checked in, which is why this repo
has no tree of config files to keep in sync (ADR 0001).

This module is imported by discovery, so anything decorated with ``@preset`` here is
runnable from the CLI. Add your own below; upstream does not touch this file, so it will
not conflict when you merge.
"""

from __future__ import annotations

from dsio.config.presets import preset, stage_hook
from dsio.config.schema import RunConfig

_STARTER = "spine_starter"


@preset
def spine_baseline(
    store: str = _STARTER,
    labels: str = _STARTER,
    split: str = _STARTER,
    lr: float = 3e-3,
    seed: int = 42,
) -> RunConfig:
    """Starter baseline: `dsio run spine_baseline` works verbatim in a fresh clone.

    At its defaults this trains on a tiny synthetic tone-vs-noise signal — not a real
    dataset — staged by a registered stage hook (`_stage_if_default`, below) rather than
    by this function: resolving a preset, including under `--dry-run`, must write
    nothing and mutate no registry, so staging happens only on the path that is
    actually about to execute. Overriding `store`, `labels` or `split` (not any
    argument — `lr` and `seed` are unrelated to the corpus) opts out of the synthetic
    corpus, since that means the caller is bringing their own staged data.
    """
    # Imported here, not at module scope, so that enumerating presets does not pay for
    # importing a task. Bare `dsio run` lists presets and their parameters by
    # introspecting signatures; it never constructs a config, so it must not pull in
    # torch the day a built-in preset uses TorchTask.
    from dsio.data.views import WindowSpec
    from dsio.train.torch_task import Component, TorchTask, TrainerConfig

    return RunConfig(
        name=f"{store}-lr{lr}",
        seed=seed,
        tags=("baseline", "torch"),
        task=TorchTask(
            store=store,
            window=WindowSpec(length=64, stride=32, label_policy="majority"),
            labels=labels,
            split=split,
            backbone=Component(name="conv1d", params={"hidden": 8, "out_dim": 8, "depth": 1}),
            head=Component(name="linear", params={"out_dim": 2}),
            loss=Component(name="cross_entropy", params={"threshold": 0.5}),
            transform=Component(name="instance_standardize"),
            lr=lr,
            metrics=("accuracy", "roc_auc"),
            trainer=TrainerConfig(max_epochs=60, accelerator="cpu", devices=1, checkpoint=False),
        ),
    )


@stage_hook("spine_baseline")
def _stage_if_default(config: RunConfig) -> None:
    """Stage the synthetic starter corpus — only called on the execute path, never
    from `resolve()` or `--dry-run` (see `dsio.cli.run_cmd`), and only when
    `store`/`labels`/`split` are still `spine_baseline`'s defaults, so a caller who
    overrode any of them to point at their own data is never touched by this."""
    from dsio.train.torch_task import TorchTask

    task = config.task
    if isinstance(task, TorchTask) and (task.store, task.labels, task.split) == (
        _STARTER,
        _STARTER,
        _STARTER,
    ):
        _stage_starter_corpus()


def _stage_starter_corpus() -> None:
    """Build the tiny synthetic tone-vs-noise corpus `spine_baseline` trains on, once.

    Idempotent — skips the store or the split file if either already exists — so a
    second `dsio run spine_baseline` neither rebuilds the store nor re-registers the
    label. Reuses exactly the corpus shape the test suite already builds for the same
    purpose (see `tests/train/test_torch_runner.py`), not a new synthetic-data path.
    """
    import numpy as np

    from dsio.data.store import SignalStore, data_root
    from dsio.model.registry import LABELS
    from dsio.model.registry import labels as register_labels
    from dsio.splits.models import SplitFile, SplitFold
    from dsio.train.torch_task import SPLITS_ROOT

    if _STARTER not in LABELS:

        @register_labels(_STARTER)
        def _spine_starter_labels(store: SignalStore) -> np.ndarray:
            out = np.zeros(store.n_rows, dtype=np.float32)
            for entity in store.entities:
                out[entity.start_row : entity.end_row] = float(entity.attrs["positive"])
            return out

    store_path = data_root() / _STARTER
    if not store_path.exists():
        rng = np.random.default_rng(0)
        with SignalStore.builder(store_path, channels=1) as builder:
            for group in range(6):
                positive = group % 2 == 0
                t = np.arange(400) / 100.0
                signal = (rng.standard_normal((400, 1)) * 0.5).astype("float32")
                if positive:
                    signal[:, 0] += (np.sin(2 * np.pi * 5 * t) * 2.0).astype("float32")
                builder.add(
                    f"p{group}", signal, group=f"p{group}", attrs={"positive": int(positive)}
                )

    split_path = SPLITS_ROOT / _STARTER / "fold0.yaml"
    if not split_path.exists():
        SplitFile(
            store=_STARTER,
            name=_STARTER,
            folds=[
                SplitFold(
                    index=0,
                    parts={"test": ["p0", "p1"], "val": ["p2"], "train": ["p3", "p4", "p5"]},
                )
            ],
        ).save(split_path)
