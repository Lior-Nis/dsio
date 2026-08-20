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

    At its defaults this trains on a tiny synthetic tone-vs-noise signal it stages on
    first run — not a real dataset. Replace `store`/`labels`/`split` with your own once
    you have real data staged; overriding any of them opts out of the synthetic corpus.
    """
    # Imported here, not at module scope, so that enumerating presets does not pay for
    # importing a task. Bare `dsio run` lists presets and their parameters by
    # introspecting signatures; it never constructs a config, so it must not pull in
    # torch the day a built-in preset uses TorchTask.
    from dsio.data.views import WindowSpec
    from dsio.train.torch_task import Component, TorchTask, TrainerConfig

    if (store, labels, split) == (_STARTER, _STARTER, _STARTER):
        _stage_starter_corpus()

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


def _stage_starter_corpus() -> None:
    """Build the tiny synthetic tone-vs-noise corpus `spine_baseline` trains on, once.

    Idempotent — skips the store or the split file if either already exists — so a
    second `dsio run spine_baseline` neither rebuilds the store nor re-registers the
    label. Reuses exactly the corpus shape the test suite already builds for the same
    purpose (see `tests/train/test_torch_runner.py`), not a new synthetic-data path.
    """
    import numpy as np

    from dsio.data.store import SignalStore, data_root
    from dsio.nn.registry import LABELS
    from dsio.nn.registry import labels as register_labels
    from dsio.splits.models import SplitFile
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
            fold=0,
            parts={"test": ["p0", "p1"], "val": ["p2"], "train": ["p3", "p4", "p5"]},
        ).save(split_path)
