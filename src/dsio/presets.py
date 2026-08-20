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
    store: str = "spine_starter",
    labels: str = "spine_starter",
    split: str = "spine_starter",
    lr: float = 1e-3,
    seed: int = 42,
) -> RunConfig:
    """Starter baseline. Replace the task with your own once you have data staged.

    `store`, `labels` and `split` name a corpus, a label provider and a committed split
    this project has not staged yet — `LABELS` intentionally ships with no built-in
    entries, so resolving this with its defaults preflights loudly until you register
    your own label provider and commit a split file, same as any other project preset.
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
            metrics=("accuracy",),
            trainer=TrainerConfig(max_epochs=2, accelerator="cpu", devices=1, checkpoint=False),
        ),
    )
