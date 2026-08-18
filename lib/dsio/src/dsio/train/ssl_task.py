"""The pretraining runner: one fit, one encoder, one pinned reference.

Pretraining is deliberately **not** a fold loop. There is no held-out score to pool, because
the pretext loss is not the thing being estimated — the encoder is the output, and its
quality is measured downstream. Forcing it through ``cross_validate`` would produce a
cross-validated masked-reconstruction MSE, which is a number nobody should act on.

What it produces instead is an artifact with a pinned reference. The encoder goes into the
model registry, whose ``ModelRef`` has no way to express "latest", and a downstream run
names that ref. FORGE's classification checkpoints silently reloaded their MAE encoder from
a hardcoded path and failed on a fresh clone; there is no path here to hardcode.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, model_validator

from dsio.artifacts.store import ModelRegistry
from dsio.config.schema import TASKS, TaskConfig
from dsio.data.store import SignalStore, data_root
from dsio.data.views import WindowSpec, load_or_build
from dsio.nn import AUGMENTORS, BACKBONES, LABELS, TRANSFORMS, WindowDataset, make_loader
from dsio.splits import fold_paths, load_folds
from dsio.ssl.masking import MASKS
from dsio.ssl.methods import METHODS
from dsio.ssl.module import SslModule
from dsio.ssl.probe import OnlineProbe, RankMeMonitor
from dsio.train.runner import preflight, runner
from dsio.train.torch_task import Component, TrainerConfig, _accepted, _optional

if TYPE_CHECKING:
    from dsio.config.schema import RunConfig
    from dsio.runs.record import Run

ENCODER_FILE = "encoder.pt"


@TASKS.register("ssl_pretrain")
class SslPretrainTask(TaskConfig):
    """Pretrain an encoder on unlabelled windows and register it."""

    kind: Literal["ssl_pretrain"] = "ssl_pretrain"

    store: str
    window: WindowSpec
    split: str = Field(description="Committed split family; pretraining uses its train part.")
    splits_root: Path = Path("splits")
    fold: int = Field(default=0, description="Which fold's train part to pretrain on.")

    method: Component
    mask: Component | None = None
    augmentor: Component | None = None
    backbone: Component
    transform: Component | None = None

    register_as: str = Field(description="Name to register the encoder under.")

    labels: str | None = Field(
        default=None,
        description="Optional label provider, used only by the online probe.",
    )
    probe_every_n_epochs: int = Field(default=1, ge=1)
    probe_metrics: tuple[str, ...] = ("accuracy", "roc_auc")

    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=64, ge=2)
    num_workers: int = Field(default=0, ge=0)
    trainer: TrainerConfig = TrainerConfig(monitor="val/loss")

    @model_validator(mode="after")
    def _check(self) -> SslPretrainTask:
        if self.method.name == "mae" and self.mask is None:
            raise ValueError("the mae method needs a mask; set mask=Component(name=...)")
        if self.method.name in {"simclr", "vicreg"} and self.augmentor is None:
            raise ValueError(
                f"the {self.method.name} method needs an augmentor to build two views; "
                "without one both views are identical and the objective is degenerate"
            )
        return self


@preflight("ssl_pretrain")
def check_ssl(config: RunConfig) -> None:
    """Resolve every name, including the ones only the probe needs."""
    task = config.task
    assert isinstance(task, SslPretrainTask)
    METHODS.get(task.method.name)
    BACKBONES.get(task.backbone.name)
    if task.mask is not None:
        MASKS.get(task.mask.name)
    if task.augmentor is not None:
        AUGMENTORS.get(task.augmentor.name)
    if task.transform is not None:
        TRANSFORMS.get(task.transform.name)
    if task.labels is not None:
        LABELS.get(task.labels)
    fold_paths(task.splits_root, task.split)


def build_method(task: SslPretrainTask) -> Any:
    """Assemble the pretext objective, wiring its mask or augmentor into it."""
    factory = METHODS.get(task.method.name)
    params = dict(task.method.params)
    if task.mask is not None:
        params["mask"] = MASKS.get(task.mask.name)(**task.mask.params)
    if task.augmentor is not None:
        params["augment"] = AUGMENTORS.get(task.augmentor.name)(**task.augmentor.params)
    return factory(**params)


@runner("ssl_pretrain")
def run_ssl_pretrain(config: RunConfig, run: Run) -> dict[str, float]:
    """Pretrain, probe as it goes, and register the encoder with its lineage."""
    import torch
    from lightning import Trainer

    task = config.task
    assert isinstance(task, SslPretrainTask)

    store = SignalStore(data_root() / task.store)
    row_labels = None if task.labels is None else np.asarray(LABELS.get(task.labels)(store))
    index = load_or_build(store, task.window, labels=row_labels)
    folds = load_folds(index, fold_paths(task.splits_root, task.split), store=store)
    fold = next((f for f in folds if f.index == task.fold), None)
    if fold is None:
        raise ValueError(f"fold {task.fold} is not in split family {task.split!r}")

    method = build_method(task)
    factory = BACKBONES.get(task.backbone.name)
    shape = _accepted(factory, {"channels": store.channels, "length": task.window.length})
    backbone = factory(**{**shape, **task.backbone.params})
    feature_dim = int(getattr(backbone, "out_dim", 0)) or _infer_dim(
        backbone, store.channels, task.window.length
    )

    module = SslModule(
        method=method,
        backbone=backbone,
        head=method.build_head(feature_dim, store.channels, task.window.length),
        loss=torch.nn.Identity(),  # the objective owns the loss; this slot is unused
        transform=_optional(task.transform, TRANSFORMS),
        lr=task.lr,
        weight_decay=task.weight_decay,
    )

    train_loader = make_loader(
        WindowDataset(store, index, fold.train),
        batch_size=task.batch_size,
        shuffle=True,
        num_workers=task.num_workers,
        seed=config.seed,
        # SimCLR's negatives are the rest of the batch, so a short final batch changes the
        # objective rather than merely the throughput.
        drop_last=True,
    )
    validation = fold.val if fold.val is not None and fold.val.size else None
    val_loader = (
        None
        if validation is None
        else make_loader(
            WindowDataset(store, index, validation),
            batch_size=task.batch_size,
            num_workers=task.num_workers,
            seed=config.seed,
            drop_last=True,
        )
    )

    callbacks: list[Any] = []
    probe: OnlineProbe | None = None
    if val_loader is not None and row_labels is not None:
        probe = OnlineProbe(
            make_loader(
                WindowDataset(store, index, fold.train),
                batch_size=task.batch_size,
                num_workers=task.num_workers,
                seed=config.seed,
            ),
            make_loader(
                WindowDataset(store, index, fold.test),
                batch_size=task.batch_size,
                num_workers=task.num_workers,
                seed=config.seed,
            ),
            every_n_epochs=task.probe_every_n_epochs,
            metrics=task.probe_metrics,
        )
        callbacks.append(probe)
    elif val_loader is not None:
        callbacks.append(RankMeMonitor(val_loader, every_n_epochs=task.probe_every_n_epochs))

    trainer = Trainer(
        max_epochs=task.trainer.max_epochs,
        accelerator=task.trainer.accelerator,
        devices=task.trainer.devices,
        precision=task.trainer.precision,  # type: ignore[arg-type]
        gradient_clip_val=task.trainer.gradient_clip_val,
        accumulate_grad_batches=task.trainer.accumulate_grad_batches,
        log_every_n_steps=task.trainer.log_every_n_steps,
        enable_progress_bar=task.trainer.enable_progress_bar,
        enable_model_summary=False,
        deterministic="warn" if task.trainer.deterministic else False,
        default_root_dir=run.artifacts_dir,
        logger=False,
        callbacks=callbacks,
    )
    trainer.fit(module, train_loader, val_loader)

    buffer = io.BytesIO()
    torch.save(
        {
            "state_dict": module.encoder_state(),
            "backbone": task.backbone.model_dump(mode="json"),
            "transform": None if task.transform is None else task.transform.model_dump(mode="json"),
            "feature_dim": feature_dim,
            "channels": store.channels,
            "length": task.window.length,
        },
        buffer,
    )
    payload = buffer.getvalue()

    version = ModelRegistry().save(
        task.register_as,
        payload,
        run_id=run.run_id,
        config_hash=config.config_hash,
        code_hash=run.record.git.code_hash,
        data_snapshot_ids=(store.manifest().signal_sha256,),
        seed=config.seed,
    )
    (run.artifacts_dir / "encoder.json").write_text(
        json.dumps(version.ref.model_dump(mode="json"), indent=2, sort_keys=True)
    )

    metrics: dict[str, float] = {
        "train_windows": float(fold.train.size),
        "feature_dim": float(feature_dim),
        "encoder_version": float(version.version),
    }
    logged = trainer.logged_metrics
    for name in ("train/loss_epoch", "train/loss", "val/loss"):
        if name in logged:
            metrics[name.replace("/", "_")] = float(logged[name])
    if probe is not None and probe.history:
        for name, value in probe.history[-1].items():
            metrics[f"probe_{name}"] = float(value)
    run.log_metrics(metrics)
    return metrics


def _infer_dim(backbone: Any, channels: int, length: int) -> int:
    """Discover a backbone's output width by running one tensor through it.

    Preferred over asking the caller. A declared ``out_dim`` that disagrees with reality
    produces a shape error deep inside an objective's head, at which point the message
    names a linear layer rather than the configuration that was wrong.
    """
    import torch

    with torch.no_grad():
        return int(backbone(torch.zeros(2, channels, length)).shape[-1])
