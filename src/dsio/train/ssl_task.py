"""The pretraining runner: one fit, one encoder, one pinned reference.

Pretraining is deliberately **not** a fold loop. There is no held-out score to pool, because
the pretext loss is not the thing being estimated — the encoder is the output, and its
quality is measured downstream. Forcing it through ``cross_validate`` would produce a
cross-validated masked-reconstruction MSE, which is a number nobody should act on.

What it produces instead is an artifact with a pinned reference. The encoder goes into the
model registry, whose ``ModelRef`` has no way to express "latest", and a downstream run
names that ref. A downstream checkpoint that reloads its encoder from a hardcoded path
fails on a fresh clone and, worse, silently picks up whatever has since been written there;
neither is expressible when the reference is a name, a version and a digest.

**There is no pretext-objective registry any more.** Task 6b deleted ``dsio.ssl.methods``
and its ``METHODS``/``PretextObjective`` machinery along with ``SslModule`` and
``ContrastiveModule``: a pretraining run is built from the same pieces a supervised one is
(:class:`~dsio.nn.module.DsioModule`, a registered ``backbone``/``head``/``loss``), plus
exactly one of ``mask`` or ``augmentor`` telling this module which of the two
training-dataset contracts to build — masked-reconstruction (MAE's shape) or two-view
contrastive collation (SimCLR/VICReg's shape). Which one is set is what used to be implied
by a ``method`` name resolved through a registry; making it structural here is the
config-level twin of there no longer being a paradigm subclass to resolve it into.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, model_validator
from torch.utils.data import DataLoader

from dsio.artifacts.store import ModelRegistry
from dsio.config.schema import TASKS, TaskConfig
from dsio.data.adapters import SignalExamples
from dsio.data.store import SignalStore, data_root
from dsio.data.views import WindowIndex, WindowSpec, load_or_build
from dsio.eval.contract import Fold
from dsio.nn.data import TwoViewCollate, make_loader, train_dataset, val_dataset
from dsio.nn.masking import MASKS
from dsio.nn.module import DsioModule, export_encoder
from dsio.nn.registry import AUGMENTORS, BACKBONES, HEADS, LABELS, LOSSES, TRANSFORMS
from dsio.splits.folds import fold_paths, load_folds
from dsio.train.callbacks import OnlineProbe, RankMeMonitor
from dsio.train.runner import preflight, runner
from dsio.train.torch_task import Component, TrainerConfig, _accepted, _optional

if TYPE_CHECKING:
    from dsio.config.schema import RunConfig
    from dsio.runs.record import Run

ENCODER_FILE = "encoder.pt"


@TASKS.register("ssl_pretrain")
class SslPretrainTask(TaskConfig):
    """Pretrain an encoder on unlabelled windows and register it.

    There is no ``method`` name and no pretext-objective registry: ``backbone``, ``head``
    and ``loss`` are named directly, exactly as a supervised :class:`~dsio.train.torch_task.
    TorchTask` names them. What used to select between MAE, SimCLR and VICReg was a
    ``method`` string resolved into an object that built its own head and drove its own
    step; here it is simply which registered head/loss a caller names, plus exactly one of
    ``mask`` (draws a masked-reconstruction training dataset) or ``augmentor`` (builds two
    augmented views at collate time) — see :func:`build_loaders`.
    """

    kind: Literal["ssl_pretrain"] = "ssl_pretrain"

    store: str
    window: WindowSpec
    split: str = Field(description="Committed split family; pretraining uses its train part.")
    splits_root: Path = Path("splits")
    fold: int = Field(default=0, description="Which fold's train part to pretrain on.")

    backbone: Component
    head: Component
    loss: Component
    mask: Component | None = None
    augmentor: Component | None = None
    transform: Component | None = None
    normalize_target: bool = Field(
        default=True,
        description=(
            "Standardise the reconstruction target per window, per channel. Only "
            "meaningful alongside `mask`."
        ),
    )
    predict: Literal["prediction", "embedding"] = Field(
        default="embedding",
        description=(
            "What the module's predict_step returns: raw backbone features "
            "('embedding', the usual choice for a pretrained encoder someone else will "
            "load) or the objective head's own output ('prediction')."
        ),
    )

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
        if (self.mask is None) == (self.augmentor is None):
            raise ValueError(
                "an ssl_pretrain task needs exactly one of `mask` (a masked-reconstruction "
                "objective, e.g. MAE) or `augmentor` (a two-view contrastive objective, "
                f"e.g. SimCLR/VICReg) to know which training-dataset contract to build; "
                f"got mask={self.mask!r}, augmentor={self.augmentor!r}"
            )
        return self


@preflight("ssl_pretrain")
def check_ssl(config: RunConfig) -> None:
    """Resolve every name, including the ones only the probe needs."""
    task = config.task
    assert isinstance(task, SslPretrainTask)
    BACKBONES.get(task.backbone.name)
    HEADS.get(task.head.name)
    LOSSES.get(task.loss.name)
    if task.mask is not None:
        MASKS.get(task.mask.name)
    if task.augmentor is not None:
        AUGMENTORS.get(task.augmentor.name)
    if task.transform is not None:
        TRANSFORMS.get(task.transform.name)
    if task.labels is not None:
        LABELS.get(task.labels)
    fold_paths(task.splits_root, task.split)


def build_module(task: SslPretrainTask, *, channels: int, length: int) -> tuple[DsioModule, int]:
    """Assemble the pretraining module: the same chain
    :func:`~dsio.train.torch_task.build_module` assembles for a supervised run, with a head
    and loss the task names directly rather than a pretext objective building its own.

    Returns the feature dimension alongside the module — needed both to size the head above
    and to record in the encoder artifact below, so it is computed once here rather than
    twice.
    """
    factory = BACKBONES.get(task.backbone.name)
    shape = _accepted(factory, {"channels": channels, "length": length})
    backbone = factory(**{**shape, **task.backbone.params})
    feature_dim = int(getattr(backbone, "out_dim", 0)) or _infer_dim(backbone, channels, length)

    head_factory = HEADS.get(task.head.name)
    head_shape = _accepted(
        head_factory, {"in_dim": feature_dim, "channels": channels, "length": length}
    )
    module = DsioModule(
        backbone=backbone,
        head=head_factory(**{**head_shape, **task.head.params}),
        loss=LOSSES.get(task.loss.name)(**task.loss.params),
        transform=_optional(task.transform, TRANSFORMS),
        lr=task.lr,
        weight_decay=task.weight_decay,
        predict=task.predict,
    )
    return module, feature_dim


def build_loaders(
    task: SslPretrainTask, store: SignalStore, index: WindowIndex, fold: Fold, seed: int
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]] | None]:
    """The training and validation loaders, shaped by whichever of ``mask``/``augmentor``
    the task set — see :class:`SslPretrainTask`'s docstring for the two contracts.

    Both branches also build a validation loader when the fold has one, over the *same*
    contract as training: a masked reconstruction target or a two-view collated batch is
    exactly as meaningful on held-out windows as on training ones, and giving Lightning's
    validation loop a real batch to run is what lets :class:`~dsio.train.callbacks.
    OnlineProbe` and :class:`~dsio.train.callbacks.RankMeMonitor` fire at all — both hook
    ``on_validation_epoch_end``, which only runs when a validation loop actually happened.
    This loader is never the one a downstream classifier is scored against — that is
    :func:`val_dataset` used unmasked and uncollated, built separately below for the probe —
    so masking or collating it here does not touch the guarantee that protects a downstream
    evaluation split.

    The validation loader's randomness is seeded (``mask_seed``/``TwoViewCollate.seed``);
    the training loader's is not. A validation ``val/loss`` that redrew its mask or its
    views on every call would move for reasons that have nothing to do with the model —
    exactly the failure the deleted ``self.training`` guard existed to prevent, one layer
    down in the dataset. Training keeps fresh randomness every epoch deliberately; that is
    what makes it augmentation rather than a fixed transform.
    """
    validation = fold.val if fold.val is not None and fold.val.size else None

    if task.mask is not None:
        mask = MASKS.get(task.mask.name)(**task.mask.params)
        train_loader = make_loader(
            train_dataset(
                store, index, fold.train, mask=mask, normalize_target=task.normalize_target
            ),
            batch_size=task.batch_size,
            shuffle=True,
            num_workers=task.num_workers,
            seed=seed,
            # A short final batch would still reconstruct fine; drop_last here is only for
            # symmetry with the contrastive branch below, which genuinely needs it.
            drop_last=True,
        )
        val_loader = (
            None
            if validation is None
            else make_loader(
                train_dataset(
                    store,
                    index,
                    validation,
                    mask=mask,
                    normalize_target=task.normalize_target,
                    # Seeded here and only here: training wants a fresh mask every epoch,
                    # but val/loss must measure the same held-out reconstruction problem
                    # every time it is computed, not a freshly redrawn one — see
                    # WindowDataset.mask_seed.
                    mask_seed=seed,
                ),
                batch_size=task.batch_size,
                num_workers=task.num_workers,
                seed=seed,
                drop_last=True,
            )
        )
        return train_loader, val_loader

    assert task.augmentor is not None, "enforced by SslPretrainTask's model_validator"
    augment = AUGMENTORS.get(task.augmentor.name)(**task.augmentor.params)
    train_loader = make_loader(
        val_dataset(store, index, fold.train),
        batch_size=task.batch_size,
        shuffle=True,
        num_workers=task.num_workers,
        seed=seed,
        # SimCLR's negatives are the rest of the batch, so a short final batch changes the
        # objective rather than merely the throughput.
        drop_last=True,
        collate_fn=TwoViewCollate(augment),
    )
    val_loader = (
        None
        if validation is None
        else make_loader(
            val_dataset(store, index, validation),
            batch_size=task.batch_size,
            num_workers=task.num_workers,
            seed=seed,
            drop_last=True,
            # A separate, seeded collate instance: the same reasoning as mask_seed above.
            # Correctness relies on this loader never shuffling (make_loader's shuffle
            # defaults to False and nothing here overrides it), so the same rows land in
            # the same batch, in the same order, every time this loader is iterated.
            collate_fn=TwoViewCollate(augment, seed=seed),
        )
    )
    return train_loader, val_loader


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
    examples = SignalExamples(store, index)
    folds = load_folds(examples, fold_paths(task.splits_root, task.split))
    fold = next((f for f in folds if f.index == task.fold), None)
    if fold is None:
        raise ValueError(f"fold {task.fold} is not in split family {task.split!r}")

    module, feature_dim = build_module(task, channels=store.channels, length=task.window.length)
    train_loader, val_loader = build_loaders(task, store, index, fold, config.seed)

    callbacks: list[Any] = []
    probe: OnlineProbe | None = None
    if val_loader is not None and row_labels is not None:
        probe = OnlineProbe(
            # Embeddings for the probe must come from an unmasked, uncollated view
            # regardless of the pretext objective, so this always reaches for val_dataset,
            # never train_dataset and never TwoViewCollate.
            make_loader(
                val_dataset(store, index, fold.train),
                batch_size=task.batch_size,
                num_workers=task.num_workers,
                seed=config.seed,
            ),
            make_loader(
                val_dataset(store, index, fold.test),
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
            "state_dict": export_encoder(module),
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
    produces a shape error deep inside a head, at which point the message names a linear
    layer rather than the configuration that was wrong.
    """
    import torch

    with torch.no_grad():
        return int(backbone(torch.zeros(2, channels, length)).shape[-1])
