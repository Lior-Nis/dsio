"""The torch runner: Lightning's loop lives inside one fold, not around it.

This is the question the fold loop was built to answer. Lightning owns a training loop, and
so does cross-validation — but they are not the same loop and they do not compete. A
``Fold`` is one call to ``fit_predict``; a ``Trainer`` is created, fitted and discarded
inside that call. The outer loop never learns what a Trainer is, and the runner never
learns how out-of-fold predictions are accumulated or scored.

The consequence is that this file is short, and almost all of it is configuration. The
parts that are not configuration are three bugs FORGE paid for:

**Callbacks are never instantiated inside a bare ``except``.** FORGE's
``instantiate_callbacks`` catches everything and logs a warning, so a misconfigured
``ModelCheckpoint`` silently disables checkpointing for a multi-hour run and nobody learns
until they go looking for the weights.

**Metric names in filename templates are sanitised.** ``ap{metrics/val_ap:.3f}`` produced
six stray ``checkpoints/checkpoint-epoch=NN-metrics/`` *directories*, because the ``/``
inside the format field became a path separator.

**Checkpoints close over their lineage.** A FORGE classification checkpoint reloaded its
MAE encoder from a hardcoded path and failed on a fresh clone. An encoder here is named by
run id and digest, resolved through the ledger, and verified on load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, model_validator

from dsio.config.schema import TASKS, TaskConfig
from dsio.contracts import DsioModel
from dsio.data.store import SignalStore, data_root
from dsio.data.views import WindowSpec, load_or_build
from dsio.eval import Fold, FoldPrediction, cross_validate, write_report
from dsio.eval.metrics import METRICS
from dsio.nn import (
    AUGMENTORS,
    BACKBONES,
    HEADS,
    LABELS,
    LOSSES,
    PREPROCESSORS,
    TRANSFORMS,
    DsioModule,
    WindowDataset,
    make_loader,
)
from dsio.splits import fold_paths, load_folds
from dsio.train.runner import preflight, runner

if TYPE_CHECKING:
    from dsio.config.schema import RunConfig
    from dsio.runs.record import Run

SPLITS_ROOT = Path("splits")


class Component(DsioModel):
    """A registered component plus its keyword arguments.

    Name and parameters travel together so a config records exactly what was built. The
    alternative — a name here and a parameter block somewhere else — is how FORGE ended up
    with `adamw_differential_lr_1e{5,6,7}.yaml`.
    """

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class TrainerConfig(DsioModel):
    """Lightning trainer settings, restricted to what changes a result or its cost."""

    max_epochs: int = Field(default=10, ge=1)
    accelerator: str = "auto"
    devices: int | str = "auto"
    precision: str = "32-true"
    gradient_clip_val: float | None = Field(default=None, ge=0.0)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    early_stopping_patience: int | None = Field(default=None, ge=1)
    monitor: str = "val/loss"
    monitor_mode: Literal["min", "max"] = "min"
    checkpoint: bool = True
    log_every_n_steps: int = Field(default=10, ge=1)
    enable_progress_bar: bool = False
    deterministic: bool = True


@TASKS.register("torch")
class TorchTask(TaskConfig):
    """Cross-validate a torch model over a canonical store, fold by fold."""

    kind: Literal["torch"] = "torch"

    store: str = Field(description="Store name under the data root.")
    window: WindowSpec
    labels: str = Field(description="Registered per-row label provider.")
    split: str = Field(description="Committed split family under splits_root.")
    splits_root: Path = SPLITS_ROOT
    folds: tuple[int, ...] = Field(
        default=(),
        description="Which folds to run. Empty means all of them.",
    )

    backbone: Component
    head: Component = Component(name="linear")
    loss: Component = Component(name="cross_entropy")
    transform: Component | None = None
    preprocessor: Component | None = None
    augmentor: Component | None = None
    spectral_augmentor: Component | None = None

    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)
    trainer: TrainerConfig = TrainerConfig()

    metrics: tuple[str, ...] = ("accuracy", "f1_macro")
    keep_checkpoints: bool = True

    @model_validator(mode="after")
    def _check(self) -> TorchTask:
        if not self.metrics:
            raise ValueError(
                "a run with no metrics produces predictions nobody can compare; "
                f"pick from {', '.join(METRICS.names())}"
            )
        if self.window.label_policy == "none":
            raise ValueError(
                "a supervised torch run needs window labels, but the window spec's "
                "label_policy is 'none'; set 'any', 'majority' or 'ratio'"
            )
        return self


# --- pre-flight ---------------------------------------------------------------------


@preflight("torch")
def check_torch(config: RunConfig) -> None:
    """Resolve every name before a single byte of the corpus is read.

    FORGE's leaf schemas were ``extra="allow"`` with a bare ``_target_: str``, so a typo
    surfaced inside ``hydra.utils.instantiate`` after the data had already loaded. On a
    42M-window corpus that is the difference between a typo costing microseconds and
    costing twenty minutes.
    """
    task = config.task
    assert isinstance(task, TorchTask)

    LABELS.get(task.labels)
    BACKBONES.get(task.backbone.name)
    HEADS.get(task.head.name)
    LOSSES.get(task.loss.name)
    for optional, registry in (
        (task.transform, TRANSFORMS),
        (task.preprocessor, PREPROCESSORS),
        (task.augmentor, AUGMENTORS),
        (task.spectral_augmentor, AUGMENTORS),
    ):
        if optional is not None:
            registry.get(optional.name)
    for name in task.metrics:
        METRICS.get(name)

    # The split files are provenance, so their absence is a real error rather than an
    # invitation to generate something unrecorded on the fly.
    fold_paths(task.splits_root, task.split)


# --- assembly -----------------------------------------------------------------------


def build_module(task: TorchTask, *, channels: int, length: int) -> DsioModule:
    """Assemble the component chain for one fold.

    A fresh module per fold, never a reset one. ``load_state_dict`` back to an initial
    snapshot looks equivalent and is not: optimiser state, BatchNorm running statistics and
    any lazily-built buffer survive it, so fold 2 would start from fold 1's normalisation.
    """
    factory = BACKBONES.get(task.backbone.name)
    shape = _accepted(factory, {"channels": channels, "length": length})
    backbone = factory(**{**shape, **task.backbone.params})
    out_dim = getattr(backbone, "out_dim", None)
    head_params = dict(task.head.params)
    if out_dim is not None:
        head_params.setdefault("in_dim", out_dim)

    return DsioModule(
        backbone=backbone,
        head=HEADS.get(task.head.name)(**head_params),
        loss=LOSSES.get(task.loss.name)(**task.loss.params),
        transform=_optional(task.transform, TRANSFORMS),
        preprocessor=_optional(task.preprocessor, PREPROCESSORS),
        augmentor=_optional(task.augmentor, AUGMENTORS),
        spectral_augmentor=_optional(task.spectral_augmentor, AUGMENTORS),
        lr=task.lr,
        weight_decay=task.weight_decay,
    )


def _optional(component: Component | None, registry: Any) -> Any:
    return None if component is None else registry.get(component.name)(**component.params)


def _accepted(factory: Any, shape: dict[str, Any]) -> dict[str, Any]:
    """Keep only the shape arguments this factory actually takes.

    Framework-supplied shape and user-supplied params are filtered differently on purpose.
    A backbone that pools over time is genuinely length-agnostic, and making it declare a
    ``length`` it ignores would be a lie that later reads as a constraint. User params stay
    strict and unfiltered, so a typo in a config still fails loudly instead of vanishing
    into a catch-all.
    """
    import inspect

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        return shape
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        # A factory that takes **kwargs has told us nothing, so pass everything and let it
        # decide. Registering the class itself gives a real signature and avoids this.
        return shape
    return {key: value for key, value in shape.items() if key in signature.parameters}


def sanitise_metric(name: str) -> str:
    """Make a metric name safe inside a checkpoint filename template.

    ``val/loss`` in a template becomes a directory separator, which is how FORGE grew six
    stray ``checkpoints/checkpoint-epoch=NN-metrics/`` directories that looked like a
    corrupted run. Lightning offers no escaping, so the substitution happens here.
    """
    return name.replace("/", "_").replace("\\", "_").replace("=", "-")


def build_callbacks(task: TorchTask, directory: Path) -> list[Any]:
    """Construct the callbacks a fold needs, letting any failure propagate.

    FORGE wraps this in a bare ``except`` that logs a warning. A ModelCheckpoint that fails
    to construct then silently disables checkpointing for a multi-hour run, and the loss of
    the weights is discovered days later. A misconfigured callback is a configuration bug
    and must stop the run.
    """
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    callbacks: list[Any] = []
    if task.trainer.checkpoint:
        monitor = task.trainer.monitor
        callbacks.append(
            ModelCheckpoint(
                dirpath=directory,
                filename="epoch{epoch:02d}-" + sanitise_metric(monitor) + "{" + monitor + ":.4f}",
                monitor=monitor,
                mode=task.trainer.monitor_mode,
                save_top_k=1,
                auto_insert_metric_name=False,
            )
        )
    if task.trainer.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=task.trainer.monitor,
                mode=task.trainer.monitor_mode,
                patience=task.trainer.early_stopping_patience,
            )
        )
    return callbacks


# --- run ----------------------------------------------------------------------------


@runner("torch")
def run_torch(config: RunConfig, run: Run) -> dict[str, float]:
    """Cross-validate a torch model, writing the same artifact contract as every runner."""
    from lightning import Trainer

    task = config.task
    assert isinstance(task, TorchTask)

    store = SignalStore(data_root() / task.store)
    row_labels = np.asarray(LABELS.get(task.labels)(store))
    if row_labels.size != store.n_rows:
        raise ValueError(
            f"label provider {task.labels!r} returned {row_labels.size} values for "
            f"{store.n_rows} rows; labels must be per-row over the whole store"
        )

    index = load_or_build(store, task.window, labels=row_labels)
    folds = load_folds(index, fold_paths(task.splits_root, task.split), store=store)
    if task.folds:
        wanted = set(task.folds)
        folds = [fold for fold in folds if fold.index in wanted]
        if not folds:
            raise ValueError(
                f"folds {sorted(wanted)} match none of the split family {task.split!r}"
            )

    window_labels = index.labels
    if window_labels is None:
        raise ValueError("the window index carries no labels; check the label policy")

    def fit_predict(fold: Fold) -> FoldPrediction:
        module = build_module(task, channels=store.channels, length=task.window.length)
        directory = run.artifacts_dir / f"fold{fold.index}"
        directory.mkdir(parents=True, exist_ok=True)

        train_loader = make_loader(
            WindowDataset(store, index, fold.train),
            batch_size=task.batch_size,
            shuffle=True,
            num_workers=task.num_workers,
            seed=config.seed + fold.index,
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
            )
        )

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
            # "warn" rather than True: several ATen kernels have no deterministic variant,
            # and a hard failure would make whole model families unrunnable. The warning is
            # the signal that this run's numbers will not reproduce bit-for-bit.
            deterministic="warn" if task.trainer.deterministic else False,
            default_root_dir=directory,
            logger=False,
            callbacks=build_callbacks(task, directory) if validation is not None else [],
        )
        trainer.fit(module, train_loader, val_loader)

        predict_loader = make_loader(
            WindowDataset(store, index, fold.test),
            batch_size=task.batch_size,
            num_workers=task.num_workers,
            seed=config.seed,
        )
        batches = trainer.predict(module, predict_loader)
        return _assemble(batches, fold, window_labels)

    report, oof = cross_validate(
        folds,
        fit_predict,
        metrics=task.metrics,
        n_rows=len(index),
        on_fold=lambda fold: run.log_metrics(
            {f"fold/{name}": value for name, value in fold.metrics.items()}, step=fold.fold
        ),
    )
    write_report(run.artifacts_dir, report, oof)
    (run.artifacts_dir / "windows.json").write_text(
        json.dumps(
            {
                "store": store.path.name,
                "store_sha256": store.manifest().signal_sha256,
                "window_digest": task.window.digest,
                "windows": len(index),
                "split": task.split,
            },
            indent=2,
            sort_keys=True,
        )
    )

    metrics = dict(report.metrics)
    metrics["coverage"] = report.coverage
    metrics.update({f"{name}_fold_sd": value for name, value in report.per_fold_std.items()})
    run.log_metrics(metrics)
    return metrics


def _assemble(
    batches: Any, fold: Fold, window_labels: np.ndarray
) -> FoldPrediction:
    """Reassemble predicted batches into fold order, keyed by the row each carried.

    The predictions come back in loader order, which happens to match fold order today.
    Sorting by the reported row makes that a checked fact rather than an assumption that
    quietly stops holding the moment anyone adds a sampler.
    """
    import torch

    if not batches:
        raise ValueError(f"{fold.name}: the predict loop produced no batches")
    rows = torch.cat([torch.as_tensor(batch["row"]) for batch in batches]).numpy()
    logits = torch.cat([batch["prediction"] for batch in batches]).float()

    order = np.argsort(rows, kind="mergesort")
    rows, logits = rows[order], logits[order]
    if not np.array_equal(rows, np.sort(fold.test)):
        raise ValueError(
            f"{fold.name}: predictions cover {rows.size} rows that do not match the "
            f"{fold.test.size} held out; the loader and the fold disagree"
        )

    truth = window_labels[rows]
    if logits.ndim == 2 and logits.shape[1] >= 2:
        probabilities = torch.softmax(logits, dim=-1)
        prediction = probabilities.argmax(dim=-1).numpy()
        score = probabilities[:, 1].numpy() if logits.shape[1] == 2 else probabilities.numpy()
    else:
        score = torch.sigmoid(logits.squeeze(-1)).numpy()
        prediction = (score > 0.5).astype(np.int64)

    return FoldPrediction(
        y_true=(truth > 0.5).astype(np.int64) if truth.dtype.kind == "f" else truth,
        y_pred=prediction,
        y_score=score,
    )
