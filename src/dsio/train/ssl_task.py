"""The pretraining runner: one fit, one encoder, one pinned reference.

Pretraining deliberately produces no held-out score to pool, because the pretext loss is not
the thing being estimated — the encoder is the output, and its quality is measured
downstream. Scoring it across folds would produce a cross-validated masked-reconstruction
MSE, which is a number nobody should act on, so this runner writes no ``predictions.npz``
for :func:`dsio.eval.pool.pool_folds` to find.

What it produces instead is an MLflow artifact with a pinned reference. A downstream run
names the source run, artifact path and content digest. A checkpoint that reloads its
encoder from a hardcoded path fails on a fresh clone and, worse, silently picks up whatever
has since been written there; a verified artifact reference can do neither.

**There is no pretext-objective registry.** A pretraining run is built from the same pieces
a supervised one is
(:class:`~dsio.model.module.DsioModule`, an importable ``backbone``/``head``/``loss``), plus
exactly one of ``mask`` or ``augmentor`` telling this module which accelerator batch
augmentation to inject — masked reconstruction (MAE's shape) or two-view contrastive
augmentation (SimCLR/VICReg's shape). Which one is set is what used to be implied
by a ``method`` name resolved through a registry; making it structural here is the
config-level twin of there no longer being a paradigm subclass to resolve it into.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from pydantic import Field, model_validator

from dsio.batches import BatchLoader, WindowBatch
from dsio.config.components import ComponentConfig, ConfiguredComponent, resolve_component
from dsio.config.schema import TASKS, TaskConfig
from dsio.data.adapters import SignalExamples
from dsio.data.splits.folds import load_folds, require_fold, split_path
from dsio.data.store import SignalStore, data_root
from dsio.data.views import WindowIndex, WindowSpec, load_or_build
from dsio.dataset.dataset import make_loader, val_dataset
from dsio.eval.contract import Fold
from dsio.model.chain import ComponentChain, LossObjective
from dsio.model.module import DsioModule, export_encoder
from dsio.train.artifacts import save_artifact
from dsio.train.assembly import (
    accepted_shape_arguments,
    build_optional_component,
    component_factory,
)
from dsio.train.augmentation import MaskedReconstruction, TwoView
from dsio.train.callbacks import OnlineProbe, RankMeMonitor
from dsio.train.capabilities import (
    check_requested_capabilities,
    check_training_capabilities,
    log_capabilities,
    representative_batch,
)
from dsio.train.runner import preflight, runner
from dsio.train.tracking import (
    finite_metrics,
    log_run_artifacts,
    require_mlflow,
    tracked_run,
)
from dsio.train.trainer import TrainerConfig, build_callbacks, build_trainer

if TYPE_CHECKING:
    from dsio.config.schema import RunConfig
    from dsio.runs.record import Run


@TASKS.register("ssl_pretrain")
class SslPretrainTask(TaskConfig):
    """Pretrain an encoder on unlabelled windows and register it.

    There is no ``method`` name and no pretext-objective registry: ``backbone``, ``head``
    and ``loss`` are named directly, exactly as a supervised :class:`~dsio.train.torch_task.
    TorchTask` names them. What used to select between MAE, SimCLR and VICReg was a
    ``method`` string resolved into an object that built its own head and drove its own
    step; here it is simply which importable head/loss a caller names, plus exactly one of
    ``mask`` (builds a masked-reconstruction batch) or ``augmentor`` (builds two views)
    inside :meth:`DsioModule.training_step`; loaders return raw windows in both cases.
    """

    kind: Literal["ssl_pretrain"] = "ssl_pretrain"

    store: str
    window: WindowSpec
    split: str = Field(description="Committed split family; pretraining uses its train part.")
    splits_root: Path = Path("splits")
    fold: int = Field(default=0, description="Which fold's train part to pretrain on.")

    backbone: ConfiguredComponent
    head: ConfiguredComponent
    loss: ConfiguredComponent
    mask: ConfiguredComponent | None = None
    augmentor: ConfiguredComponent | None = None
    transform: ConfiguredComponent | None = None
    normalize_target: bool = Field(
        default=True,
        description=(
            "Standardise the reconstruction target per window, per channel. Only "
            "meaningful alongside `mask`."
        ),
    )

    register_as: str = Field(description="Name the encoder artifact is saved under.")

    labels: ConfiguredComponent | None = Field(
        default=None,
        description="Optional label provider, used only by the online probe.",
    )
    probe_every_n_epochs: int = Field(default=1, ge=1)
    probe_metrics: tuple[str, ...] = ("accuracy", "roc_auc")

    optimizer: ConfiguredComponent = Field(
        default_factory=lambda: ComponentConfig(
            reference="torch.optim:AdamW",
            parameters={"lr": 1e-3, "weight_decay": 0.0},
        )
    )
    scheduler: ConfiguredComponent | None = None
    batch_size: int = Field(default=64, ge=2)
    num_workers: int = Field(default=0, ge=0)
    trainer: TrainerConfig = TrainerConfig()

    @model_validator(mode="after")
    def _check(self) -> SslPretrainTask:
        if (self.mask is None) == (self.augmentor is None):
            raise ValueError(
                "an ssl_pretrain task needs exactly one of `mask` (a masked-reconstruction "
                "objective, e.g. MAE) or `augmentor` (a two-view contrastive objective, "
                f"e.g. SimCLR/VICReg) to know which training augmentation to build; "
                f"got mask={self.mask!r}, augmentor={self.augmentor!r}"
            )
        if self.trainer.early_stopping_patience is not None:
            raise ValueError(
                "ssl_pretrain early stopping requires a validation objective; stochastic "
                "training augmentation is intentionally unavailable in validation"
            )
        return self


@preflight("ssl_pretrain")
def check_ssl(config: RunConfig) -> None:
    """Resolve every name, including the ones only the probe needs.

    ``require_mlflow`` runs first, ahead of every component import below: decision 7 makes
    MLflow a run's hard dependency, and a run that cannot write to it should not spend even
    a typo-check's worth of time before saying so.
    """
    require_mlflow()
    task = config.task
    assert isinstance(task, SslPretrainTask)
    check_requested_capabilities(task.trainer)
    component_factory(task.backbone)
    component_factory(task.head)
    component_factory(task.loss)
    component_factory(task.optimizer)
    if task.scheduler is not None:
        _ssl_scheduler(task.scheduler)
    if task.mask is not None:
        component_factory(task.mask)
    if task.augmentor is not None:
        component_factory(task.augmentor)
    if task.transform is not None:
        component_factory(task.transform)
    if task.labels is not None:
        component_factory(task.labels)
    require_fold(task.splits_root, task.split, task.fold)


def build_module(
    task: SslPretrainTask, *, channels: int, length: int, seed: int = 0
) -> tuple[DsioModule, int]:
    """Assemble the pretraining module: the same chain
    :func:`~dsio.train.torch_task.build_module` assembles for a supervised run, with a head
    and loss the task names directly rather than a pretext objective building its own.

    Returns the feature dimension alongside the module — needed both to size the head above
    and to record in the encoder artifact below, so it is computed once here rather than
    twice.
    """
    factory, _ = component_factory(task.backbone)
    shape = accepted_shape_arguments(factory, {"channels": channels, "length": length})
    backbone = resolve_component(task.backbone, expected=torch.nn.Module, **shape)
    feature_dim = int(getattr(backbone, "out_dim", 0)) or _infer_dim(backbone, channels, length)

    head_factory, _ = component_factory(task.head)
    head_shape = accepted_shape_arguments(
        head_factory, {"in_dim": feature_dim, "channels": channels, "length": length}
    )
    optimizer_factory, optimizer_parameters = component_factory(task.optimizer)
    scheduler_factory = None
    scheduler_parameters: dict[str, Any] = {}
    if task.scheduler is not None:
        scheduler_factory, scheduler_parameters = _ssl_scheduler(task.scheduler)

    training_augmentation: torch.nn.Module
    if task.mask is not None:
        training_augmentation = MaskedReconstruction(
            resolve_component(task.mask), normalize_target=task.normalize_target
        )
        augmentation_identity: dict[str, Any] = {
            "kind": "masked_reconstruction",
            "component": dict(task.mask),
            "normalize_target": task.normalize_target,
        }
    else:
        assert task.augmentor is not None, "enforced by SslPretrainTask's model_validator"
        views = ("view-0", "view-1")
        training_augmentation = TwoView(
            resolve_component(task.augmentor, expected=torch.nn.Module), views=views
        )
        augmentation_identity = {
            "kind": "two_view",
            "component": dict(task.augmentor),
            "views": list(views),
        }

    module = DsioModule(
        model=ComponentChain(
            backbone=backbone,
            head=resolve_component(task.head, expected=torch.nn.Module, **head_shape),
            transform=build_optional_component(task.transform),
        ),
        objective=LossObjective(resolve_component(task.loss, expected=torch.nn.Module)),
        optimizer_factory=optimizer_factory,
        optimizer_parameters=optimizer_parameters,
        scheduler_factory=scheduler_factory,
        scheduler_parameters=scheduler_parameters,
        training_augmentation=training_augmentation,
        augmentation_seed=seed,
        augmentation_identity=augmentation_identity,
    )
    return module, feature_dim


def build_loaders(
    task: SslPretrainTask, store: SignalStore, index: WindowIndex, fold: Fold, seed: int
) -> tuple[BatchLoader[WindowBatch], None]:
    """Build one raw-window loader; stochastic pretext work starts after device transfer."""
    train_loader = make_loader(
        val_dataset(store, index, fold.train),
        batch_size=task.batch_size,
        shuffle=True,
        num_workers=task.num_workers,
        seed=seed,
        drop_last=True,
    )
    return train_loader, None


@runner("ssl_pretrain")
def run_ssl_pretrain(config: RunConfig, run: Run) -> dict[str, float]:
    """Pretrain, probe as it goes, and register the encoder with its lineage."""
    import torch

    # `tracked_run` keeps MLflow reachability first and stamps provenance before any
    # store, model or Trainer work. It also marks an existing run failed if anything in
    # this task-specific body crashes, including after Lightning has finalized success.
    with tracked_run(config, run) as mlflow_logger:
        task = config.task
        assert isinstance(task, SslPretrainTask)

        check_requested_capabilities(task.trainer)
        require_fold(task.splits_root, task.split, task.fold)

        store = SignalStore(data_root() / task.store)
        row_labels = (
            None
            if task.labels is None
            else resolve_component(task.labels, store, expected=np.ndarray)
        )
        index = load_or_build(store, task.window, labels=row_labels)
        examples = SignalExamples(store, index)
        folds = load_folds(examples, split_path(task.splits_root, task.split))
        # `require_fold` above already guarantees `task.fold` is declared, so this lookup
        # cannot fail on a live split file; kept as an assertion rather than silently
        # trusting it, so a TOCTOU (the file changing between the two reads) still fails
        # loudly.
        fold = next((f for f in folds if f.index == task.fold), None)
        assert fold is not None, (
            f"require_fold guaranteed fold {task.fold} exists in {task.split!r}"
        )

        module, feature_dim = build_module(
            task, channels=store.channels, length=task.window.length, seed=config.seed
        )
        train_loader, _ = build_loaders(task, store, index, fold, config.seed)

        callbacks: list[Any] = []
        probe: OnlineProbe | None = None
        if row_labels is not None:
            probe = OnlineProbe(
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
        else:
            callbacks.append(
                RankMeMonitor(
                    make_loader(
                        val_dataset(store, index, fold.train),
                        batch_size=task.batch_size,
                        num_workers=task.num_workers,
                        seed=config.seed,
                    ),
                    every_n_epochs=task.probe_every_n_epochs,
                )
            )

        # I1/"the rule": `task.trainer` is a full `TrainerConfig` here too, including a
        # deliberate `monitor="val/loss"` default (this class's own field default,
        # above) -- but until now nothing in this function ever read `checkpoint`,
        # `early_stopping_patience`, `monitor` or `monitor_mode`, so `early_stopping_
        # patience=5` on a 500-epoch pretrain silently ran the full 500. Reusing
        # `build_callbacks` -- the same construction `run_torch` uses -- wires all four
        # for pretraining too, rather than leaving them honoured nowhere.
        callbacks += build_callbacks(task.trainer, run.artifacts_dir, has_validation=False)

        trainer = build_trainer(
            task.trainer,
            run.artifacts_dir,
            mlflow_logger,
            callbacks,
        )
        capabilities = check_training_capabilities(
            trainer,
            module,
            representative_batch(train_loader),
            requested=task.trainer,
        )
        log_capabilities(mlflow_logger, capabilities)
        trainer.fit(module, train_loader)

        buffer = io.BytesIO()
        torch.save(
            {
                "state_dict": export_encoder(module),
                "backbone": dict(task.backbone),
                "transform": None if task.transform is None else dict(task.transform),
                "feature_dim": feature_dim,
                "channels": store.channels,
                "length": task.window.length,
            },
            buffer,
        )
        payload = buffer.getvalue()

        # `run_id` cites MLflow's own run id, not dsio's human-readable label
        # (`run.run_id`): decision 7 makes MLflow's the real, collision-free identity
        # (see `dsio.runs.record`'s module docstring), and this manifest row is exactly
        # the kind of place a stale or colliding label would be misleading.
        # The encoder is an artifact of *this* run, not a registered model: nothing serves
        # it and nothing aliases it, so it lives under the run that produced it and its
        # lineage is that run's own params and tags rather than a second copy on a registry
        # entry. Promotion, if it ever happens, is `mlflow.register_model` over this URI.
        assert run.mlflow_run_id is not None, "stamp_provenance runs first and sets this"
        ref = save_artifact(payload, run_id=run.mlflow_run_id, name=task.register_as)
        (run.artifacts_dir / "encoder.json").write_text(
            json.dumps(ref.model_dump(mode="json"), indent=2, sort_keys=True)
        )

        metrics: dict[str, float] = {
            "train_windows": float(fold.train.size),
            "feature_dim": float(feature_dim),
        }
        logged = trainer.logged_metrics
        for name in ("train/loss_epoch", "train/loss"):
            if name in logged:
                metrics[name.replace("/", "_")] = float(logged[name])
        if probe is not None and probe.history:
            for name, value in probe.history[-1].items():
                metrics[f"probe_{name}"] = float(value)
        # `train_windows` and `feature_dim` are computed here, not
        # inside a `*_step` hook, so nothing during training's own `self.log(...)` calls
        # captured them -- logged explicitly, through the same logger the Trainer
        # streamed epoch metrics to, so they land in the same MLflow run as everything
        # else this pretraining fold produced.
        mlflow_logger.log_metrics(finite_metrics(metrics))
        log_run_artifacts(run, mlflow_logger)
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


def _ssl_scheduler(config: ConfiguredComponent) -> tuple[Any, dict[str, Any]]:
    factory, parameters = component_factory(config)
    if isinstance(factory, type) and issubclass(
        factory, torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        raise ValueError(
            "ssl_pretrain ReduceLROnPlateau requires a validation objective; stochastic "
            "training augmentation is intentionally unavailable in validation"
        )
    return factory, parameters
