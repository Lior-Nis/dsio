"""Shared Lightning Trainer and callback construction for concrete runners."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from dsio.contracts import DsioModel

if TYPE_CHECKING:
    from lightning import Trainer


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


def sanitise_metric(name: str) -> str:
    """Make a metric name safe inside a checkpoint filename template.

    A ``/`` inside a format field becomes a directory separator, so ``val/loss`` in a
    template silently creates nested directories that look like a corrupted run. Lightning
    offers no escaping, so the substitution happens here.
    """
    return name.replace("/", "_").replace("\\", "_").replace("=", "-")


def build_callbacks(
    trainer: TrainerConfig, directory: Path, *, has_validation: bool
) -> list[Any]:
    """Construct the checkpoint/early-stopping callbacks a fold needs, letting any
    construction failure propagate.

    Wrapping this in a bare ``except`` that logs a warning means a ModelCheckpoint which
    fails to construct silently disables checkpointing for a multi-hour run, and the loss of
    the weights is discovered days later. A misconfigured callback is a configuration bug and
    must stop the run.

    Takes a bare :class:`TrainerConfig`, not a task, so both runners that read one --
    ``run_torch``'s ``TorchTask`` and ``run_ssl_pretrain``'s ``SslPretrainTask`` -- can
    share this one construction (I1: pretraining used to read none of
    ``checkpoint``/``early_stopping_patience``/``monitor``/``monitor_mode`` at all, so
    ``SslPretrainTask``'s own deliberate ``TrainerConfig(monitor="val/loss")`` default did
    nothing).

    ``has_validation`` gates anything that monitors ``trainer.monitor``: with no
    validation loader, that metric is never logged (``DsioModule._common_step`` only logs
    under its own ``stage``), so ``EarlyStopping`` would raise on the metric it can never
    find, and a metric-ranked ``ModelCheckpoint`` would silently save nothing every
    epoch. ``checkpoint=True`` still gets a real ``ModelCheckpoint`` in that case -- it
    just keeps the most recent epoch instead of ranking by a metric that does not exist --
    rather than being dropped to an empty list the way a fold with no validation set used
    to be: an empty callback list combined with Lightning's own ``enable_checkpointing``
    default (``True`` unless a caller says otherwise) is exactly how Lightning ended up
    installing its *own* default ``ModelCheckpoint`` regardless of what
    ``checkpoint=False`` asked for.
    """
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    callbacks: list[Any] = []
    if trainer.checkpoint:
        if has_validation:
            monitor = trainer.monitor
            callbacks.append(
                ModelCheckpoint(
                    dirpath=directory,
                    filename=(
                        "epoch{epoch:02d}-" + sanitise_metric(monitor) + "{" + monitor + ":.4f}"
                    ),
                    monitor=monitor,
                    mode=trainer.monitor_mode,
                    save_top_k=1,
                    auto_insert_metric_name=False,
                )
            )
        else:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=directory,
                    filename="epoch{epoch:02d}",
                    save_top_k=1,
                    auto_insert_metric_name=False,
                )
            )
    if has_validation and trainer.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=trainer.monitor,
                mode=trainer.monitor_mode,
                patience=trainer.early_stopping_patience,
            )
        )
    return callbacks


def build_trainer(
    config: TrainerConfig,
    directory: Path,
    logger: Any,
    callbacks: list[Any],
) -> Trainer:
    """Construct Lightning's trainer from the shared runner configuration."""
    from lightning import Trainer

    return Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,  # type: ignore[arg-type]
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        enable_progress_bar=config.enable_progress_bar,
        enable_model_summary=False,
        # "warn" rather than True: several ATen kernels have no deterministic variant,
        # and a hard failure would make whole model families unrunnable. The warning is
        # the signal that this run's numbers will not reproduce bit-for-bit.
        deterministic="warn" if config.deterministic else False,
        default_root_dir=directory,
        logger=logger,
        # Without this, Lightning installs its own default ModelCheckpoint regardless of
        # `config.checkpoint`, defeating the explicit checkpoint policy and potentially
        # uploading an undigested model through the runner's bulk artifact upload.
        enable_checkpointing=config.checkpoint,
        callbacks=callbacks,
    )
