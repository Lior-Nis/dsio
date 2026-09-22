"""The exact reusable Lightning module for every training task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, cast

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import Metric

from dsio.batches import PredictionBatch
from dsio.config.components import (
    ComponentError as ConfiguredComponentError,
)
from dsio.config.components import require_importable_component, validate_component_config
from dsio.model.chain import ComponentError, export_encoder

Stage = Literal["train", "validate", "test"]
type Batch = Mapping[str, Any]
type ObjectiveValue = Tensor | Metric
type ObjectiveResult = Mapping[str, ObjectiveValue]

_INTEGER_ROW_DTYPES = {
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class Objective(Protocol):
    """One task step: execute a model on a batch and return loss plus metrics."""

    def __call__(
        self,
        model: nn.Module,
        batch: Batch,
        stage: Stage,
    ) -> ObjectiveResult: ...


class ModuleError(ValueError):
    """The configured model, objective, or batch violated the training contract."""


class DsioModule(LightningModule):
    """Compose one PyTorch model and objective behind native Lightning lifecycle hooks."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del kwargs
        raise TypeError(
            "DsioModule cannot be subclassed; vary training through injected components"
        )

    def __init__(
        self,
        *,
        model: nn.Module,
        objective: Objective,
        optimizer_factory: Any = torch.optim.AdamW,
        optimizer_parameters: Mapping[str, Any] | None = None,
        scheduler_factory: Any | None = None,
        scheduler_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise ModuleError(f"model must be a torch nn.Module, got {type(model).__name__}")
        if not callable(objective):
            raise ModuleError("objective must be callable")
        if not callable(optimizer_factory):
            raise ModuleError("optimizer_factory must be callable")
        if scheduler_factory is not None and not callable(scheduler_factory):
            raise ModuleError("scheduler_factory must be callable when configured")
        if scheduler_factory is None and scheduler_parameters:
            raise ModuleError("scheduler_parameters require a scheduler_factory")
        try:
            require_importable_component(model, "model")
            require_importable_component(objective, "objective")
            optimizer_reference = require_importable_component(
                optimizer_factory, "optimizer_factory"
            )
            optimizer_config = validate_component_config(
                {
                    "reference": optimizer_reference,
                    "parameters": dict(optimizer_parameters or {}),
                }
            )
            scheduler_config = None
            if scheduler_factory is not None:
                scheduler_reference = require_importable_component(
                    scheduler_factory, "scheduler_factory"
                )
                scheduler_config = validate_component_config(
                    {
                        "reference": scheduler_reference,
                        "parameters": dict(scheduler_parameters or {}),
                    }
                )
        except ConfiguredComponentError as error:
            raise ModuleError(str(error)) from None
        self.model = model
        self.objective = objective
        self.optimizer_factory = optimizer_factory
        self.optimizer_parameters = optimizer_config["parameters"]
        self.scheduler_factory = scheduler_factory
        self.scheduler_parameters = (
            {} if scheduler_config is None else scheduler_config["parameters"]
        )
        self._metric_log_names: dict[int, str] = {}
        self.save_hyperparameters(
            {
                "optimizer": optimizer_config,
                "scheduler": scheduler_config,
            }
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate inference to the configured native PyTorch model."""
        return self.model(*args, **kwargs)

    def encode(self, x: Tensor) -> Tensor:
        """Expose an encoder capability when the injected model deliberately provides it."""
        encode = getattr(self.model, "encode", None)
        if not callable(encode):
            raise ModuleError("the configured model does not expose an encode() capability")
        result = encode(x)
        if not isinstance(result, Tensor):
            raise ModuleError("model.encode() must return a tensor")
        return result

    def _common_step(self, batch: Batch, stage: Stage) -> Tensor:
        """Execute and log the one objective contract shared by every training stage."""
        sample_ids = _batch_ids(batch)
        result = self.objective(self.model, batch, stage)
        values = _objective_values(result)
        prefix = "val" if stage == "validate" else stage
        for name, value in values.items():
            log_name = f"{prefix}/{name}"
            if isinstance(value, Metric):
                self._validate_metric(value, log_name)
            self.log(
                log_name,
                value,
                batch_size=len(sample_ids),
                on_step=stage == "train" and name == "loss",
                on_epoch=True,
                prog_bar=stage == "validate" and name == "loss",
            )
        loss = values["loss"]
        assert isinstance(loss, Tensor)
        return loss

    def _validate_metric(self, metric: Metric, log_name: str) -> None:
        if not any(module is metric for module in self.modules()):
            raise ModuleError(
                f"TorchMetric for {log_name!r} must be registered on the objective or model"
            )
        previous = self._metric_log_names.setdefault(id(metric), log_name)
        if previous != log_name:
            raise ModuleError(
                "each TorchMetric instance may have one Lightning log name; "
                f"{previous!r} and {log_name!r} need distinct metric instances"
            )

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        del batch_idx
        return self._common_step(batch, "train")

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        del batch_idx
        return self._common_step(batch, "validate")

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:
        del batch_idx
        return self._common_step(batch, "test")

    def predict_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> PredictionBatch:
        del batch_idx, dataloader_idx
        sample_ids = _batch_ids(batch)
        try:
            x = batch["x"]
        except KeyError:
            raise ModuleError("prediction batch must contain 'x'") from None
        prediction = self.model(x)
        if not isinstance(prediction, Tensor):
            raise ModuleError(f"model prediction must be a tensor, got {type(prediction).__name__}")
        if prediction.ndim == 0 or prediction.shape[0] != len(sample_ids):
            size = None if prediction.ndim == 0 else prediction.shape[0]
            raise ModuleError(
                f"model returned {size} predictions for {len(sample_ids)} sample_id values"
            )
        result = PredictionBatch(
            sample_id=sample_ids,
            prediction=prediction.detach(),
        )
        if "row" in batch:
            row = batch["row"]
            if not isinstance(row, Tensor):
                raise ModuleError(f"prediction row must be a tensor, got {type(row).__name__}")
            if row.ndim != 1 or row.shape[0] != len(sample_ids):
                size = None if row.ndim == 0 else row.shape[0]
                raise ModuleError(
                    f"prediction row must be one-dimensional with {len(sample_ids)} values; "
                    f"got shape {tuple(row.shape)} with leading size {size}"
                )
            if row.layout != torch.strided or row.dtype not in _INTEGER_ROW_DTYPES:
                raise ModuleError(
                    f"prediction row must be a dense integer tensor, got {row.layout} {row.dtype}"
                )
            result["row"] = row
        return result

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer_factory(self.parameters(), **self.optimizer_parameters)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ModuleError(
                f"optimizer_factory must return a native torch optimizer, "
                f"got {type(optimizer).__name__}"
            )
        if self.scheduler_factory is None:
            return optimizer
        scheduler = self.scheduler_factory(optimizer, **self.scheduler_parameters)
        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            configured_scheduler: torch.optim.lr_scheduler.LRScheduler | dict[str, Any] = (
                scheduler
            )
        elif isinstance(scheduler, Mapping):
            configured_scheduler = _scheduler_configuration(scheduler)
        else:
            raise ModuleError(
                "scheduler_factory must return a native torch scheduler or "
                f"Lightning scheduler mapping, got {type(scheduler).__name__}"
            )
        return {"optimizer": optimizer, "lr_scheduler": configured_scheduler}


def _scheduler_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "scheduler",
        "name",
        "interval",
        "frequency",
        "reduce_on_plateau",
        "monitor",
        "strict",
    }
    unknown = set(value) - allowed
    if unknown:
        fields = ", ".join(sorted(str(field) for field in unknown))
        raise ModuleError(f"Lightning scheduler mapping has unknown fields: {fields}")
    scheduler = value.get("scheduler")
    if not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
        raise ModuleError(
            "Lightning scheduler mapping requires 'scheduler' as a native torch scheduler"
        )
    interval = value.get("interval", "epoch")
    if interval not in {"step", "epoch"}:
        raise ModuleError("Lightning scheduler mapping interval must be 'step' or 'epoch'")
    frequency = value.get("frequency", 1)
    if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency < 1:
        raise ModuleError("Lightning scheduler mapping frequency must be a positive integer")
    for name in ("name", "monitor"):
        configured = value.get(name)
        if configured is not None and not isinstance(configured, str):
            raise ModuleError(f"Lightning scheduler mapping {name} must be a string")
    for name in ("reduce_on_plateau", "strict"):
        configured = value.get(name)
        if configured is not None and not isinstance(configured, bool):
            raise ModuleError(f"Lightning scheduler mapping {name} must be a boolean")
    return dict(value)


def _batch_ids(batch: object) -> list[str]:
    if not isinstance(batch, Mapping):
        raise ModuleError(f"training batch must be a mapping, got {type(batch).__name__}")
    identities = batch.get("sample_id")
    if isinstance(identities, str) or not isinstance(identities, Sequence):
        raise ModuleError("training batch must contain sample_id as a sequence of strings")
    result = list(identities)
    if not result or any(not isinstance(sample_id, str) for sample_id in result):
        raise ModuleError("training batch sample_id must be a non-empty sequence of strings")
    return cast("list[str]", result)


def _objective_values(result: object) -> dict[str, ObjectiveValue]:
    if not isinstance(result, Mapping):
        raise ModuleError(f"objective must return a mapping, got {type(result).__name__}")
    if "loss" not in result:
        raise ModuleError("objective result requires mandatory scalar tensor 'loss'")
    values: dict[str, ObjectiveValue] = {}
    for name, value in result.items():
        if not isinstance(name, str) or not name or "/" in name:
            raise ModuleError("objective result names must be non-empty strings without '/'")
        if name in {"loss_step", "loss_epoch"}:
            raise ModuleError(f"objective result name {name!r} is reserved by Lightning")
        if name != "loss" and isinstance(value, Metric):
            values[name] = value
            continue
        if not isinstance(value, Tensor):
            raise ModuleError(
                f"objective result {name!r} must be a tensor or TorchMetric, "
                f"got {type(value).__name__}"
            )
        if value.ndim != 0:
            raise ModuleError(
                f"objective result {name!r} must be scalar, got shape {tuple(value.shape)}"
            )
        values[name] = value
    return values


__all__ = [
    "Batch",
    "ComponentError",
    "DsioModule",
    "ModuleError",
    "Objective",
    "ObjectiveResult",
    "Stage",
    "export_encoder",
]
