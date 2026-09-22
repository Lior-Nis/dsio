"""Prove the configured training path before handing it to Lightning's fit loop."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torchmetrics import Metric

if TYPE_CHECKING:
    from lightning import Trainer

    from dsio.model.module import DsioModule
    from dsio.train.trainer import TrainerConfig


class CapabilityError(ValueError):
    """The declared training path cannot be proved in this environment."""


def check_requested_capabilities(requested: TrainerConfig) -> None:
    """Reject a requested path that cannot enter the stable matrix at all."""
    errors: list[str] = []
    if requested.accelerator not in {"auto", "cpu", "cuda"}:
        errors.append(
            f"accelerator {requested.accelerator!r} is outside the stable auto/CPU/CUDA matrix"
        )
    if requested.accelerator == "cuda" and not torch.cuda.is_available():
        errors.append("requested accelerator 'cuda' is unavailable; DSIO will not fall back to CPU")
    if isinstance(requested.devices, int) and requested.devices != 1:
        errors.append(f"devices={requested.devices} is outside the stable single-device matrix")
    if requested.precision != "32-true":
        errors.append(
            f"precision {requested.precision!r} is outside the stable 32-true matrix"
        )
    if errors:
        raise CapabilityError(_message(errors))


def representative_batch(loader: Any) -> Any:
    """Read one real batch without changing the loader's later shuffle order."""
    generator = loader.generator
    generator_state = None if generator is None else generator.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        return next(iter(loader))
    except StopIteration:
        raise CapabilityError("training capability preflight requires a non-empty loader") from None
    finally:
        torch.random.set_rng_state(torch_state)
        if generator is not None and generator_state is not None:
            generator.set_state(generator_state)


def check_training_capabilities(
    trainer: Trainer,
    module: DsioModule,
    batch: Any,
    *,
    requested: TrainerConfig,
) -> dict[str, str]:
    """Exercise the exact assembled training boundary on Lightning's resolved path.

    Lightning remains the device and precision authority. DSIO only narrows the stable
    matrix and proves that the configured components can execute there; it never retries
    on another device or with another dtype.
    """
    device = trainer.strategy.root_device
    precision = str(trainer.precision)
    errors = _matrix_errors(device, trainer.num_devices, precision)
    if errors:
        raise CapabilityError(_message(errors))

    try:
        transferred = trainer.strategy.batch_to_device(batch, device)
    except Exception as error:  # noqa: BLE001 - report arbitrary component failures together
        errors.append(f"batch transfer to {device}: {type(error).__name__}: {error}")
        transferred = None

    original_modes = [(child, child.training) for child in module.modules()]
    buffers = {name: value.detach().clone() for name, value in module.named_buffers()}
    try:
        module.to(device)
        module.train()
        with _rng_context(device):
            if transferred is not None:
                augmented = transferred
                if module.training_augmentation is not None:
                    try:
                        with _precision_context(trainer):
                            augmented = module.training_augmentation(
                                transferred,
                                seed=module.augmentation_seed,
                                epoch=0,
                                step=0,
                                identity=module.augmentation_identity,
                            )
                    except Exception as error:  # noqa: BLE001 - component boundary
                        errors.append(
                            "training augmentation "
                            f"{_name(module.training_augmentation)} on {device}/{precision}: "
                            f"{type(error).__name__}: {error}"
                        )
                        augmented = None
                if augmented is not None:
                    try:
                        with _precision_context(trainer):
                            result = module.objective(module.model, augmented, "train")
                        loss = _require_loss(result)
                        if loss.requires_grad:
                            loss.backward()
                    except Exception as error:  # noqa: BLE001 - component boundary
                        errors.append(
                            "model/objective "
                            f"{_name(module.model)} + {_name(module.objective)} "
                            f"on {device}/{precision}: {type(error).__name__}: {error}"
                        )
    finally:
        current_buffers = dict(module.named_buffers())
        with torch.no_grad():
            for name, value in buffers.items():
                if name in current_buffers:
                    current_buffers[name].copy_(value.to(current_buffers[name].device))
        for metric in module.modules():
            if isinstance(metric, Metric):
                metric.reset()
        module.zero_grad(set_to_none=True)
        for child, training in original_modes:
            child.training = training

    if errors:
        raise CapabilityError(_message(errors))
    return _evidence(trainer, requested, device, precision)


def log_capabilities(logger: Any, evidence: Mapping[str, str]) -> None:
    """Log resolved execution evidence as native MLflow parameters."""
    logger.log_hyperparams({f"execution.{key}": value for key, value in evidence.items()})


def _matrix_errors(device: torch.device, devices: int, precision: str) -> list[str]:
    errors: list[str] = []
    if devices != 1:
        errors.append(f"devices={devices} is outside the stable single-device matrix")
    if device.type not in {"cpu", "cuda"}:
        errors.append(f"device {device.type!r} is outside the stable CPU/CUDA matrix")
    if precision != "32-true":
        errors.append(f"precision {precision!r} is outside the stable 32-true matrix")
    return errors


def _precision_context(trainer: Trainer) -> Any:
    context = getattr(trainer.precision_plugin, "forward_context", None)
    return context() if callable(context) else nullcontext()


def _rng_context(device: torch.device) -> Any:
    devices = [] if device.type != "cuda" else [device.index or torch.cuda.current_device()]
    return torch.random.fork_rng(devices=devices)


def _require_loss(result: Any) -> Tensor:
    if not isinstance(result, Mapping):
        raise TypeError(f"objective returned {type(result).__name__}, expected a mapping")
    loss = result.get("loss")
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        shape = None if not isinstance(loss, Tensor) else tuple(loss.shape)
        raise TypeError(f"objective loss must be a scalar tensor, got {shape!r}")
    return loss


def _name(component: Any) -> str:
    return type(component).__qualname__


def _message(errors: list[str]) -> str:
    details = "\n".join(f"- {error}" for error in errors)
    return (
        "training capability preflight failed:\n"
        f"{details}\n"
        "Unverified behavior must enter DSIO through the experimental admission path."
    )


def _evidence(
    trainer: Trainer,
    requested: TrainerConfig,
    device: torch.device,
    precision: str,
) -> dict[str, str]:
    cudnn = torch.backends.cudnn.version()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    return {
        "requested_accelerator": requested.accelerator,
        "requested_devices": str(requested.devices),
        "requested_precision": requested.precision,
        "resolved_device": str(device),
        "resolved_devices": str(trainer.num_devices),
        "resolved_precision": precision,
        "strategy": type(trainer.strategy).__qualname__,
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(cudnn),
        "device_name": device_name,
    }


__all__ = [
    "CapabilityError",
    "check_requested_capabilities",
    "check_training_capabilities",
    "log_capabilities",
    "representative_batch",
]
