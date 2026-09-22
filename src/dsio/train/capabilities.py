"""Prove the configured training path before handing it to Lightning's fit loop."""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

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
    if requested.devices not in {1, "auto"}:
        errors.append(f"devices={requested.devices} is outside the stable single-device matrix")
    if requested.precision != "32-true":
        errors.append(
            f"precision {requested.precision!r} is outside the stable 32-true matrix"
        )
    if errors:
        raise CapabilityError(_message(errors))


def representative_batch(loader: Any) -> Any:
    """Read one real batch through a throwaway loader, leaving training untouched."""
    torch_state = torch.random.get_rng_state()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        probe = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=loader.collate_fn,
            drop_last=loader.drop_last,
        )
        return next(iter(probe))
    except StopIteration:
        raise CapabilityError("training capability preflight requires a non-empty loader") from None
    finally:
        torch.random.set_rng_state(torch_state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


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

    with _rng_context(device):
        try:
            probe = copy.deepcopy(module)
        except Exception as error:  # noqa: BLE001 - arbitrary native module state
            errors.append(f"module isolation: {type(error).__name__}: {error}")
            probe = None

        ready = probe is not None
        if probe is not None:
            try:
                probe.to(device)
                probe.train()
            except Exception as error:  # noqa: BLE001 - component boundary
                components = f"{_name(probe.model)} + {_name(probe.objective)}"
                if probe.training_augmentation is not None:
                    components += f" + {_name(probe.training_augmentation)}"
                errors.append(
                    f"module transfer for {components} to {device}/{precision}: "
                    f"{type(error).__name__}: {error}"
                )
                ready = False
        if probe is not None and ready and transferred is not None:
            augmented = transferred
            if probe.training_augmentation is not None:
                try:
                    with _precision_context(trainer):
                        augmented = probe.training_augmentation(
                            transferred,
                            seed=probe.augmentation_seed,
                            epoch=0,
                            step=0,
                            identity=probe.augmentation_identity,
                        )
                except Exception as error:  # noqa: BLE001 - component boundary
                    errors.append(
                        "training augmentation "
                        f"{_name(probe.training_augmentation)} on {device}/{precision}: "
                        f"{type(error).__name__}: {error}"
                    )
                    augmented = None
            if augmented is not None:
                try:
                    with _precision_context(trainer):
                        result = probe.objective(probe.model, augmented, "train")
                    values = probe.validate_objective_result(result)
                    loss = values["loss"]
                    if not isinstance(loss, Tensor) or not loss.requires_grad:
                        raise TypeError("objective loss does not require gradients")
                    loss.backward()
                except Exception as error:  # noqa: BLE001 - component boundary
                    errors.append(
                        "model/objective "
                        f"{_name(probe.model)} + {_name(probe.objective)} "
                        f"on {device}/{precision}: {type(error).__name__}: {error}"
                    )
        if probe is not None:
            try:
                probe.configure_optimizers()
            except Exception as error:  # noqa: BLE001 - component boundary
                errors.append(f"optimizer/scheduler: {type(error).__name__}: {error}")

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


@contextmanager
def _rng_context(device: torch.device) -> Iterator[None]:
    devices: list[int] = []
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        devices.append(index)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


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
