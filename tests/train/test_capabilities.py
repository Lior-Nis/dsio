"""The pre-fit proof uses the real Lightning path without becoming a second trainer."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

pytest.importorskip("lightning")

from lightning import Trainer  # noqa: E402

from dsio.model.chain import ComponentChain, LossObjective  # noqa: E402
from dsio.model.components import CrossEntropy  # noqa: E402
from dsio.model.module import DsioModule  # noqa: E402
from dsio.train.capabilities import (  # noqa: E402
    CapabilityError,
    check_requested_capabilities,
    check_training_capabilities,
    representative_batch,
)
from dsio.train.trainer import TrainerConfig  # noqa: E402


class _BatchDataset(Dataset[dict[str, Any]]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "sample_id": f"sample-{index}",
            "row": index,
            "x": torch.full((1, 8), float(index)),
            "y": torch.tensor(index % 2, dtype=torch.float32),
        }


class UnavailableObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: dict[str, Any], stage: str
    ) -> dict[str, Tensor]:
        del model, batch, stage
        raise RuntimeError("imaginary_fft is unavailable")


def _module(
    *, objective: nn.Module | None = None, backbone: nn.Module | None = None
) -> DsioModule:
    return DsioModule(
        model=ComponentChain(
            backbone=backbone or nn.Flatten(),
            head=nn.Linear(8, 2),
        ),
        objective=objective or LossObjective(CrossEntropy(threshold=0.5)),
    )


def _batch() -> dict[str, Any]:
    return {
        "sample_id": ["a", "b"],
        "row": torch.tensor([0, 1]),
        "x": torch.arange(16, dtype=torch.float32).reshape(2, 1, 8),
        "y": torch.tensor([0.0, 1.0]),
    }


def _trainer(*, precision: Any = "32-true") -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


def test_supported_path_is_proved_on_the_resolved_device_without_changing_mode() -> None:
    normalizer = nn.BatchNorm1d(8)
    module = _module(
        backbone=nn.Sequential(nn.Flatten(), normalizer, nn.Dropout(p=0.5))
    )
    module.train()
    rng = torch.random.get_rng_state().clone()
    assert normalizer.running_mean is not None
    running_mean = normalizer.running_mean.clone()
    evidence = check_training_capabilities(
        _trainer(),
        module,
        _batch(),
        requested=TrainerConfig(accelerator="auto", devices="auto"),
    )

    assert module.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(normalizer.running_mean, running_mean)
    assert all(parameter.grad is None for parameter in module.parameters())
    assert evidence["requested_accelerator"] == "auto"
    assert evidence["resolved_device"] == "cpu"
    assert evidence["resolved_precision"] == "32-true"
    assert evidence["resolved_devices"] == "1"
    assert evidence["torch_version"] == torch.__version__


def test_unverified_precision_fails_closed_before_the_objective_runs() -> None:
    objective = UnavailableObjective()
    with pytest.raises(CapabilityError) as caught:
        check_training_capabilities(
            _trainer(precision="64-true"),
            _module(objective=objective),
            _batch(),
            requested=TrainerConfig(accelerator="cpu", devices=1, precision="64-true"),
        )

    message = str(caught.value)
    assert "precision '64-true'" in message
    assert "experimental admission" in message
    assert "imaginary_fft" not in message


def test_an_unavailable_declared_accelerator_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(CapabilityError) as caught:
        check_requested_capabilities(TrainerConfig(accelerator="cuda", devices=1))

    message = str(caught.value)
    assert "accelerator 'cuda' is unavailable" in message
    assert "will not fall back to CPU" in message


def test_an_unavailable_operation_names_the_configured_boundary() -> None:
    with pytest.raises(CapabilityError) as caught:
        check_training_capabilities(
            _trainer(),
            _module(objective=UnavailableObjective()),
            _batch(),
            requested=TrainerConfig(accelerator="cpu", devices=1),
        )

    message = str(caught.value)
    assert "model/objective" in message
    assert "UnavailableObjective" in message
    assert "imaginary_fft is unavailable" in message
    assert "experimental admission" in message
    assert "fallback" not in message.lower()


def test_representative_batch_does_not_advance_the_training_order() -> None:
    left_generator = torch.Generator().manual_seed(41)
    right_generator = torch.Generator().manual_seed(41)
    loader = DataLoader(
        _BatchDataset(), batch_size=3, shuffle=True, generator=left_generator
    )
    control = DataLoader(
        _BatchDataset(), batch_size=3, shuffle=True, generator=right_generator
    )

    probed = representative_batch(loader)

    assert len(probed["sample_id"]) == 3
    assert next(iter(loader))["sample_id"] == next(iter(control))["sample_id"]
