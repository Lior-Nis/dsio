"""The pre-fit proof uses the real Lightning path without becoming a second trainer."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

pytest.importorskip("lightning")

from lightning import Trainer  # noqa: E402

import dsio.train.capabilities as capabilities  # noqa: E402
from dsio.model.chain import ComponentChain, LossObjective  # noqa: E402
from dsio.model.components import CrossEntropy, EmbeddingEncoder  # noqa: E402
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


class DetachedObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: dict[str, Any], stage: str
    ) -> dict[str, Tensor]:
        del model, batch, stage
        return {"loss": torch.tensor(1.0)}


class StatefulObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: dict[str, Any], stage: str
    ) -> dict[str, Tensor]:
        del stage
        random.random()
        np.random.random()
        prediction = model(batch["x"])
        with torch.no_grad():
            next(model.parameters()).add_(100)
        return {"loss": prediction.sum()}


class InvalidExtraOutputObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: dict[str, Any], stage: str
    ) -> dict[str, Tensor]:
        del stage
        return {"loss": model(batch["x"]).sum(), "vector": torch.ones(2)}


def _module(
    *,
    objective: nn.Module | None = None,
    backbone: nn.Module | None = None,
    optimizer_parameters: dict[str, Any] | None = None,
) -> DsioModule:
    return DsioModule(
        model=ComponentChain(
            backbone=backbone or nn.Flatten(),
            head=nn.Linear(8, 2),
        ),
        objective=objective or LossObjective(CrossEntropy(threshold=0.5)),
        optimizer_parameters=optimizer_parameters,
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


@pytest.mark.parametrize("devices", ["2", "-1", "0,1"])
def test_string_multi_device_selectors_are_not_a_single_device(devices: str) -> None:
    with pytest.raises(CapabilityError, match="single-device"):
        check_requested_capabilities(TrainerConfig(accelerator="cpu", devices=devices))


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


def test_a_detached_training_loss_is_rejected_before_fit() -> None:
    with pytest.raises(CapabilityError) as caught:
        check_training_capabilities(
            _trainer(),
            _module(objective=DetachedObjective()),
            _batch(),
            requested=TrainerConfig(accelerator="cpu", devices=1),
        )

    assert "does not require gradients" in str(caught.value)


def test_the_complete_objective_result_contract_is_checked() -> None:
    with pytest.raises(CapabilityError, match="objective result 'vector'.*scalar"):
        check_training_capabilities(
            _trainer(),
            _module(objective=InvalidExtraOutputObjective()),
            _batch(),
            requested=TrainerConfig(accelerator="cpu", devices=1),
        )


def test_optimizer_construction_is_part_of_the_proof() -> None:
    with pytest.raises(CapabilityError) as caught:
        check_training_capabilities(
            _trainer(),
            _module(optimizer_parameters={"bogus": 1}),
            _batch(),
            requested=TrainerConfig(accelerator="cpu", devices=1),
        )

    message = str(caught.value)
    assert "optimizer/scheduler" in message
    assert "bogus" in message


def test_module_transfer_failures_keep_capability_context() -> None:
    meta_backbone = nn.Sequential(nn.Flatten(), nn.Linear(8, 8, device="meta"))

    with pytest.raises(CapabilityError) as caught:
        check_training_capabilities(
            _trainer(),
            _module(backbone=meta_backbone),
            _batch(),
            requested=TrainerConfig(accelerator="cpu", devices=1),
        )

    message = str(caught.value)
    assert "module transfer for ComponentChain + LossObjective to cpu/32-true" in message
    assert "experimental admission" in message


def test_the_probe_cannot_consume_lazy_state_from_the_real_module() -> None:
    encoder = EmbeddingEncoder(vocab_size=16, embed_dim=4, out_dim=8)
    module = _module(backbone=encoder)

    check_training_capabilities(
        _trainer(),
        module,
        _batch(),
        requested=TrainerConfig(accelerator="cpu", devices=1),
    )

    assert not encoder._range_checked


def test_the_probe_isolates_parameters_lazy_modules_and_seeded_rngs() -> None:
    lazy = nn.LazyLinear(8)
    module = _module(
        backbone=nn.Sequential(nn.Flatten(), lazy),
        objective=StatefulObjective(),
    )
    assert isinstance(module.model, ComponentChain)
    assert isinstance(module.model.head, nn.Linear)
    head = module.model.head
    original_head = head.weight.detach().clone()
    random.seed(17)
    np.random.seed(17)
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    check_training_capabilities(
        _trainer(),
        module,
        _batch(),
        requested=TrainerConfig(accelerator="cpu", devices=1),
    )

    assert lazy.has_uninitialized_params()
    torch.testing.assert_close(head.weight, original_head)
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]


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


def test_representative_batch_never_initializes_persistent_training_workers() -> None:
    left_generator = torch.Generator().manual_seed(41)
    right_generator = torch.Generator().manual_seed(41)
    loader = DataLoader(
        _BatchDataset(),
        batch_size=3,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        generator=left_generator,
    )
    control = DataLoader(
        _BatchDataset(),
        batch_size=3,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        generator=right_generator,
    )

    representative_batch(loader)
    left = iter(loader)
    right = iter(control)
    try:
        assert next(left)["sample_id"] == next(right)["sample_id"]
    finally:
        left._shutdown_workers()  # type: ignore[attr-defined]
        right._shutdown_workers()  # type: ignore[attr-defined]
        loader._iterator = None  # type: ignore[attr-defined]
        control._iterator = None  # type: ignore[attr-defined]


def test_cuda_zero_rng_is_never_replaced_by_the_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[int] = []

    @contextmanager
    def fake_fork_rng(*, devices: list[int]) -> Iterator[None]:
        captured.extend(devices)
        yield

    monkeypatch.setattr(capabilities.torch.random, "fork_rng", fake_fork_rng)
    monkeypatch.setattr(
        capabilities.torch.cuda,
        "current_device",
        lambda: pytest.fail("cuda:0 has an explicit index"),
    )

    with capabilities._rng_context(torch.device("cuda:0")):
        pass

    assert captured == [0]
