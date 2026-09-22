"""Training augmentation lives at one accelerator-side Lightning seam."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import nn

from dsio.model.components import Jitter
from dsio.model.masking import SpanMask
from dsio.model.module import DsioModule
from dsio.train.augmentation import MaskedReconstruction, TwoView


class CaptureObjective(nn.Module):
    """Record exactly what reached the objective boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[Mapping[str, Any]] = []

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        stage: str,
    ) -> Mapping[str, torch.Tensor]:
        del stage
        self.seen.append(batch)
        return {"loss": model(batch["x"]).sum() * 0.0}


class AddOne(nn.Module):
    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        seed: int,
        epoch: int,
        step: int,
        identity: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del seed, epoch, step, identity
        return {**batch, "x": batch["x"] + 1}


def _batch() -> dict[str, Any]:
    return {
        "sample_id": ["sample-a", "sample-b"],
        "x": torch.arange(16, dtype=torch.float32).reshape(2, 1, 8),
        "y": torch.tensor([0, 1]),
        "row": torch.tensor([3, 7]),
    }


def test_training_step_is_the_only_augmented_lifecycle_path() -> None:
    objective = CaptureObjective()
    module = DsioModule(
        model=nn.Identity(),
        objective=objective,
        training_augmentation=AddOne(),
        augmentation_seed=17,
        augmentation_identity={"kind": "test", "component": "add-one"},
    )
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    batch = _batch()

    module.training_step(batch, 0)
    module.validation_step(batch, 0)
    module.test_step(batch, 0)
    prediction = module.predict_step(batch, 0)

    assert torch.equal(objective.seen[0]["x"], batch["x"] + 1)
    assert torch.equal(objective.seen[1]["x"], batch["x"])
    assert torch.equal(objective.seen[2]["x"], batch["x"])
    assert torch.equal(prediction["prediction"], batch["x"])
    assert torch.equal(batch["x"], _batch()["x"]), "augmentation must not mutate source data"


def test_two_view_replays_from_explicit_identity_without_touching_global_rng() -> None:
    augmentation = TwoView(Jitter(sigma=0.5), views=("online", "target"))
    batch = _batch()
    identity = {
        "kind": "two_view",
        "component": {
            "reference": "dsio.model.components:Jitter",
            "parameters": {"sigma": 0.5},
        },
        "views": ["online", "target"],
    }
    before = torch.random.get_rng_state().clone()

    first = augmentation(batch, seed=11, epoch=2, step=5, identity=identity)
    second = augmentation(batch, seed=11, epoch=2, step=5, identity=identity)

    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(first["x"], second["x"])
    assert not torch.equal(first["x"][:2], first["x"][2:])
    assert first["sample_id"] == ["sample-a", "sample-b"] * 2
    assert first["view_id"] == ["online", "online", "target", "target"]
    assert torch.equal(first["row"], torch.tensor([3, 7, 3, 7]))
    assert torch.equal(first["y"], torch.tensor([2, 3, 0, 1]))

    changed_seed = augmentation(batch, seed=12, epoch=2, step=5, identity=identity)
    changed_epoch = augmentation(batch, seed=11, epoch=3, step=5, identity=identity)
    changed_step = augmentation(batch, seed=11, epoch=2, step=6, identity=identity)
    changed_sample = augmentation(
        {**batch, "sample_id": ["sample-a", "sample-c"]},
        seed=11,
        epoch=2,
        step=5,
        identity=identity,
    )
    changed_component = augmentation(
        batch,
        seed=11,
        epoch=2,
        step=5,
        identity={**identity, "component": {"reference": "project:Other"}},
    )
    changed_view = TwoView(Jitter(sigma=0.5), views=("first", "second"))(
        batch,
        seed=11,
        epoch=2,
        step=5,
        identity={**identity, "views": ["first", "second"]},
    )
    assert not torch.equal(first["x"], changed_seed["x"])
    assert not torch.equal(first["x"], changed_epoch["x"])
    assert not torch.equal(first["x"], changed_step["x"])
    assert not torch.equal(first["x"], changed_sample["x"])
    assert not torch.equal(first["x"], changed_component["x"])
    assert not torch.equal(first["x"], changed_view["x"])


def test_masked_reconstruction_builds_the_existing_loss_contract_on_device() -> None:
    batch = _batch()
    augmentation = MaskedReconstruction(SpanMask(ratio=0.5, span=2))
    result = augmentation(
        batch,
        seed=11,
        epoch=2,
        step=5,
        identity={
            "kind": "masked_reconstruction",
            "component": {"reference": "dsio.model.masking:SpanMask"},
        },
    )

    hidden = ~torch.isnan(result["y"])
    assert result["x"].device == batch["x"].device
    assert result["y"].device == batch["x"].device
    assert hidden.any() and (~hidden).any()
    assert torch.equal(result["x"][hidden], torch.zeros_like(result["x"][hidden]))
    assert result["sample_id"] == batch["sample_id"]
    assert torch.equal(result["row"], batch["row"])


def test_two_view_rejects_ambiguous_view_identity() -> None:
    with pytest.raises(ValueError, match="distinct non-empty"):
        TwoView(Jitter(), views=("same", "same"))
