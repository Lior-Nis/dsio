"""Compositions initialize and compute exactly like the hand-written models they replace."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dsio.experimental.model import MLP, Chain, Stages

ENCODER = {
    "reference": "dsio.experimental.model.compositions:MLP",
    "parameters": {"input_shape": [28, 28], "hidden": [32], "output": 16},
}


def _same(first: nn.Module, second: nn.Module, x: torch.Tensor) -> None:
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)
    assert torch.equal(first(x), second(x))


def test_mlp_matches_a_seeded_hand_written_sequential() -> None:
    torch.manual_seed(7)
    reference = nn.Sequential(
        nn.Flatten(), nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 4), nn.Softplus()
    )
    torch.manual_seed(7)
    model = MLP(input_shape=[3, 4], hidden=[32], output=4, output_activation="softplus")
    _same(reference, model, torch.randn(5, 3, 4))


def test_mlp_checks_shape_casts_integers_and_validates_configuration() -> None:
    model = MLP(input_shape=[1, 3], output=1)
    with pytest.raises(ValueError, match=r"expects \[batch, 1, 3\], got \(2, 3\)"):
        model(torch.ones(2, 3))
    assert model(torch.ones(2, 1, 3, dtype=torch.int64)).dtype == torch.float32
    with pytest.raises(ValueError, match="activation must be one of"):
        MLP(input_shape=[2], output=1, activation="swish")
    with pytest.raises(ValueError, match="positive integers"):
        MLP(input_shape=[2], output=1, hidden=[0])


def test_chain_reproduces_a_frozen_encoder_classifier() -> None:
    torch.manual_seed(3)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 16))
    classifier = nn.Linear(16, 10)
    torch.manual_seed(3)
    model = Chain(
        backbone=ENCODER,
        head={
            "reference": "torch.nn:Linear",
            "parameters": {"in_features": 16, "out_features": 10},
        },
        frozen_backbone=True,
    )
    x = torch.rand(4, 28, 28)
    with torch.no_grad():
        expected = classifier(encoder(x))
    assert torch.equal(model(x), expected)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    model(x).sum().backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert model.head.weight.grad is not None


def test_chain_reproduces_an_autoencoder_and_exposes_encode() -> None:
    torch.manual_seed(5)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 16))
    decoder = nn.Sequential(
        nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 784), nn.Sigmoid()
    )
    torch.manual_seed(5)
    model = Chain(
        backbone=ENCODER,
        head={
            "reference": "dsio.experimental.model.compositions:Stages",
            "parameters": {
                "stages": [
                    {"reference": "torch.nn:ReLU"},
                    {
                        "reference": "dsio.experimental.model.compositions:MLP",
                        "parameters": {
                            "input_shape": [16],
                            "hidden": [32],
                            "output": 784,
                            "output_activation": "sigmoid",
                        },
                    },
                    {
                        "reference": "torch.nn:Unflatten",
                        "parameters": {"dim": 1, "unflattened_size": [28, 28]},
                    },
                ]
            },
        },
    )
    x = torch.rand(3, 28, 28)
    assert torch.equal(model.encode(x), encoder(x))
    assert torch.equal(model(x), decoder(encoder(x)).reshape(-1, 28, 28))


def test_chain_preprocessor_runs_first_and_stages_require_components() -> None:
    model = Chain(
        preprocessor={
            "reference": "dsio.experimental.model.standardization:Standardize",
            "parameters": {"mean": [1.0], "scale": [2.0]},
        },
        backbone={"reference": "torch.nn:Identity"},
        head={"reference": "torch.nn:Identity"},
    )
    assert model(torch.tensor([[3.0]])).tolist() == [[1.0]]
    with pytest.raises(ValueError, match="at least one component"):
        Stages(stages=[])


def test_a_failing_stage_is_named_in_the_error() -> None:
    model = Chain(
        backbone=ENCODER,
        head={
            "reference": "dsio.experimental.model.compositions:Stages",
            "parameters": {
                "stages": [
                    {"reference": "torch.nn:ReLU"},
                    {
                        "reference": "torch.nn:Linear",
                        "parameters": {"in_features": 8, "out_features": 2},
                    },
                ]
            },
        },
    )
    with pytest.raises(RuntimeError) as raised:
        model(torch.rand(2, 28, 28))
    assert raised.value.__notes__ == [
        "in Stages stage 1 (Linear), input (2, 16)",
        "in Chain head (Stages), input (2, 16)",
    ]
    with pytest.raises(ValueError, match=r"MLP expects \[batch, 28, 28\]") as shape:
        model(torch.rand(2, 784))
    assert shape.value.__notes__ == ["in Chain backbone (MLP), input (2, 784)"]
