"""The supervised objective reproduces consumer losses exactly and weights samples soundly."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dsio.config.components import ComponentError
from dsio.experimental.model import RootMeanSquaredError, SupervisedObjective

MSE = {"reference": "torch.nn:MSELoss"}
MAE = {"reference": "torch.nn:L1Loss"}


def test_losses_and_metrics_match_the_consumer_objectives_bit_for_bit() -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    x, y = torch.randn(5, 3), torch.randn(5, 2)
    prediction = model(x)

    regression = SupervisedObjective(
        loss=MSE,
        metrics={
            "mae": MAE,
            "rmsle": {"reference": "dsio.experimental.model:RootMeanSquaredError"},
        },
    )(model, {"x": x, "y": y}, "train")
    assert torch.equal(regression["loss"], F.mse_loss(prediction, y.float()))
    assert torch.equal(regression["mae"], F.l1_loss(prediction, y.float()))
    assert torch.equal(regression["rmsle"], torch.sqrt(F.mse_loss(prediction, y).detach()))
    assert regression["loss"].requires_grad and not regression["mae"].requires_grad

    labels = torch.tensor([0, 1, 1, 0, 1])
    classification = SupervisedObjective(loss={"reference": "torch.nn:CrossEntropyLoss"})
    assert torch.equal(
        classification(model, {"x": x, "y": labels}, "train")["loss"],
        F.cross_entropy(prediction, labels.long()),
    )

    flags = torch.tensor([[1.0, 0.0]]).expand(5, 2)
    binary = SupervisedObjective(loss={"reference": "torch.nn:BCEWithLogitsLoss"})
    assert torch.equal(
        binary(model, {"x": x, "y": flags}, "train")["loss"],
        F.binary_cross_entropy_with_logits(prediction, flags.float()),
    )

    reconstruction = SupervisedObjective(loss=MSE, target="x")
    images = torch.rand(4, 28, 28)
    assert torch.equal(
        reconstruction(nn.Sigmoid(), {"x": images}, "train")["loss"],
        F.mse_loss(torch.sigmoid(images), images.float()),
    )


def test_class_weights_keep_native_semantics() -> None:
    logits, labels = torch.randn(6, 3), torch.tensor([0, 1, 2, 2, 1, 0])
    weights = [0.5, 2.0, 1.0]
    objective = SupervisedObjective(
        loss={"reference": "torch.nn:CrossEntropyLoss", "parameters": {"weight": weights}}
    )
    assert torch.equal(
        objective(nn.Identity(), {"x": logits, "y": labels}, "train")["loss"],
        F.cross_entropy(logits, labels, weight=torch.tensor(weights)),
    )
    assert "loss.weight" in dict(objective.named_buffers())  # moves with the module


def test_a_target_is_never_silently_broadcast_or_squeezed() -> None:
    objective = SupervisedObjective(loss=MSE)
    with pytest.raises(ValueError, match=r"target 'y' has shape \(4, 1\); prediction \(4,\)"):
        objective(nn.Identity(), {"x": torch.zeros(4), "y": torch.zeros(4, 1)}, "train")

    classes = {"x": torch.randn(4, 3), "y": torch.tensor([[0], [1], [2], [1]])}
    cross_entropy = {"reference": "torch.nn:CrossEntropyLoss"}
    with pytest.raises(ValueError, match="class-index target"):
        SupervisedObjective(loss=cross_entropy)(nn.Identity(), classes, "train")
    adapted = SupervisedObjective(loss=cross_entropy, target_shape=[], target_dtype="int64")
    assert torch.equal(
        adapted(nn.Identity(), classes, "train")["loss"],
        F.cross_entropy(classes["x"], classes["y"].reshape(-1)),
    )


def _weighted(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    objective = SupervisedObjective(loss=MSE, metrics={"mae": MAE}, sample_weighted=True)
    return objective(nn.Identity(), batch, "train")


def test_mean_one_sample_weights_are_invariant_to_micro_batch_partitions() -> None:
    torch.manual_seed(1)
    x, y = torch.randn(8, 1), torch.randn(8, 1)
    weights = torch.rand(8) + 0.1
    weights = weights / weights.mean()  # the training-role mean-1 rule
    full = _weighted({"x": x, "y": y, "sample_weight": weights})
    assert torch.allclose(full["loss"], ((x - y).pow(2).reshape(-1) * weights).mean())

    for sizes in ((4, 4), (2, 6), (3, 5), (1, 7)):
        parts = [
            _weighted({"x": xs, "y": ys, "sample_weight": ws})
            for xs, ys, ws in zip(
                x.split(list(sizes)), y.split(list(sizes)), weights.split(list(sizes)), strict=True
            )
        ]
        for name in ("loss", "mae"):
            # DsioModule logs each micro-batch value weighted by its batch size.
            logged = sum(size * part[name] for size, part in zip(sizes, parts, strict=True)) / 8
            assert torch.allclose(logged, full[name]), (sizes, name)

    # Per-micro-batch /sum(w) normalization would make the loss depend on the partition.
    halves = [
        ((xs - ys).pow(2).reshape(-1) * ws).sum() / ws.sum()
        for xs, ys, ws in zip(x.split([1, 7]), y.split([1, 7]), weights.split([1, 7]), strict=True)
    ]
    assert not torch.allclose((halves[0] + 7 * halves[1]) / 8, full["loss"])


def test_metrics_follow_the_configured_stages() -> None:
    objective = SupervisedObjective(loss=MSE, metrics={"mae": MAE}, metric_stages=["validate"])
    batch = {"x": torch.zeros(2, 1), "y": torch.ones(2, 1)}
    assert set(objective(nn.Identity(), batch, "train")) == {"loss"}
    assert set(objective(nn.Identity(), batch, "validate")) == {"loss", "mae"}


def test_configuration_and_batches_are_validated() -> None:
    with pytest.raises(ValueError, match="reduction is owned by the objective"):
        SupervisedObjective(
            loss={"reference": "torch.nn:MSELoss", "parameters": {"reduction": "sum"}}
        )
    with pytest.raises(ValueError, match="target_dtype"):
        SupervisedObjective(loss=MSE, target_dtype="float16")
    with pytest.raises(ValueError, match="metric_stages"):
        SupervisedObjective(loss=MSE, metric_stages=["predict"])
    with pytest.raises(ValueError, match="not 'loss'"):
        SupervisedObjective(loss=MSE, metrics={"loss": MAE})
    with pytest.raises(ComponentError, match=r"metric 'rmse'.*reduction='none'"):
        SupervisedObjective(
            loss=MSE,
            metrics={"rmse": {"reference": "dsio.experimental.model:RootMeanSquaredError"}},
            sample_weighted=True,
        )

    batch = {"x": torch.zeros(2, 1), "y": torch.ones(2, 1)}
    with pytest.raises(ValueError, match="prediction tensor, got tuple"):
        SupervisedObjective(loss=MSE)(nn.LSTM(1, 1, batch_first=True), batch, "train")
    with pytest.raises(ValueError, match="target field 'z' is missing"):
        SupervisedObjective(loss=MSE, target="z")(nn.Identity(), batch, "train")
    with pytest.raises(ValueError, match="needs a sample_weight tensor"):
        _weighted(batch)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _weighted({**batch, "sample_weight": torch.tensor([1.0, -1.0])})
    with pytest.raises(ValueError, match="3 values for 2 samples"):
        _weighted({**batch, "sample_weight": torch.ones(3)})
    assert RootMeanSquaredError()(torch.ones(2), torch.ones(2)).item() == 0.0
