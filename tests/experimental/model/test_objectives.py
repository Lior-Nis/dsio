"""The supervised objective reproduces consumer losses exactly and weights samples soundly."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dsio.config.components import ComponentError, resolve_component
from dsio.experimental.model import MaskedObjective, RootMeanSquaredError, SupervisedObjective

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
    assert classification._sample_mean_loss is False
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
    assert objective._sample_mean_loss is False

    sample_weighted = SupervisedObjective(
        loss={"reference": "torch.nn:CrossEntropyLoss", "parameters": {"weight": weights}},
        sample_weighted=True,
    )
    assert sample_weighted._sample_mean_loss is True


def test_a_target_is_never_silently_broadcast_or_squeezed() -> None:
    objective = SupervisedObjective(loss=MSE)
    with pytest.raises(ValueError, match=r"target 'y' has shape \(4, 1\); prediction \(4,\)"):
        objective(nn.Identity(), {"x": torch.zeros(4), "y": torch.zeros(4, 1)}, "train")

    classes = {"x": torch.randn(4, 3), "y": torch.tensor([[0], [1], [2], [1]])}
    cross_entropy = {"reference": "torch.nn:CrossEntropyLoss"}
    with pytest.raises(ValueError, match=r"needs \(4,\)"):
        SupervisedObjective(loss=cross_entropy)(nn.Identity(), classes, "train")
    # Only class-index losses drop the class axis: an integer count target must match exactly.
    counts = {"x": torch.zeros(4, 1), "y": torch.tensor([1, 0, 2, 3])}
    with pytest.raises(ValueError, match=r"needs \(4, 1\)"):
        SupervisedObjective(loss={"reference": "torch.nn:PoissonNLLLoss"})(
            nn.Identity(), counts, "train"
        )
    labels = {"x": torch.randn(2, 3), "y": torch.tensor([[0, 2, -1], [1, -1, 0]])}
    multilabel = SupervisedObjective(loss={"reference": "torch.nn:MultiLabelMarginLoss"})
    assert torch.equal(
        multilabel(nn.Identity(), labels, "train")["loss"],
        F.multilabel_margin_loss(labels["x"], labels["y"]),
    )
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
    with pytest.raises(ValueError, match=r"\[.reduction.\] owned by the objective"):
        SupervisedObjective(
            loss={"reference": "torch.nn:MSELoss", "parameters": {"reduction": "sum"}}
        )
    with pytest.raises(ValueError, match="target_dtype"):
        SupervisedObjective(loss=MSE, target_dtype="float16")
    with pytest.raises(ValueError, match="metric_stages"):
        SupervisedObjective(loss=MSE, metric_stages=["predict"])
    with pytest.raises(ValueError, match="metric names must be identifiers"):
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


def test_equal_micro_batches_accumulate_the_full_batch_gradient() -> None:
    torch.manual_seed(2)
    model = nn.Linear(2, 1)
    x, y = torch.randn(8, 2), torch.randn(8, 1)
    weights = torch.rand(8) + 0.5
    weights = weights / weights.mean()
    objective = SupervisedObjective(loss=MSE, sample_weighted=True)

    objective(model, {"x": x, "y": y, "sample_weight": weights}, "train")["loss"].backward()
    full = [parameter.grad.clone() for parameter in model.parameters()]
    model.zero_grad()
    for part in range(2):  # Lightning divides each accumulated loss by the micro-batch count
        rows = slice(4 * part, 4 * part + 4)
        batch = {"x": x[rows], "y": y[rows], "sample_weight": weights[rows]}
        (objective(model, batch, "train")["loss"] / 2).backward()
    for expected, parameter in zip(full, model.parameters(), strict=True):
        assert torch.allclose(parameter.grad, expected)


def test_sample_weights_scale_class_weighted_losses_and_refuse_ignored_targets() -> None:
    logits, labels = torch.randn(4, 3), torch.tensor([0, 1, 2, 1])
    weights = torch.tensor([0.5, 1.5, 1.0, 1.0])
    objective = SupervisedObjective(
        loss={"reference": "torch.nn:CrossEntropyLoss", "parameters": {"weight": [1.0, 2.0, 0.5]}},
        sample_weighted=True,
    )
    batch = {"x": logits, "y": labels, "sample_weight": weights}
    per_sample = F.cross_entropy(
        logits, labels, weight=torch.tensor([1.0, 2.0, 0.5]), reduction="none"
    )
    assert torch.equal(
        objective(nn.Identity(), batch, "train")["loss"], (per_sample * weights).mean()
    )
    with pytest.raises(ValueError, match="ignore_index -100"):
        objective(nn.Identity(), {**batch, "y": torch.tensor([0, -100, 2, 1])}, "train")


def test_owned_parameters_names_and_stateful_metrics_are_refused() -> None:
    for parameters in ({"size_average": False}, {"reduce": False}):
        with pytest.raises(ValueError, match="owned by the objective"):
            SupervisedObjective(loss={"reference": "torch.nn:MSELoss", "parameters": parameters})
    for name in ("train", "top.1", "loss_step"):
        with pytest.raises(ValueError, match="metric names must be identifiers"):
            SupervisedObjective(loss=MSE, metrics={name: MAE})
    with pytest.raises(ValueError, match="stateful TorchMetrics"):
        SupervisedObjective(
            loss=MSE, metrics={"auc": {"reference": "torchmetrics.classification:BinaryAUROC"}}
        )
    scalar = SupervisedObjective(
        loss={"reference": "torch.nn:BCEWithLogitsLoss", "parameters": {"pos_weight": 2.0}}
    )
    batch = {"x": torch.zeros(2, 1), "y": torch.ones(2, 1)}
    assert torch.equal(
        scalar(nn.Identity(), batch, "train")["loss"],
        F.binary_cross_entropy_with_logits(batch["x"], batch["y"], pos_weight=torch.tensor([2.0])),
    )
    double = {**batch, "sample_weight": torch.ones(2, dtype=torch.float64)}
    assert _weighted(double)["loss"].dtype == torch.float32
    with pytest.raises(ValueError, match=r"\[batch\] or \[batch, 1\]"):
        _weighted({**batch, "sample_weight": torch.ones(1, 2)})


def test_masked_objective_matches_dense_bce_and_mse_consumers_exactly() -> None:
    logits = torch.tensor(
        [
            [[0.2, -0.4, 1.0], [0.7, -1.0, 0.1], [0.3, 0.2, -0.8]],
            [[-0.2, 0.4, 0.0], [1.7, -0.3, 0.5], [-0.1, 0.9, 0.6]],
        ]
    )
    labels = torch.tensor(
        [
            [[1, 0, 1], [0, 1, 0], [1, 1, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 1]],
        ]
    )
    point_mask = torch.tensor([[True, False, True], [False, True, True]])
    bce = MaskedObjective(loss={"reference": "torch.nn:BCEWithLogitsLoss"}, target_dtype="float32")
    assert torch.equal(
        bce(nn.Identity(), {"x": logits, "y": labels, "mask": point_mask}, "train")["loss"],
        F.binary_cross_entropy_with_logits(logits[point_mask], labels[point_mask].float()),
    )

    prediction = logits[..., 0]
    target = labels[..., 0].float()
    mse = MaskedObjective(
        loss=MSE,
        metrics={
            "rmse": {
                "reference": "dsio.experimental.model.objectives:RootMeanSquaredError",
                "parameters": {"scale": 20_000.0},
            }
        },
    )
    expected = F.mse_loss(prediction[point_mask], target[point_mask])
    result = mse(nn.Identity(), {"x": prediction, "y": target, "mask": point_mask}, "validate")
    assert torch.equal(result["loss"], expected)
    assert torch.equal(result["rmse"], torch.sqrt(expected.detach()) * 20_000.0)


def test_masked_values_never_reach_losses_metrics_or_gradients() -> None:
    prediction = torch.tensor([[1.0, float("nan")], [3.0, float("inf")]], requires_grad=True)
    target = torch.tensor([[2.0, float("nan")], [1.0, float("inf")]])
    mask = torch.tensor([[True, False], [True, False]])
    objective = MaskedObjective(
        loss=MSE,
        metrics={"mae": MAE},
    )

    result = objective(nn.Identity(), {"x": prediction, "y": target, "mask": mask}, "train")
    assert torch.equal(result["loss"], torch.tensor(2.5))
    assert torch.equal(result["mae"], torch.tensor(1.5))
    assert not result["mae"].requires_grad
    result["loss"].backward()
    assert torch.equal(prediction.grad, torch.tensor([[-1.0, 0.0], [2.0, 0.0]]))

    changed = objective(
        nn.Identity(),
        {
            "x": torch.tensor([[1.0, -9_999.0], [3.0, 9_999.0]]),
            "y": torch.tensor([[2.0, -8_888.0], [1.0, 8_888.0]]),
            "mask": mask,
        },
        "train",
    )
    assert torch.equal(changed["loss"], result["loss"])
    assert torch.equal(changed["mae"], result["mae"])


def test_masked_objective_validates_the_reserved_batch_contract() -> None:
    objective = MaskedObjective(loss=MSE, metrics={"mae": MAE}, metric_stages=["validate"])
    batch = {
        "x": torch.ones(2, 3),
        "y": torch.zeros(2, 3),
        "mask": torch.tensor([[True, False, True], [True, True, False]]),
    }
    assert set(objective(nn.Identity(), batch, "train")) == {"loss"}
    assert set(objective(nn.Identity(), batch, "validate")) == {"loss", "mae"}
    assert objective._sample_mean_loss is False

    with pytest.raises(ValueError, match="stage must be"):
        objective(nn.Identity(), batch, "validation")
    with pytest.raises(ValueError, match="input field 'x' is missing"):
        objective(nn.Identity(), {"y": batch["y"], "mask": batch["mask"]}, "train")
    with pytest.raises(ValueError, match="no valid positions"):
        objective(nn.Identity(), {**batch, "mask": torch.zeros(2, 3, dtype=torch.bool)}, "train")
    with pytest.raises(ValueError, match="batch field 'mask' is missing"):
        objective(nn.Identity(), {"x": batch["x"], "y": batch["y"]}, "train")
    with pytest.raises(ValueError, match="mask must be a boolean tensor"):
        objective(nn.Identity(), {**batch, "mask": batch["mask"].long()}, "train")
    with pytest.raises(ValueError, match=r"mask shape \(2, 1\).+\(2, 3\)"):
        objective(nn.Identity(), {**batch, "mask": torch.ones(2, 1, dtype=torch.bool)}, "train")
    with pytest.raises(ValueError, match="target field 'y' is missing"):
        objective(nn.Identity(), {"x": batch["x"], "mask": batch["mask"]}, "train")
    with pytest.raises(ValueError, match="target 'y' must be a tensor"):
        objective(nn.Identity(), {**batch, "y": [0, 1]}, "train")
    with pytest.raises(ValueError, match=r"target 'y' has shape \(2, 1\); prediction \(2, 3\)"):
        objective(nn.Identity(), {**batch, "y": torch.zeros(2, 1)}, "train")
    with pytest.raises(ValueError, match="prediction tensor, got tuple"):
        objective(
            nn.LSTM(3, 3, batch_first=True),
            {"x": torch.ones(2, 1, 3), "y": torch.zeros(2, 1, 3), "mask": batch["mask"]},
            "train",
        )


def test_masked_objective_preserves_output_axes_and_scalar_results() -> None:
    prediction = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prefix_mask = torch.tensor([True, False])
    weighted = MaskedObjective(
        loss={
            "reference": "torch.nn:BCEWithLogitsLoss",
            "parameters": {"pos_weight": [1.0, 2.0]},
        }
    )
    assert torch.equal(
        weighted(nn.Identity(), {"x": prediction, "y": target, "mask": prefix_mask}, "train")[
            "loss"
        ],
        F.binary_cross_entropy_with_logits(
            prediction[prefix_mask], target[prefix_mask], pos_weight=torch.tensor([1.0, 2.0])
        ),
    )
    with pytest.raises(ValueError, match="full-shape mask.+pos_weight"):
        weighted(
            nn.Identity(),
            {"x": prediction, "y": target, "mask": torch.ones_like(target, dtype=torch.bool)},
            "train",
        )

    vector_metric = MaskedObjective(
        loss=MSE,
        metrics={"similarity": {"reference": "torch.nn:CosineSimilarity"}},
    )
    with pytest.raises(ValueError, match="metric 'similarity' must return a scalar tensor"):
        vector_metric(nn.Identity(), {"x": prediction, "y": target, "mask": prefix_mask}, "train")

    with pytest.raises(ComponentError, match="loss:.*reduction"):
        MaskedObjective(loss={"reference": "torch.nn:CosineSimilarity"})


def test_masked_objective_handles_rank_one_and_empty_final_axes() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([0.0, 2.0, 4.0])
    result = MaskedObjective(loss=MSE)(
        nn.Identity(),
        {"x": prediction, "y": target, "mask": torch.tensor(True)},
        "train",
    )
    assert torch.equal(result["loss"], F.mse_loss(prediction, target))

    empty = torch.empty(2, 0)
    with pytest.raises(ValueError, match="selected no values"):
        MaskedObjective(loss=MSE)(
            nn.Identity(),
            {"x": empty, "y": empty.clone(), "mask": torch.tensor([True, False])},
            "train",
        )


def test_masked_objective_is_importable_and_rejects_invalid_configuration() -> None:
    configured = resolve_component(
        {
            "reference": "dsio.experimental.model.masked_objective:MaskedObjective",
            "parameters": {
                "loss": {"reference": "torch.nn:MSELoss"},
                "target_dtype": "float32",
            },
        },
        expected=MaskedObjective,
    )
    assert isinstance(configured, MaskedObjective)

    with pytest.raises(ValueError, match="target_dtype"):
        MaskedObjective(loss=MSE, target_dtype="float16")
    with pytest.raises(ValueError, match="metric_stages"):
        MaskedObjective(loss=MSE, metric_stages=["predict"])
    with pytest.raises(ValueError, match="metric names must be identifiers"):
        MaskedObjective(loss=MSE, metrics={"loss": MAE})
    with pytest.raises(ValueError, match=r"\[.reduction.\] owned by the objective"):
        MaskedObjective(loss={"reference": "torch.nn:MSELoss", "parameters": {"reduction": "sum"}})
    for scale in (0, -1, True, float("inf"), 10**1000):
        with pytest.raises(ValueError, match="scale must be finite and positive"):
            RootMeanSquaredError(scale=scale)
