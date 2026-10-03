"""Outputs reproduce the consumer normalizers, name every violation, and survive MLflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import mlflow
import numpy as np
import pytest
import torch
from mlflow.tracking import MlflowClient
from torch import Tensor, nn

from dsio.experimental.inference import (
    BinaryOutput,
    MulticlassOutput,
    PredictionViolation,
    RegressionOutput,
)
from dsio.inference import Predictor, log_predictor


def test_outputs_reproduce_the_consumer_normalizers_bit_for_bit() -> None:
    logits = torch.randn(6, 1)
    binary = BinaryOutput()(logits)
    score = torch.sigmoid(logits).reshape(-1)
    assert torch.equal(binary["score"], score)
    assert torch.equal(binary["prediction"], (score >= 0.5).to(torch.int64))

    classes = torch.randn(5, 10)
    assert torch.equal(MulticlassOutput(classes=10)(classes)["prediction"], classes.argmax(dim=1))
    ordinal = MulticlassOutput(classes=6, scores=True, label_offset=1)(classes[:, :6])
    probabilities = torch.softmax(classes[:, :6], dim=1)
    assert torch.equal(ordinal["score"], probabilities)
    assert torch.equal(ordinal["prediction"], probabilities.argmax(dim=1) + 1)

    # Store Sales: softplus log-space outputs, so its former clamp_min(0) never changed a value.
    log_values = nn.functional.softplus(torch.randn(4, 7))
    store = RegressionOutput(shape=[7], inverse="expm1", non_negative=True, raw_field="log")(
        log_values
    )
    assert torch.equal(store["prediction"], torch.expm1(log_values).clamp_min(0))
    assert store["log"] is log_values

    dense_logits = torch.randn(2, 7, 3)
    dense_binary = BinaryOutput(shape=[None, 3], score_field="probability")(dense_logits)
    probability = torch.sigmoid(dense_logits)
    assert torch.equal(dense_binary["probability"], probability)
    assert torch.equal(dense_binary["prediction"], (probability >= 0.5).to(torch.int64))

    normalized = torch.randn(2, 7)
    dense_regression = RegressionOutput(shape=[None], scale=20_000)(normalized)
    assert torch.equal(dense_regression["prediction"], normalized * 20_000)


def _violation(validate: Callable[[Mapping[str, Any]], None], output: Mapping[str, Any]) -> str:
    with pytest.raises(PredictionViolation) as raised:
        validate(output)
    return raised.value.kind


def test_binary_violations_are_named() -> None:
    validate = BinaryOutput(threshold=0.5).validator
    ok = {"prediction": torch.tensor([1, 0]), "score": torch.tensor([0.9, 0.2])}
    validate(ok)
    assert _violation(validate, {**ok, "score": torch.tensor([0.9])}) == "shape"
    assert _violation(validate, {**ok, "prediction": torch.tensor([1.0, 0.0])}) == "dtype"
    assert _violation(validate, {**ok, "score": torch.tensor([0.9, torch.nan])}) == "finiteness"
    assert _violation(validate, {**ok, "score": torch.tensor([1.5, 0.2])}) == "range"
    assert _violation(validate, {**ok, "prediction": torch.tensor([2, 0])}) == "range"
    assert _violation(validate, {**ok, "prediction": torch.tensor([0, 0])}) == (
        "threshold consistency"
    )
    with pytest.raises(PredictionViolation, match="shape"):
        BinaryOutput()(torch.zeros(2, 3))
    with pytest.raises(PredictionViolation) as raised:
        BinaryOutput()(torch.tensor([torch.inf]))
    assert raised.value.kind == "finiteness"
    with pytest.raises(PredictionViolation) as raised:
        BinaryOutput()(torch.tensor([1], dtype=torch.int64))
    assert raised.value.kind == "dtype"


def test_dense_binary_uses_a_non_default_threshold() -> None:
    probability = torch.tensor([[[0.8, 0.6]]])
    logits = torch.logit(probability)
    result = BinaryOutput(threshold=0.75, shape=[None, 2])(logits)

    assert result["prediction"].tolist() == [[[1, 0]]]


def test_dense_binary_violations_are_named() -> None:
    output = BinaryOutput(shape=[None, 3], score_field="probability")
    probability = torch.rand(2, 4, 3)
    ok = {
        "prediction": (probability >= 0.5).to(torch.int64),
        "probability": probability,
    }
    output.validator(ok)
    assert _violation(output.validator, {**ok, "probability": probability[:, :, :2]}) == "shape"
    assert _violation(output.validator, {**ok, "prediction": ok["prediction"].float()}) == ("dtype")
    assert (
        _violation(output.validator, {**ok, "probability": probability.clone().fill_(torch.nan)})
        == "finiteness"
    )
    assert _violation(output.validator, {**ok, "probability": probability + 1}) == "range"
    assert _violation(output.validator, {**ok, "prediction": ok["prediction"] + 2}) == "range"
    assert (
        _violation(output.validator, {**ok, "prediction": 1 - ok["prediction"]})
        == "threshold consistency"
    )
    with pytest.raises(PredictionViolation) as raised:
        output(torch.zeros(2, 4, 2))
    assert raised.value.kind == "shape"
    with pytest.raises(PredictionViolation) as raised:
        output(torch.empty(2, 0, 3))
    assert raised.value.kind == "shape"


def test_multiclass_violations_are_named() -> None:
    validate = MulticlassOutput(classes=3, scores=True, label_offset=1).validator
    score = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    ok = {"prediction": torch.tensor([1, 3]), "score": score}
    validate(ok)
    assert _violation(validate, {**ok, "score": score[:, :2]}) == "shape"
    assert _violation(validate, {**ok, "score": score.clone().fill_(torch.nan)}) == "finiteness"
    assert _violation(validate, {**ok, "prediction": torch.tensor([0, 3])}) == "range"
    assert _violation(validate, {**ok, "score": score * 2}) == "simplex"
    negative = torch.tensor([[1.2, -0.1, -0.1], [0.1, 0.1, 0.8]])
    assert _violation(validate, {**ok, "score": negative}) == "simplex"
    assert _violation(validate, {**ok, "prediction": torch.tensor([2, 3])}) == (
        "argmax consistency"
    )
    with pytest.raises(PredictionViolation) as raised:
        MulticlassOutput(classes=3)(torch.tensor([[0.0, torch.nan, 1.0]]))
    assert raised.value.kind == "finiteness"


def test_regression_violations_are_named() -> None:
    validate = RegressionOutput(shape=[2], non_negative=True, raw_field="raw").validator
    ok = {"prediction": torch.ones(3, 2), "raw": torch.ones(3, 2)}
    validate(ok)
    assert _violation(validate, {**ok, "prediction": torch.ones(3, 1)}) == "shape"
    assert _violation(validate, {**ok, "raw": torch.ones(3)}) == "shape"
    assert _violation(validate, {**ok, "prediction": torch.ones(3, 2, dtype=torch.int64)}) == (
        "dtype"
    )
    assert _violation(validate, {**ok, "prediction": torch.full((3, 2), torch.inf)}) == (
        "finiteness"
    )
    assert _violation(validate, {**ok, "prediction": -torch.ones(3, 2)}) == "sign"
    # Without non_negative, negative values are legitimate.
    RegressionOutput(shape=[2]).validator({"prediction": -torch.ones(3, 2)})


def test_regression_rejects_nonfinite_or_nonfloating_model_values() -> None:
    output = RegressionOutput(shape=[1], inverse="expm1")
    for value in (torch.inf, -torch.inf, torch.nan):
        with pytest.raises(PredictionViolation) as raised:
            output(torch.tensor([[value]]))
        assert raised.value.kind == "finiteness"
    with pytest.raises(PredictionViolation) as raised:
        output(torch.ones(1, 1, dtype=torch.int64))
    assert raised.value.kind == "dtype"


def test_regression_validator_checks_batch_and_transform_consistency() -> None:
    validate = RegressionOutput(shape=[None], inverse="expm1", scale=2, raw_field="raw").validator
    raw = torch.zeros(3, 2)
    ok = {"prediction": torch.zeros(3, 2), "raw": raw}
    validate(ok)
    assert _violation(validate, {**ok, "raw": raw[:2]}) == "shape"
    assert _violation(validate, {**ok, "raw": torch.zeros(3, 4)}) == "shape"
    assert _violation(validate, {**ok, "prediction": torch.ones(3, 2)}) == ("transform consistency")


def test_dense_regression_accepts_dynamic_time_and_rejects_other_shapes() -> None:
    output = RegressionOutput(shape=[None], scale=20_000)
    for points in (1, 7):
        result = output(torch.ones(2, points))
        assert result["prediction"].shape == (2, points)
        output.validator(result)
    with pytest.raises(PredictionViolation) as raised:
        output(torch.ones(2, 7, 1))
    assert raised.value.kind == "shape"
    with pytest.raises(PredictionViolation) as raised:
        output(torch.empty(2, 0))
    assert raised.value.kind == "shape"


@pytest.mark.parametrize(
    ("output", "values", "fields"),
    [
        (BinaryOutput(), torch.tensor([[2.0], [-1.0]]), {"prediction", "score"}),
        (
            MulticlassOutput(classes=3, scores=True, label_offset=1),
            torch.eye(3)[:2] * 4,
            {
                "prediction",
                "score",
            },
        ),
        (
            RegressionOutput(shape=[2], inverse="expm1", non_negative=True, raw_field="raw"),
            torch.tensor([[0.5, 1.0], [2.0, 0.0]]),
            {"prediction", "raw"},
        ),
    ],
)
def test_outputs_round_trip_through_mlflow_with_declared_signatures(
    output: nn.Module, values: Tensor, fields: set[str]
) -> None:
    predictor = Predictor(
        model=nn.Identity(),
        preprocessor=nn.Identity(),
        normalizer=output,
        validator=output.validator,
        checkpoint_uri="runs:/training/checkpoint",
        checkpoint_digest="c" * 64,
    )
    example = {"sample_id": ["a", "b"], "x": values}
    expected = predictor(example)
    client = MlflowClient()
    run_id = client.create_run(client.create_experiment("outputs")).info.run_id
    models = log_predictor(
        predictor, run_id=run_id, input_example=example, forms=("pytorch", "pyfunc")
    )

    for info in models.values():
        assert info.signature is not None and info.signature.outputs is not None
        assert set(info.signature.outputs.input_names()) == {"sample_id", *fields}
    native = mlflow.pytorch.load_model(models["pytorch"].model_uri)(example)
    pyfunc = mlflow.pyfunc.load_model(models["pyfunc"].model_uri).predict(
        {"sample_id": np.asarray(["a", "b"]), "x": values.numpy()}
    )
    for field in fields:
        assert torch.equal(native[field], expected[field])
        np.testing.assert_array_equal(pyfunc[field], expected[field].numpy())


@pytest.mark.parametrize(
    ("output", "example_shape", "other_shape", "fields"),
    [
        (
            BinaryOutput(shape=[None, 3], score_field="probability"),
            (2, 4, 3),
            (2, 7, 3),
            {"prediction", "probability"},
        ),
        (RegressionOutput(shape=[None], scale=20_000), (2, 4), (2, 7), {"prediction"}),
    ],
)
def test_dense_output_round_trip_preserves_dynamic_time_axes(
    output: Any,
    example_shape: tuple[int, ...],
    other_shape: tuple[int, ...],
    fields: set[str],
) -> None:
    predictor = Predictor(
        model=nn.Identity(),
        preprocessor=nn.Identity(),
        normalizer=output,
        validator=output.validator,
        checkpoint_uri="runs:/training/checkpoint",
        checkpoint_digest="d" * 64,
    )
    example = {"sample_id": ["a", "b"], "x": torch.randn(example_shape)}
    client = MlflowClient()
    run_id = client.create_run(client.create_experiment("dense-outputs")).info.run_id
    models = log_predictor(
        predictor,
        run_id=run_id,
        input_example=example,
        forms=("pytorch", "pyfunc"),
        dynamic_axes={field: [1] for field in {"x", *fields}},
    )

    for info in models.values():
        assert info.signature is not None and info.signature.outputs is not None
        assert set(info.signature.outputs.input_names()) == {"sample_id", *fields}
    other = {"sample_id": ["c", "d"], "x": torch.randn(other_shape)}
    expected = predictor(other)
    native = mlflow.pytorch.load_model(models["pytorch"].model_uri)(other)
    pyfunc = mlflow.pyfunc.load_model(models["pyfunc"].model_uri).predict(
        {"sample_id": np.asarray(other["sample_id"]), "x": other["x"].numpy()}
    )
    for field in fields:
        assert torch.equal(native[field], expected[field])
        np.testing.assert_array_equal(pyfunc[field], expected[field].numpy())


def test_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="threshold"):
        BinaryOutput(threshold=1.0)
    with pytest.raises(ValueError, match="classes"):
        MulticlassOutput(classes=1)
    with pytest.raises(ValueError, match="inverse"):
        RegressionOutput(shape=[1], inverse="exp")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="raw_field"):
        RegressionOutput(shape=[1], raw_field="prediction")
    with pytest.raises(ValueError, match="shape"):
        RegressionOutput(shape=[0])
    with pytest.raises(ValueError, match="shape"):
        BinaryOutput(shape=[None, 0])
    with pytest.raises(ValueError, match="scale"):
        RegressionOutput(shape=[None], scale=0)
    for scale in (-1, float("inf"), float("nan"), 10**10_000):
        with pytest.raises(ValueError, match="scale"):
            RegressionOutput(shape=[None], scale=scale)
    with pytest.raises(ValueError, match="int64"):
        MulticlassOutput(classes=2, label_offset=2**63 - 1)
    with pytest.raises(ValueError, match="int64"):
        MulticlassOutput(classes=2, label_offset=-(2**63) - 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_scores_in_reduced_precision_pass_their_own_validator(dtype: torch.dtype) -> None:
    output = MulticlassOutput(classes=100, scores=True)
    output.validator(output(torch.randn(8, 100).to(dtype)))


def test_reduced_precision_does_not_hide_an_invalid_simplex() -> None:
    output = MulticlassOutput(classes=100, scores=True)
    score = torch.zeros(1, 100, dtype=torch.float16)
    score[0, :2] = 0.51

    assert _violation(output.validator, {"prediction": torch.tensor([0]), "score": score}) == (
        "simplex"
    )


def test_violations_pickle_and_dtype_is_its_own_kind() -> None:
    import copy
    import pickle

    error = PredictionViolation("sign", "predictions must be non-negative")
    for clone in (copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert (clone.kind, str(clone)) == (error.kind, str(error))
    validate = RegressionOutput(shape=[1]).validator
    assert _violation(validate, {"prediction": torch.ones(2, 1, dtype=torch.int64)}) == "dtype"


def test_score_field_renames_and_validators_check_their_parameters() -> None:
    output = MulticlassOutput(classes=3, scores=True, score_field="probability")
    result = output(torch.randn(2, 3))
    assert set(result) == {"prediction", "probability"}
    output.validator(result)
    with pytest.raises(ValueError, match="threshold"):
        BinaryOutput(threshold=1.0).validator.__class__(threshold=2.0)
    from dsio.experimental.inference import (
        BinaryValidator,
        MulticlassValidator,
        RegressionValidator,
    )

    with pytest.raises(ValueError, match="classes"):
        MulticlassValidator(classes=1)
    with pytest.raises(ValueError, match="shape"):
        RegressionValidator(shape=[0])
    with pytest.raises(ValueError, match="threshold"):
        BinaryValidator(threshold=2.0)
