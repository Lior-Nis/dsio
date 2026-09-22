"""Immutable model inference is one strict MLflow PyFunc boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlflow
import numpy as np
import pytest
import torch
from mlflow.tracking import MlflowClient
from mlflow.types.schema import Schema, TensorSpec
from torch import Tensor, nn

from dsio.inference import (
    InferenceError,
    Predictor,
    PredictorError,
    TensorOutput,
    log_predictor,
    predict,
)


class AddOne(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value + 1


def require_finite_nonnegative_prediction(output: Mapping[str, Any]) -> None:
    prediction = output.get("prediction")
    if not isinstance(prediction, Tensor):
        raise PredictorError("prediction must be a tensor")
    if not bool(torch.isfinite(prediction).all()):
        raise PredictorError("prediction must be finite")
    if bool((prediction < 0).any()):
        raise PredictorError("prediction must be non-negative")


def _predictor() -> Predictor:
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 3.0]]))
        model.bias.fill_(1.0)
    return Predictor(
        model=model,
        preprocessor=AddOne(),
        normalizer=TensorOutput(),
        validator=require_finite_nonnegative_prediction,
        checkpoint_uri="runs:/training/checkpoint",
        checkpoint_digest="a" * 64,
    )


def _native_example() -> dict[str, Any]:
    return {
        "sample_id": ["second", "first"],
        "x": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }


def _inputs() -> dict[str, np.ndarray[Any, Any]]:
    return {
        "sample_id": np.asarray(["second", "first"]),
        "x": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }


def _logged_models() -> dict[str, Any]:
    client = MlflowClient()
    experiment_id = client.create_experiment("immutable-inference")
    run_id = client.create_run(experiment_id).info.run_id
    return log_predictor(
        _predictor(),
        run_id=run_id,
        input_example=_native_example(),
        forms=("pytorch", "pyfunc"),
    )


def test_predict_loads_packaged_behavior_and_returns_native_data() -> None:
    model_uri = _logged_models()["pyfunc"].model_uri

    result = predict(model_uri, _inputs())

    assert isinstance(result, dict)
    assert result["sample_id"].tolist() == ["second", "first"]
    np.testing.assert_array_equal(result["prediction"], [[14.0], [24.0]])


@pytest.mark.parametrize(
    "model_uri",
    [
        "runs:/abc/predictor",
        "models:/classifier/1",
        "models:/classifier@champion",
        "/tmp/predictor",
        "https://example.test/predictor",
    ],
)
def test_predict_rejects_mutable_or_non_model_references(model_uri: str) -> None:
    with pytest.raises(InferenceError, match="immutable MLflow Logged Model URI"):
        predict(model_uri, _inputs())


def test_predict_rejects_missing_logged_model() -> None:
    with pytest.raises(InferenceError, match="cannot be resolved"):
        predict(f"models:/m-{'0' * 32}", _inputs())


@pytest.mark.parametrize("inputs", [None, [1, 2]])
def test_predict_rejects_non_mapping_inputs(inputs: Any) -> None:
    model_uri = _logged_models()["pyfunc"].model_uri

    with pytest.raises(InferenceError, match="inputs must be a mapping"):
        predict(model_uri, inputs)


def test_predict_rejects_native_form_and_unsupported_device() -> None:
    models = _logged_models()

    with pytest.raises(InferenceError, match="pyfunc export form"):
        predict(models["pytorch"].model_uri, _inputs())
    with pytest.raises(InferenceError, match="device.*cpu"):
        predict(models["pyfunc"].model_uri, _inputs(), device="cuda")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"x": np.ones((2, 2), dtype=np.float64)}, "input field 'x'.*dtype"),
        ({"x": np.ones((2, 3), dtype=np.float32)}, "input field 'x'.*shape"),
        ({"x": [[1.0, 2.0], [3.0, 4.0]]}, "input field 'x'.*NumPy array"),
        ({"extra": np.asarray([1, 2])}, "input fields"),
    ],
)
def test_predict_rejects_input_without_schema_coercion(
    change: dict[str, Any], message: str
) -> None:
    model_uri = _logged_models()["pyfunc"].model_uri
    inputs: dict[str, Any] = _inputs()
    inputs.update(change)

    with pytest.raises(InferenceError, match=message):
        predict(model_uri, inputs)


def test_predict_preserves_packaged_semantic_validation() -> None:
    model_uri = _logged_models()["pyfunc"].model_uri
    inputs = {
        "sample_id": np.asarray(["negative"]),
        "x": np.asarray([[-2.0, -2.0]], dtype=np.float32),
    }

    with pytest.raises(InferenceError, match="logged predictor inference failed") as caught:
        predict(model_uri, inputs)

    assert isinstance(caught.value.__cause__, PredictorError)


def test_predict_translates_deserialization_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_uri = _logged_models()["pyfunc"].model_uri

    def missing_dependency(_: str) -> None:
        raise ModuleNotFoundError("consumer dependency is unavailable")

    monkeypatch.setattr(mlflow.pyfunc, "load_model", missing_dependency)

    with pytest.raises(InferenceError, match="cannot be loaded") as caught:
        predict(model_uri, _inputs())

    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


def test_predict_revalidates_model_evidence_after_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_uri = _logged_models()["pyfunc"].model_uri
    model_id = model_uri.removeprefix("models:/")
    real_load = mlflow.pyfunc.load_model

    def load_then_retag(uri: str) -> Any:
        loaded = real_load(uri)
        MlflowClient().set_logged_model_tags(
            model_id, {"dsio.export_form": "pytorch"}
        )
        return loaded

    monkeypatch.setattr(mlflow.pyfunc, "load_model", load_then_retag)

    with pytest.raises(InferenceError, match="pyfunc export form"):
        predict(model_uri, _inputs())


def test_predict_rejects_duplicate_signature_fields() -> None:
    import dsio.inference.loading as loading

    schema = Schema(
        [
            TensorSpec(np.dtype(str), (-1,), name="sample_id"),
            TensorSpec(np.dtype(str), (-1,), name="sample_id"),
        ]
    )

    with pytest.raises(InferenceError, match="duplicate field names"):
        loading._tensor_specs(schema, "input")


@pytest.mark.parametrize("name", [1, b"prediction", ""])
def test_predict_rejects_invalid_signature_field_names(name: Any) -> None:
    import dsio.inference.loading as loading

    schema = Schema(
        [
            TensorSpec(np.dtype(str), (-1,), name="sample_id"),
            TensorSpec(np.dtype(np.float32), (-1,), name=name),
        ]
    )

    with pytest.raises(InferenceError, match="non-empty named tensor fields"):
        loading._tensor_specs(schema, "output")


@pytest.mark.parametrize(
    "identity",
    [
        TensorSpec(np.dtype(np.int64), (-1,), name="sample_id"),
        TensorSpec(np.dtype(str), (-1, 1), name="sample_id"),
    ],
)
def test_predict_rejects_invalid_identity_signature(identity: TensorSpec) -> None:
    import dsio.inference.loading as loading

    with pytest.raises(InferenceError, match="one-dimensional string tensor"):
        loading._tensor_specs(Schema([identity]), "input")


def test_predict_rejects_output_identity_or_signature_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_uri = _logged_models()["pyfunc"].model_uri
    loaded = mlflow.pyfunc.load_model(model_uri)
    real_predict = loaded.predict

    def reordered(values: Any) -> dict[str, np.ndarray[Any, Any]]:
        result = real_predict(values)
        return {**result, "sample_id": result["sample_id"][::-1]}

    monkeypatch.setattr(loaded, "predict", reordered)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda _: loaded)

    with pytest.raises(InferenceError, match="sample_id order"):
        predict(model_uri, _inputs())
