"""Load one immutable MLflow predictor and run one inference call."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import mlflow
import numpy as np
from mlflow.entities import LoggedModelStatus
from mlflow.exceptions import MlflowException
from mlflow.models import validate_schema
from mlflow.pyfunc import PyFuncModel
from mlflow.types.schema import Schema, TensorSpec

type Prediction = dict[str, np.ndarray[Any, Any]]

_LOGGED_MODEL_URI = re.compile(r"models:/(m-[0-9a-f]{32})")


class InferenceError(ValueError):
    """Immutable model evidence or one prediction violated its declared contract."""


def predict(
    model_uri: str,
    inputs: Mapping[str, Any],
    *,
    device: str = "cpu",
) -> Prediction:
    """Run a signed DSIO PyFunc predictor without owning its execution lifecycle."""
    if device != "cpu":
        raise InferenceError(
            f"predictor device must be 'cpu'; requested device {device!r} is unsupported"
        )
    if not isinstance(inputs, Mapping):
        raise InferenceError(
            f"predictor inputs must be a mapping, got {type(inputs).__name__}"
        )
    loaded, model_id = _load_predictor(model_uri)
    input_schema = loaded.metadata.get_input_schema()
    output_schema = loaded.metadata.get_output_schema()
    _validate_arrays(inputs, input_schema, "input")
    try:
        result = loaded.predict(dict(inputs))
    except Exception as error:
        raise InferenceError(f"logged predictor inference failed: {error}") from error
    if not isinstance(result, Mapping):
        raise InferenceError(
            f"logged predictor output must be a mapping, got {type(result).__name__}"
        )
    _validate_arrays(result, output_schema, "output")
    _require_identity(inputs, result)
    _require_logged_predictor(model_id, model_uri)
    return dict(result)


def _load_predictor(model_uri: str) -> tuple[PyFuncModel, str]:
    if not isinstance(model_uri, str):
        raise InferenceError("model_uri must be an immutable MLflow Logged Model URI")
    matched = _LOGGED_MODEL_URI.fullmatch(model_uri)
    if matched is None:
        raise InferenceError(
            "model_uri must be an immutable MLflow Logged Model URI "
            "in the form 'models:/m-<id>'"
        )
    model_id = matched.group(1)
    _require_logged_predictor(model_id, model_uri)
    try:
        loaded = mlflow.pyfunc.load_model(model_uri)
    except Exception as error:
        raise InferenceError(
            f"immutable Logged Model {model_id!r} cannot be loaded: {error}"
        ) from error
    _require_logged_predictor(model_id, model_uri)
    return loaded, model_id


def _require_logged_predictor(model_id: str, model_uri: str) -> None:
    try:
        logged = mlflow.get_logged_model(model_id)
    except Exception as error:
        raise InferenceError(
            f"immutable Logged Model {model_id!r} cannot be resolved: {error}"
        ) from error
    if logged.model_uri != model_uri or logged.status != LoggedModelStatus.READY:
        raise InferenceError(
            f"immutable Logged Model {model_id!r} is not ready for inference"
        )
    if (
        logged.model_type != "dsio.predictor"
        or logged.tags.get("dsio.export_form") != "pyfunc"
    ):
        raise InferenceError(
            f"immutable Logged Model {model_id!r} must be a DSIO pyfunc export form"
        )


def _validate_arrays(values: Mapping[str, Any], schema: Schema | None, role: str) -> None:
    specs = _tensor_specs(schema, role)
    expected: set[str] = set()
    for spec in specs:
        assert spec.name is not None
        expected.add(spec.name)
    if set(values) != expected:
        raise InferenceError(
            f"predictor {role} fields must be {sorted(expected)!r}; "
            f"received {sorted(map(str, values))!r}"
        )
    for spec in specs:
        name = spec.name
        assert name is not None
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise InferenceError(
                f"predictor {role} field {name!r} must be a NumPy array; "
                f"got {type(value).__name__}"
            )
        if not _same_dtype(value.dtype, spec.type):
            raise InferenceError(
                f"predictor {role} field {name!r} has dtype {value.dtype}; "
                f"expected {spec.type}"
            )
        if not _same_shape(value.shape, spec.shape):
            raise InferenceError(
                f"predictor {role} field {name!r} has shape {value.shape}; "
                f"expected {spec.shape}"
            )
    try:
        assert schema is not None
        validate_schema(dict(values), schema)
    except (MlflowException, TypeError, ValueError) as error:
        raise InferenceError(f"predictor {role} violates its MLflow Signature: {error}") from error


def _tensor_specs(schema: Schema | None, role: str) -> list[TensorSpec]:
    if schema is None or not schema.inputs:
        raise InferenceError(f"logged predictor is missing its {role} signature")
    specs: list[TensorSpec] = []
    field_names: list[str] = []
    for spec in schema.inputs:
        if (
            not isinstance(spec, TensorSpec)
            or not isinstance(spec.name, str)
            or not spec.name
        ):
            raise InferenceError(
                f"logged predictor {role} signature must contain non-empty named tensor fields"
            )
        specs.append(spec)
        field_names.append(spec.name)
    if len(set(field_names)) != len(field_names):
        raise InferenceError(
            f"logged predictor {role} signature contains duplicate field names"
        )
    names = set(field_names)
    if "sample_id" not in names:
        raise InferenceError(
            f"logged predictor {role} signature must declare sample_id identity"
        )
    identity = next(spec for spec in specs if spec.name == "sample_id")
    if identity.type.kind != "U" or len(identity.shape) != 1:
        raise InferenceError(
            f"logged predictor {role} sample_id must be a one-dimensional string tensor"
        )
    return specs


def _same_dtype(actual: np.dtype[Any], expected: np.dtype[Any]) -> bool:
    if expected.kind in {"U", "S"}:
        return actual.kind == expected.kind
    return actual == expected


def _same_shape(actual: tuple[int, ...], expected: tuple[int, ...]) -> bool:
    return len(actual) == len(expected) and all(
        declared == -1 or declared == observed
        for observed, declared in zip(actual, expected, strict=True)
    )


def _require_identity(inputs: Mapping[str, Any], output: Mapping[str, Any]) -> None:
    if not np.array_equal(output["sample_id"], inputs["sample_id"]):
        raise InferenceError(
            "logged predictor output sample_id order does not match its input identities"
        )
