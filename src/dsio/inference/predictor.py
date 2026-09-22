"""A deterministic inference module built from immutable training evidence."""

from __future__ import annotations

import copy
import io
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from torch import Tensor, nn

from dsio.config.components import ComponentError, require_importable_component
from dsio.model.module import DsioModule
from dsio.train.artifacts import ArtifactIntegrityError, ArtifactRef, load_artifact
from dsio.train.tracking import resolve_tracking_uri

type Prediction = dict[str, Any]
type Validator = Callable[[Mapping[str, Any]], None]


class PredictorError(ValueError):
    """Predictor evidence, components, input, or output violated its contract."""


class TensorOutput(nn.Module):
    """Name one native tensor model output without changing it."""

    def __init__(self, field: str = "prediction") -> None:
        super().__init__()
        if not isinstance(field, str) or not field or field == "sample_id":
            raise PredictorError("tensor output field must be non-empty and not 'sample_id'")
        self.field = field

    def forward(self, value: Any) -> Mapping[str, Tensor]:
        if not isinstance(value, Tensor):
            raise PredictorError(
                f"tensor output normalizer expected a tensor, got {type(value).__name__}"
            )
        return {self.field: value}


def validate_tensor_prediction(output: Mapping[str, Any]) -> None:
    """Require the standard prediction field to be a finite tensor."""
    prediction = output.get("prediction")
    if not isinstance(prediction, Tensor):
        raise PredictorError("prediction must be a tensor")
    if not bool(torch.isfinite(prediction).all()):
        raise PredictorError("prediction tensor must contain only finite values")


class Predictor(nn.Module):
    """Deterministic preprocessing, model, normalization, and semantic validation."""

    def __init__(
        self,
        *,
        model: nn.Module,
        preprocessor: nn.Module,
        normalizer: nn.Module,
        validator: Validator,
        checkpoint_uri: str,
        checkpoint_digest: str,
    ) -> None:
        super().__init__()
        if isinstance(model, DsioModule):
            raise PredictorError(
                "a DsioModule is a training system; package its deterministic model only"
            )
        for role, component in (
            ("model", model),
            ("preprocessor", preprocessor),
            ("normalizer", normalizer),
        ):
            if not isinstance(component, nn.Module):
                raise PredictorError(f"{role} must be a torch nn.Module")
        if not callable(validator):
            raise PredictorError("validator must be callable")
        _require_importable_module(model, "model")
        _require_importable_module(preprocessor, "preprocessor")
        _require_importable_module(normalizer, "normalizer")
        _require_importable(validator, "validator")
        self.model = model
        self.preprocessor = preprocessor
        self.normalizer = normalizer
        self.validator = validator
        self.checkpoint_uri = checkpoint_uri
        self.checkpoint_digest = checkpoint_digest
        self.eval()

    def train(self, mode: bool = True) -> Predictor:
        if mode:
            raise PredictorError("Predictor is inference-only and cannot enter training mode")
        return super().train(False)

    @torch.inference_mode()
    def forward(self, batch: Mapping[str, Any]) -> Prediction:
        sample_ids, x = _inputs(batch)
        try:
            prepared = self.preprocessor(x.clone())
        except Exception as error:
            raise PredictorError(
                f"preprocessor {_name(self.preprocessor)} failed: {error}"
            ) from error
        try:
            raw = self.model(prepared)
        except Exception as error:
            raise PredictorError(f"model {_name(self.model)} failed: {error}") from error
        try:
            normalized = self.normalizer(raw)
        except Exception as error:
            raise PredictorError(f"normalizer {_name(self.normalizer)} failed: {error}") from error
        result = _prediction(sample_ids, normalized)
        try:
            returned = self.validator(copy.deepcopy(result))
        except Exception as error:
            raise PredictorError(f"validator {_name(self.validator)} failed: {error}") from error
        if returned is not None:
            raise PredictorError("validator must return None after validating predictions")
        return result


def build_predictor(
    checkpoint: ArtifactRef,
    *,
    model: nn.Module,
    preprocessor: nn.Module | None,
    normalizer: nn.Module,
    validator: Validator,
    input_example: Mapping[str, Any],
    tracking_uri: str | None = None,
) -> Predictor:
    """Build a serializable inference module from a successful Lightning checkpoint."""
    if isinstance(model, DsioModule):
        raise PredictorError(
            "a DsioModule contains training-only behavior; pass its deterministic model only"
        )
    prepared = preprocessor if preprocessor is not None else nn.Identity()
    _require_importable_module(model, "model")
    _require_importable_module(prepared, "preprocessor")
    _require_importable_module(normalizer, "normalizer")
    _require_importable(validator, "validator")
    uri = resolve_tracking_uri(tracking_uri)
    _require_successful_run(checkpoint, uri)
    try:
        payload = load_artifact(checkpoint, tracking_uri=uri)
    except ArtifactIntegrityError as error:
        raise PredictorError(f"checkpoint evidence is invalid: {error}") from error
    state = _model_state(payload)
    cloned_model = _clone_component(model, "model")
    cloned_preprocessor = _clone_component(prepared, "preprocessor")
    cloned_normalizer = _clone_component(normalizer, "normalizer")
    cloned_validator = _clone_component(validator, "validator")
    try:
        cloned_model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise PredictorError(f"checkpoint model state is incompatible: {error}") from error
    predictor = Predictor(
        model=cloned_model,
        preprocessor=cloned_preprocessor,
        normalizer=cloned_normalizer,
        validator=cloned_validator,
        checkpoint_uri=checkpoint.uri,
        checkpoint_digest=checkpoint.digest,
    )
    try:
        probe = copy.deepcopy(predictor)
        with _preserve_rng():
            probe(input_example)
        torch.save(probe, io.BytesIO())
        torch.save(predictor, io.BytesIO())
    except PredictorError as error:
        raise PredictorError(f"predictor components are incompatible: {error}") from error
    except Exception as error:
        raise PredictorError(f"predictor cannot be isolated or serialized: {error}") from error
    _require_successful_run(checkpoint, uri)
    return predictor


def _model_state(payload: bytes) -> dict[str, Any]:
    try:
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise PredictorError(f"checkpoint cannot be loaded safely: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise PredictorError("Lightning checkpoint must contain a mapping")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise PredictorError("Lightning checkpoint is missing state_dict")
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("model.")
    }
    if not model_state:
        raise PredictorError("Lightning checkpoint contains no model state")
    return model_state


def _require_successful_run(checkpoint: ArtifactRef, tracking_uri: str) -> None:
    try:
        run = MlflowClient(tracking_uri).get_run(checkpoint.run_id)
    except (MlflowException, OSError) as error:
        raise PredictorError(
            f"checkpoint source Run {checkpoint.run_id!r} cannot be loaded: {error}"
        ) from error
    if run.info.lifecycle_stage != "active":
        raise PredictorError(
            f"checkpoint source Run is {run.info.lifecycle_stage}; predictor evidence "
            "requires an active Run"
        )
    if run.info.status != "FINISHED":
        raise PredictorError(
            f"checkpoint source Run is {run.info.status}; predictor evidence requires FINISHED"
        )


def _inputs(batch: object) -> tuple[list[str], Tensor]:
    if not isinstance(batch, Mapping):
        raise PredictorError(f"predictor input must be a mapping, got {type(batch).__name__}")
    identities = batch.get("sample_id")
    if isinstance(identities, str) or not isinstance(identities, Sequence):
        raise PredictorError("predictor input requires sample_id as a sequence of strings")
    sample_ids = list(identities)
    if not sample_ids or any(not isinstance(value, str) or not value for value in sample_ids):
        raise PredictorError("sample_id must be a non-empty sequence of non-empty strings")
    x = batch.get("x")
    if not isinstance(x, Tensor):
        raise PredictorError("predictor input requires x as a tensor")
    if x.ndim == 0 or x.shape[0] != len(sample_ids):
        size = None if x.ndim == 0 else x.shape[0]
        raise PredictorError(
            f"predictor input has {len(sample_ids)} sample_id values but x has {size} rows"
        )
    return sample_ids, x


def _prediction(sample_ids: list[str], normalized: object) -> Prediction:
    if not isinstance(normalized, Mapping):
        raise PredictorError(
            f"normalizer must return a mapping, got {type(normalized).__name__}"
        )
    if "sample_id" in normalized:
        raise PredictorError("normalizer cannot replace reserved sample_id identity")
    if not normalized:
        raise PredictorError("normalizer must declare at least one prediction field")
    result: Prediction = {"sample_id": list(sample_ids)}
    for name, value in normalized.items():
        if not isinstance(name, str) or not name:
            raise PredictorError("prediction field names must be non-empty strings")
        if isinstance(value, Tensor):
            size = None if value.ndim == 0 else value.shape[0]
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            size = len(value)
        else:
            raise PredictorError(
                f"prediction field {name!r} must be a tensor or sequence, "
                f"got {type(value).__name__}"
            )
        if size != len(sample_ids):
            raise PredictorError(
                f"prediction field {name!r} has {size} rows for "
                f"{len(sample_ids)} sample_id values"
            )
        result[name] = value
    return result


def _require_importable_module(module: nn.Module, role: str) -> None:
    for path, component in module.named_modules():
        child_role = role if not path else f"{role}.{path}"
        _require_importable(component, child_role)


def _require_importable(value: object, role: str) -> None:
    try:
        require_importable_component(value, role)
    except ComponentError as error:
        raise PredictorError(f"{role} must be importable: {error}") from None


def _clone_component[T](value: T, role: str) -> T:
    try:
        cloned = copy.deepcopy(value)
        torch.save(cloned, io.BytesIO())
    except Exception as error:
        raise PredictorError(f"{role} cannot be isolated or serialized: {error}") from error
    return cloned


@contextmanager
def _preserve_rng() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _name(value: object) -> str:
    return str(getattr(value, "__qualname__", type(value).__qualname__))


__all__ = [
    "Predictor",
    "PredictorError",
    "TensorOutput",
    "build_predictor",
    "validate_tensor_prediction",
]
