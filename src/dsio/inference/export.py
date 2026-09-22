"""Log complete predictors through MLflow's native model flavors."""

from __future__ import annotations

import copy
import io
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import mlflow
import numpy as np
import torch
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.tracking import MlflowClient
from torch import Tensor

from dsio.inference.predictor import Predictor, PredictorError, _preserve_rng

type ExportForm = Literal["pytorch", "pyfunc"]

_FORMS = frozenset({"pytorch", "pyfunc"})


class ExportError(ValueError):
    """A requested MLflow representation cannot preserve the predictor contract."""


class _PredictorPyFunc(PythonModel):
    _skip_type_hint_validation = True

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def predict(
        self,
        context: PythonModelContext | None,
        model_input: Any,
        params: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray[Any, Any]]:
        del params
        if not isinstance(model_input, Mapping):
            raise PredictorError(
                f"PyFunc input must be a mapping, got {type(model_input).__name__}"
            )
        identities = model_input.get("sample_id")
        if isinstance(identities, np.ndarray):
            identities = identities.tolist()
        if isinstance(identities, str) or not isinstance(identities, Sequence):
            raise PredictorError("PyFunc input requires sample_id as a sequence")
        values = model_input.get("x")
        if not isinstance(values, np.ndarray | Tensor):
            raise PredictorError("PyFunc input requires x as a numpy array or tensor")
        result = self.predictor(
            {
                "sample_id": list(identities),
                "x": torch.as_tensor(values),
            }
        )
        return _arrays(result, "PyFunc output")


def log_predictor(
    predictor: Predictor,
    *,
    run_id: str,
    input_example: Mapping[str, Any],
    forms: Sequence[ExportForm],
    name: str = "predictor",
) -> dict[ExportForm, ModelInfo]:
    """Log exactly the declared representations against one existing MLflow Run."""
    declared = _forms(forms)
    if not isinstance(predictor, Predictor):
        raise ExportError("predictor export requires a dsio.inference.Predictor")
    if not isinstance(name, str) or not name:
        raise ExportError("predictor model name must be a non-empty string")
    run = _running_run(run_id)

    probe = copy.deepcopy(predictor)
    with _preserve_rng():
        output = probe(input_example)
    example = _input_arrays(input_example)
    expected = _arrays(output, "predictor output")
    try:
        signature = infer_signature(example, expected)
    except Exception as error:
        raise ExportError(f"export form signature is incompatible: {error}") from error
    _preflight(predictor, example, declared)

    metadata = {
        "dsio.checkpoint_uri": predictor.checkpoint_uri,
        "dsio.checkpoint_digest": predictor.checkpoint_digest,
    }
    infos: dict[ExportForm, ModelInfo] = {}
    for form in declared:
        tags = {
            "dsio.export_form": form,
            "dsio.checkpoint_uri": predictor.checkpoint_uri,
            "dsio.checkpoint_digest": predictor.checkpoint_digest,
        }
        logged = mlflow.initialize_logged_model(
            name=f"{name}-{form}",
            source_run_id=run_id,
            experiment_id=run.info.experiment_id,
            model_type="dsio.predictor",
            tags=tags,
        )
        if form == "pytorch":
            infos[form] = mlflow.pytorch.log_model(
                copy.deepcopy(predictor),
                model_id=logged.model_id,
                signature=signature,
                input_example=example,
                metadata={**metadata, "dsio.export_form": form},
                serialization_format="pickle",
            )
        else:
            infos[form] = mlflow.pyfunc.log_model(
                model_id=logged.model_id,
                python_model=_PredictorPyFunc(copy.deepcopy(predictor)),
                signature=signature,
                input_example=example,
                metadata={**metadata, "dsio.export_form": form},
            )
    return infos


def _forms(forms: Sequence[ExportForm]) -> tuple[ExportForm, ...]:
    if isinstance(forms, str) or not isinstance(forms, Sequence) or not forms:
        raise ExportError("at least one export form must be declared")
    declared = tuple(forms)
    unknown = [form for form in declared if form not in _FORMS]
    if unknown:
        raise ExportError(f"unsupported export form {unknown[0]!r}")
    if len(set(declared)) != len(declared):
        raise ExportError("each export form may be declared only once")
    return declared


def _running_run(run_id: str) -> Run:
    if not isinstance(run_id, str) or not run_id:
        raise ExportError("a non-empty MLflow Run ID is required for predictor export")
    try:
        run = MlflowClient().get_run(run_id)
    except (MlflowException, OSError) as error:
        raise ExportError(f"MLflow Run {run_id!r} cannot be loaded: {error}") from error
    if run.info.lifecycle_stage != "active" or run.info.status != "RUNNING":
        raise ExportError(
            f"MLflow Run {run_id!r} must be active and RUNNING for predictor export"
        )
    return run


def _input_arrays(example: Mapping[str, Any]) -> dict[str, np.ndarray[Any, Any]]:
    identities = example.get("sample_id")
    values = example.get("x")
    if isinstance(identities, str) or not isinstance(identities, Sequence):
        raise ExportError("input example requires sample_id as a sequence")
    if not isinstance(values, Tensor):
        raise ExportError("input example requires x as a tensor")
    return {
        "sample_id": np.asarray(list(identities), dtype=np.str_),
        "x": values.detach().cpu().numpy(),
    }


def _arrays(
    values: Mapping[str, Any], role: str
) -> dict[str, np.ndarray[Any, Any]]:
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for field, value in values.items():
        if isinstance(value, Tensor):
            array = value.detach().cpu().numpy()
        else:
            try:
                array = np.asarray(value)
            except (TypeError, ValueError) as error:
                raise ExportError(f"{role} field {field!r} is not array-compatible") from error
        if array.dtype == np.dtype("O"):
            raise ExportError(f"{role} field {field!r} has unsupported object values")
        arrays[field] = array
    return arrays


def _preflight(
    predictor: Predictor,
    example: dict[str, np.ndarray[Any, Any]],
    forms: tuple[ExportForm, ...],
) -> None:
    try:
        if "pytorch" in forms:
            torch.save(predictor, io.BytesIO())
        if "pyfunc" in forms:
            if any(value.device.type != "cpu" for value in predictor.parameters()):
                raise ExportError("pyfunc export form requires predictor parameters on CPU")
            if any(value.device.type != "cpu" for value in predictor.buffers()):
                raise ExportError("pyfunc export form requires predictor buffers on CPU")
            wrapper = _PredictorPyFunc(copy.deepcopy(predictor))
            with _preserve_rng():
                wrapper.predict(None, example)
    except ExportError:
        raise
    except Exception as error:
        requested = "pyfunc" if "pyfunc" in forms else "pytorch"
        raise ExportError(f"{requested} export form is incompatible: {error}") from error
