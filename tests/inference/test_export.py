"""Predictor export uses MLflow's model contract without a parallel DSIO registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlflow
import numpy as np
import pytest
import torch
from mlflow.tracking import MlflowClient
from torch import Tensor, nn

from dsio.inference import (
    Predictor,
    PredictorError,
    TensorOutput,
    log_predictor,
)


class AddOne(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value + 1


class MetaBufferPreprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("unused", torch.empty(0, device="meta"))

    def forward(self, value: Tensor) -> Tensor:
        return value


class MixedSequenceOutput(nn.Module):
    def forward(self, value: Tensor) -> Mapping[str, list[object]]:
        del value
        return {"prediction": [1, "2"]}


class NumpySequenceOutput(nn.Module):
    def forward(self, value: Tensor) -> Mapping[str, list[np.float32]]:
        del value
        return {"prediction": [np.float32(1), np.float32(2)]}


class DeepcopyOnlyState:
    def __deepcopy__(self, memo: dict[int, Any]) -> DeepcopyOnlyState:
        del memo
        return DeepcopyOnlyState()

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError("cannot pickle deepcopy-only state")


class LoadFailingState:
    def __deepcopy__(self, memo: dict[int, Any]) -> LoadFailingState:
        del memo
        return self

    def __getstate__(self) -> dict[str, Any]:
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        del state
        raise RuntimeError("cannot restore serialized state")


def require_finite_nonnegative_prediction(output: Mapping[str, Any]) -> None:
    prediction = output.get("prediction")
    if not isinstance(prediction, Tensor):
        raise PredictorError("prediction must be a tensor")
    if not bool(torch.isfinite(prediction).all()):
        raise PredictorError("prediction must be finite")
    if bool((prediction < 0).any()):
        raise PredictorError("prediction must be non-negative")


def accept_prediction(output: Mapping[str, Any]) -> None:
    del output


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


def _run() -> tuple[str, str]:
    client = MlflowClient()
    experiment_id = client.create_experiment("predictor-export")
    run_id = client.create_run(experiment_id).info.run_id
    return experiment_id, run_id


def _example() -> dict[str, Any]:
    return {
        "sample_id": ["second", "first"],
        "x": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }


def _identity_predictor() -> Predictor:
    return Predictor(
        model=nn.Identity(),
        preprocessor=nn.Identity(),
        normalizer=TensorOutput(),
        validator=require_finite_nonnegative_prediction,
        checkpoint_uri="runs:/training/checkpoint",
        checkpoint_digest="b" * 64,
    )


def test_declared_forms_log_native_signed_models_with_equivalent_predictions() -> None:
    experiment_id, run_id = _run()

    models = log_predictor(
        _predictor(),
        run_id=run_id,
        input_example=_example(),
        forms=("pytorch", "pyfunc"),
        name="classifier",
    )

    assert set(models) == {"pytorch", "pyfunc"}
    for form, info in models.items():
        assert info.model_uri.startswith("models:/m-")
        assert info.model_id is not None
        assert info.artifact_path
        assert info.signature is not None
        assert info.saved_input_example_info is not None
        logged = mlflow.get_logged_model(info.model_id)
        assert logged.source_run_id == run_id
        assert logged.experiment_id == experiment_id
        assert logged.tags["dsio.export_form"] == form
        assert logged.tags["dsio.checkpoint_digest"] == "a" * 64

    native = mlflow.pytorch.load_model(models["pytorch"].model_uri)
    native_result = native(_example())
    assert native_result["sample_id"] == ["second", "first"]
    torch.testing.assert_close(native_result["prediction"], torch.tensor([[14.0], [24.0]]))

    pyfunc = mlflow.pyfunc.load_model(models["pyfunc"].model_uri)
    pyfunc_result = pyfunc.predict(
        {
            "sample_id": np.asarray(["second", "first"]),
            "x": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        }
    )
    assert pyfunc_result["sample_id"].tolist() == ["second", "first"]
    np.testing.assert_allclose(pyfunc_result["prediction"], [[14.0], [24.0]])

    client = MlflowClient()
    assert client.get_run(run_id).info.status == "RUNNING"
    assert mlflow.active_run() is None
    assert client.search_registered_models() == []


def test_semantic_validator_remains_inside_both_logged_forms() -> None:
    _, run_id = _run()
    models = log_predictor(
        _predictor(),
        run_id=run_id,
        input_example=_example(),
        forms=("pytorch", "pyfunc"),
    )
    invalid_native = {
        "sample_id": ["negative"],
        "x": torch.tensor([[-2.0, -2.0]]),
    }
    invalid_pyfunc = {
        "sample_id": np.asarray(["negative"]),
        "x": np.asarray([[-2.0, -2.0]], dtype=np.float32),
    }

    with pytest.raises(PredictorError, match="non-negative"):
        mlflow.pytorch.load_model(models["pytorch"].model_uri)(invalid_native)
    with pytest.raises(PredictorError, match="non-negative"):
        mlflow.pyfunc.load_model(models["pyfunc"].model_uri).predict(invalid_pyfunc)


def test_undeclared_form_produces_no_model() -> None:
    experiment_id, run_id = _run()

    models = log_predictor(
        _predictor(),
        run_id=run_id,
        input_example=_example(),
        forms=("pyfunc",),
    )

    assert set(models) == {"pyfunc"}
    logged = mlflow.search_logged_models(
        experiment_ids=[experiment_id], output_format="list"
    )
    assert len(logged) == 1
    assert logged[0].tags["dsio.export_form"] == "pyfunc"


@pytest.mark.parametrize("forms", [(), ("pyfunc", "pyfunc"), ("pt2",), ([],)])
def test_invalid_form_declaration_fails_before_logging(
    forms: tuple[Any, ...],
) -> None:
    experiment_id, run_id = _run()

    with pytest.raises(ValueError, match="export form"):
        log_predictor(
            _predictor(),
            run_id=run_id,
            input_example=_example(),
            forms=forms,
        )

    assert (
        mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list") == []
    )


def test_invalid_fixture_fails_before_logging() -> None:
    experiment_id, run_id = _run()

    with pytest.raises(PredictorError, match="non-negative"):
        log_predictor(
            _predictor(),
            run_id=run_id,
            input_example={
                "sample_id": ["negative"],
                "x": torch.tensor([[-2.0, -2.0]]),
            },
            forms=("pyfunc",),
        )

    assert (
        mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list") == []
    )


def test_incompatible_requested_form_fails_before_logging() -> None:
    experiment_id, run_id = _run()
    predictor = _predictor()
    predictor.preprocessor = MetaBufferPreprocessor()

    with pytest.raises(ValueError, match="pyfunc export form.*CPU"):
        log_predictor(
            predictor,
            run_id=run_id,
            input_example=_example(),
            forms=("pyfunc",),
        )

    assert (
        mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list") == []
    )


def test_export_requires_an_active_running_source_run() -> None:
    experiment_id, run_id = _run()
    MlflowClient().set_terminated(run_id, "FINISHED")

    with pytest.raises(ValueError, match="active and RUNNING"):
        log_predictor(
            _predictor(),
            run_id=run_id,
            input_example=_example(),
            forms=("pyfunc",),
        )

    assert (
        mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list") == []
    )


def test_each_failed_preflight_names_its_actual_form() -> None:
    experiment_id, run_id = _run()
    predictor = _predictor()
    predictor.model.deepcopy_only = DeepcopyOnlyState()

    with pytest.raises(ValueError, match="pytorch export form.*cannot pickle"):
        log_predictor(
            predictor,
            run_id=run_id,
            input_example=_example(),
            forms=("pytorch", "pyfunc"),
        )

    assert (
        mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list") == []
    )


def test_native_preflight_proves_deserialization_and_execution() -> None:
    experiment_id, run_id = _run()
    predictor = _predictor()
    predictor.model.load_failing = LoadFailingState()

    with pytest.raises(ValueError, match="pytorch export form.*restore serialized"):
        log_predictor(
            predictor,
            run_id=run_id,
            input_example=_example(),
            forms=("pytorch",),
        )

    assert (
        mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list") == []
    )


def test_native_preflight_does_not_remap_the_predictor_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dsio.inference.export as export_module

    predictor = _predictor()
    input_example = _example()
    example = export_module._input_arrays(input_example, ("pytorch",))
    expected = export_module._arrays(predictor(input_example), "test output")
    real_load = torch.load
    observed_map_locations: list[Any] = []

    def preserving_load(*args: Any, **kwargs: Any) -> Any:
        observed_map_locations.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(export_module.torch, "load", preserving_load)

    export_module._preflight(
        predictor,
        input_example,
        example,
        expected,
        ("pytorch",),
    )

    assert observed_map_locations == [None]


def test_numpy_incompatible_dtype_names_form_and_field() -> None:
    experiment_id, run_id = _run()

    with pytest.raises(ValueError, match="pyfunc export form.*input.*x.*bfloat16"):
        log_predictor(
            _identity_predictor(),
            run_id=run_id,
            input_example={
                "sample_id": ["bfloat"],
                "x": torch.ones(1, 2, dtype=torch.bfloat16),
            },
            forms=("pyfunc",),
        )

    assert (
        mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list") == []
    )


def test_lossy_sequence_output_fails_before_logging() -> None:
    experiment_id, run_id = _run()
    predictor = Predictor(
        model=nn.Identity(),
        preprocessor=nn.Identity(),
        normalizer=MixedSequenceOutput(),
        validator=accept_prediction,
        checkpoint_uri="runs:/training/checkpoint",
        checkpoint_digest="c" * 64,
    )

    with pytest.raises(
        ValueError, match="pyfunc export form.*prediction.*losslessly"
    ):
        log_predictor(
            predictor,
            run_id=run_id,
            input_example=_example(),
            forms=("pyfunc",),
        )

    assert (
        mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list") == []
    )


def test_homogeneous_numpy_scalar_output_is_lossless() -> None:
    _, run_id = _run()
    predictor = Predictor(
        model=nn.Identity(),
        preprocessor=nn.Identity(),
        normalizer=NumpySequenceOutput(),
        validator=accept_prediction,
        checkpoint_uri="runs:/training/checkpoint",
        checkpoint_digest="d" * 64,
    )

    models = log_predictor(
        predictor,
        run_id=run_id,
        input_example=_example(),
        forms=("pyfunc",),
    )

    result = mlflow.pyfunc.load_model(models["pyfunc"].model_uri).predict(
        {
            "sample_id": np.asarray(["second", "first"]),
            "x": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        }
    )
    np.testing.assert_array_equal(result["prediction"], np.asarray([1, 2], dtype=np.float32))


@pytest.mark.parametrize(
    "values",
    [
        [np.longdouble(np.nan), np.longdouble(1)],
        [np.complex128(complex(np.nan, 0)), np.complex128(1)],
    ],
)
def test_numpy_nan_sequence_conversion_is_lossless(values: list[Any]) -> None:
    import dsio.inference.export as export_module

    array = export_module._arrays({"prediction": values}, "test output")["prediction"]

    assert array.shape == (2,)
    assert bool(np.isnan(array[0]))


def test_source_run_is_refreshed_after_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dsio.inference.export as export_module

    experiment_id, run_id = _run()
    real_preflight = export_module._preflight

    def preflight_then_finish(*args: Any, **kwargs: Any) -> None:
        real_preflight(*args, **kwargs)
        MlflowClient().set_terminated(run_id, "FINISHED")

    monkeypatch.setattr(export_module, "_preflight", preflight_then_finish)

    with pytest.raises(ValueError, match="active and RUNNING"):
        log_predictor(
            _predictor(),
            run_id=run_id,
            input_example=_example(),
            forms=("pyfunc",),
        )

    assert (
        mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list") == []
    )
