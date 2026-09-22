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
