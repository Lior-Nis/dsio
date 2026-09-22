"""A predictor is deterministic inference state, not a renamed training checkpoint."""

from __future__ import annotations

import io
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from mlflow.tracking import MlflowClient
from torch import Tensor, nn

from dsio.inference import (
    Predictor,
    PredictorError,
    TensorOutput,
    build_predictor,
    validate_tensor_prediction,
)
from dsio.model.module import DsioModule
from dsio.train.artifacts import ArtifactRef, load_artifact, save_artifact
from dsio.train.tracking import resolve_tracking_uri


class AddOne(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value + 1


class NonSerializablePreprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()

    def forward(self, value: Tensor) -> Tensor:
        return value


class IdentityStealingNormalizer(nn.Module):
    def forward(self, value: Tensor) -> Mapping[str, Any]:
        return {"sample_id": ["fake"], "prediction": value}


class StatefulLinear(nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 1)
        self.label = "original"

    def get_extra_state(self) -> dict[str, str]:
        return {"label": self.label}

    def set_extra_state(self, state: dict[str, str]) -> None:
        self.label = state["label"]


class ThresholdValidator:
    def __init__(self, maximum: float) -> None:
        self.maximum = maximum

    def __call__(self, output: Mapping[str, Any]) -> None:
        prediction = output["prediction"]
        if not isinstance(prediction, Tensor) or prediction.max().item() > self.maximum:
            raise ValueError("prediction exceeds maximum")


def corrupting_validator(output: Mapping[str, Any]) -> None:
    output["sample_id"].clear()


class EmptyObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor]:
        del model, batch, stage
        return {"loss": torch.tensor(0.0)}


def _linear() -> nn.Linear:
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 3.0]]))
        model.bias.fill_(1.0)
    return model


def _checkpoint(
    tmp_path: Path,
    *,
    model: nn.Module | None = None,
    status: str = "FINISHED",
) -> ArtifactRef:
    del tmp_path
    client = MlflowClient(resolve_tracking_uri())
    experiment_id = client.create_experiment("predictor-source")
    run_id = client.create_run(experiment_id).info.run_id
    source = model or _linear()
    payload = io.BytesIO()
    torch.save(
        {
            "state_dict": {f"model.{name}": value for name, value in source.state_dict().items()},
            "optimizer_states": [{"resume-only": True}],
        },
        payload,
    )
    ref = save_artifact(payload.getvalue(), run_id=run_id, name="checkpoint")
    if status != "RUNNING":
        client.set_terminated(run_id, status)
    return ref


def _build(ref: ArtifactRef, **overrides: Any) -> Predictor:
    values: dict[str, Any] = {
        "model": nn.Linear(2, 1),
        "preprocessor": AddOne(),
        "normalizer": TensorOutput(),
        "validator": validate_tensor_prediction,
        "input_example": {"sample_id": ["example"], "x": torch.ones(1, 2)},
    }
    values.update(overrides)
    return build_predictor(ref, **values)


def test_predictor_applies_the_declared_order_and_preserves_identity(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    predictor = _build(ref)
    serialized = io.BytesIO()
    torch.save(predictor, serialized)
    serialized.seek(0)
    predictor = torch.load(serialized, weights_only=False)

    result = predictor(
        {
            "sample_id": ["second", "first"],
            "x": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    assert result["sample_id"] == ["second", "first"]
    torch.testing.assert_close(result["prediction"], torch.tensor([[14.0], [24.0]]))
    assert predictor.checkpoint_uri == ref.uri
    assert predictor.checkpoint_digest == ref.digest
    assert not predictor.training


def test_semantically_invalid_output_fails_inside_the_predictor(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    predictor = _build(ref)
    assert isinstance(predictor.model, nn.Linear)
    with torch.no_grad():
        predictor.model.weight.fill_(float("nan"))

    with pytest.raises(PredictorError, match="prediction.*finite"):
        predictor({"sample_id": ["a"], "x": torch.ones(1, 2)})


def test_normalizer_cannot_replace_source_identity(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="normalizer cannot replace.*sample_id"):
        _build(_checkpoint(tmp_path), normalizer=IdentityStealingNormalizer())


def test_incompatible_components_fail_during_construction(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="normalizer must return a mapping"):
        _build(_checkpoint(tmp_path), normalizer=nn.Identity())


def test_non_importable_validator_fails_during_construction(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="validator.*importable"):
        _build(_checkpoint(tmp_path), validator=lambda output: None)


def test_non_serializable_component_fails_during_construction(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="preprocessor.*serializ"):
        _build(_checkpoint(tmp_path), preprocessor=NonSerializablePreprocessor())


def test_checkpoint_must_belong_to_a_successful_run(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="RUNNING.*FINISHED"):
        _build(_checkpoint(tmp_path, status="RUNNING"))


def test_checkpoint_must_belong_to_an_active_run(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    MlflowClient(resolve_tracking_uri()).delete_run(ref.run_id)

    with pytest.raises(PredictorError, match="deleted.*active"):
        _build(ref)


def test_checkpoint_run_is_rechecked_after_artifact_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = _checkpoint(tmp_path)

    def load_then_delete(
        checkpoint: ArtifactRef, *, tracking_uri: str | None = None
    ) -> bytes:
        payload = load_artifact(checkpoint, tracking_uri=tracking_uri)
        MlflowClient(resolve_tracking_uri(tracking_uri)).delete_run(checkpoint.run_id)
        return payload

    monkeypatch.setattr("dsio.inference.predictor.load_artifact", load_then_delete)

    with pytest.raises(PredictorError, match="deleted.*active"):
        _build(ref)


def test_checkpoint_state_must_match_the_inference_model_strictly(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="checkpoint model state"):
        _build(_checkpoint(tmp_path), model=nn.Linear(3, 1))


def test_checkpoint_preserves_native_pytorch_extra_state(tmp_path: Path) -> None:
    source = StatefulLinear()
    source.label = "from-checkpoint"

    predictor = _build(_checkpoint(tmp_path, model=source), model=StatefulLinear())

    assert isinstance(predictor.model, StatefulLinear)
    assert predictor.model.label == "from-checkpoint"


def test_malformed_model_state_is_not_silently_discarded(tmp_path: Path) -> None:
    payload = io.BytesIO()
    source = _linear()
    state: dict[str, Any] = {
        f"model.{name}": value for name, value in source.state_dict().items()
    }
    state["model.corrupt"] = 1
    torch.save({"state_dict": state}, payload)
    client = MlflowClient(resolve_tracking_uri())
    run_id = client.create_run(client.create_experiment("malformed-state")).info.run_id
    malformed_ref = save_artifact(payload.getvalue(), run_id=run_id, name="checkpoint")
    client.set_terminated(run_id, "FINISHED")

    with pytest.raises(PredictorError, match="Unexpected key.*corrupt"):
        _build(malformed_ref)


def test_a_training_module_cannot_be_packaged_as_a_predictor(tmp_path: Path) -> None:
    training = DsioModule(model=_linear(), objective=EmptyObjective())

    with pytest.raises(PredictorError, match="DsioModule.*model only"):
        _build(_checkpoint(tmp_path), model=training)


def test_predictor_does_not_share_a_mutable_validator_with_its_caller(tmp_path: Path) -> None:
    validator = ThresholdValidator(100.0)
    predictor = _build(_checkpoint(tmp_path), validator=validator)
    validator.maximum = 0.0

    result = predictor({"sample_id": ["a"], "x": torch.ones(1, 2)})

    assert result["sample_id"] == ["a"]


def test_validator_cannot_mutate_the_returned_prediction(tmp_path: Path) -> None:
    predictor = _build(_checkpoint(tmp_path), validator=corrupting_validator)

    result = predictor({"sample_id": ["real"], "x": torch.ones(1, 2)})

    assert result["sample_id"] == ["real"]


def test_in_place_preprocessing_does_not_mutate_caller_input(tmp_path: Path) -> None:
    x = torch.tensor([[-1.0, 2.0]])
    batch = {"sample_id": ["a"], "x": x}
    original = x.clone()
    predictor = _build(_checkpoint(tmp_path), preprocessor=nn.ReLU(inplace=True))

    predictor(batch)

    torch.testing.assert_close(batch["x"], original)


def test_checkpoint_and_predictor_remain_distinct_artifacts(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    before = load_artifact(ref)
    predictor = _build(ref)

    assert load_artifact(ref) == before
    assert all(not name.startswith("objective") for name in predictor.state_dict())
    assert all("training_augmentation" not in name for name in predictor.state_dict())
    client = MlflowClient(resolve_tracking_uri())
    assert client.search_registered_models() == []
