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
    with torch.no_grad():
        predictor.model.weight.fill_(float("nan"))  # type: ignore[union-attr]

    with pytest.raises(PredictorError, match="prediction.*finite"):
        predictor({"sample_id": ["a"], "x": torch.ones(1, 2)})


def test_normalizer_cannot_replace_source_identity(tmp_path: Path) -> None:
    predictor = _build(_checkpoint(tmp_path), normalizer=IdentityStealingNormalizer())

    with pytest.raises(PredictorError, match="normalizer cannot replace.*sample_id"):
        predictor({"sample_id": ["real"], "x": torch.ones(1, 2)})


def test_non_importable_validator_fails_during_construction(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="validator.*importable"):
        _build(_checkpoint(tmp_path), validator=lambda output: None)


def test_non_serializable_component_fails_during_construction(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="serializ"):
        _build(_checkpoint(tmp_path), preprocessor=NonSerializablePreprocessor())


def test_checkpoint_must_belong_to_a_successful_run(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="RUNNING.*FINISHED"):
        _build(_checkpoint(tmp_path, status="RUNNING"))


def test_checkpoint_state_must_match_the_inference_model_strictly(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="checkpoint model state"):
        _build(_checkpoint(tmp_path), model=nn.Linear(3, 1))


def test_a_training_module_cannot_be_packaged_as_a_predictor(tmp_path: Path) -> None:
    training = DsioModule(model=_linear(), objective=EmptyObjective())

    with pytest.raises(PredictorError, match="DsioModule.*model only"):
        _build(_checkpoint(tmp_path), model=training)


def test_checkpoint_and_predictor_remain_distinct_artifacts(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    before = load_artifact(ref)
    predictor = _build(ref)

    assert load_artifact(ref) == before
    assert all(not name.startswith("objective") for name in predictor.state_dict())
    assert all("training_augmentation" not in name for name in predictor.state_dict())
    client = MlflowClient(resolve_tracking_uri())
    assert client.search_registered_models() == []
