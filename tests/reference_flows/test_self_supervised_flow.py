"""The SSL reference project varies components, not the reusable training spine."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from lightning import Trainer
from mlflow import MlflowClient

from dsio.data.loading import DsioDataModule
from dsio.model.module import DsioModule
from dsio.train.augmentation import TwoView


def test_self_supervised_reference_replays_accelerator_views_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_services: None,
) -> None:
    del reference_services
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.self_supervised.components import (
        ContrastiveObjective,
        TinyEmbedding,
    )
    from reference_projects.self_supervised.flow import self_supervised_flow

    fits: list[tuple[type[object], type[object], object, object]] = []
    views: list[list[dict[str, Any]]] = []
    in_training_step = False
    original_fit = Trainer.fit
    original_step = DsioModule.training_step
    original_augment = TwoView.forward

    def recording_fit(
        trainer: Trainer,
        module: DsioModule,
        *args: object,
        **kwargs: object,
    ) -> Any:
        fits.append((type(module), type(kwargs.get("datamodule")), module.model, module.objective))
        views.append([])
        return original_fit(trainer, module, *args, **kwargs)

    def recording_step(
        module: DsioModule,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        nonlocal in_training_step
        in_training_step = True
        try:
            return original_step(module, batch, batch_idx)
        finally:
            in_training_step = False

    def recording_augment(
        augmentation: TwoView,
        batch: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert in_training_step
        result = original_augment(augmentation, batch, **kwargs)
        views[-1].append(
            {
                "device": result["x"].device.type,
                "sample_id": list(result["sample_id"]),
                "source_x": batch["x"].detach().cpu().clone(),
                "view_id": list(result["view_id"]),
                "x": result["x"].detach().cpu().clone(),
                "y": result["y"].detach().cpu().clone(),
            }
        )
        return result

    monkeypatch.setattr(Trainer, "fit", recording_fit)
    monkeypatch.setattr(DsioModule, "training_step", recording_step)
    monkeypatch.setattr(TwoView, "forward", recording_augment)
    with prefect_test_harness():
        workspace = str(tmp_path / "workspace")
        first = self_supervised_flow(workspace, seed=23)
        second = self_supervised_flow(workspace, seed=23)

    assert [(module, data) for module, data, _, _ in fits] == [
        (DsioModule, DsioDataModule),
        (DsioModule, DsioDataModule),
    ]
    assert all(isinstance(model, TinyEmbedding) for _, _, model, _ in fits)
    assert all(isinstance(objective, ContrastiveObjective) for _, _, _, objective in fits)
    assert len(views[0]) == len(views[1]) > 0
    for left, right in zip(views[0], views[1], strict=True):
        assert left["device"] == right["device"]
        assert left["sample_id"] == right["sample_id"]
        assert left["view_id"] == right["view_id"]
        midpoint = len(left["sample_id"]) // 2
        assert left["sample_id"][:midpoint] * 2 == left["sample_id"]
        assert left["view_id"] == ["online"] * midpoint + ["target"] * midpoint
        assert not torch.equal(left["x"][:midpoint], left["source_x"])
        assert not torch.equal(left["x"][midpoint:], left["source_x"])
        assert not torch.equal(left["x"][:midpoint], left["x"][midpoint:])
        assert torch.equal(left["x"], right["x"])
        assert torch.equal(left["y"], right["y"])

    assert first["identities"] == second["identities"]
    assert first["assignments"] == second["assignments"]
    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    assert first["prediction"].shape == (len(first["test_sample_id"]), 1)
    assert np.isfinite(first["prediction"]).all()
    assert (first["prediction"] >= 0).all()
    assert first["inference_sample_id"] == first["test_sample_id"]

    client = MlflowClient()
    for result in (first, second):
        assert client.get_run(result["parent_run_id"]).info.status == "FINISHED"
        for role in ("data", "split", "train", "export", "evaluation", "inference"):
            run = client.get_run(result[f"{role}_run_id"])
            assert run.info.status == "FINISHED"
            assert run.data.tags["mlflow.parentRunId"] == result["parent_run_id"]
            assert run.data.tags["dsio.execution_identity"] == result["identities"][role]

        training = client.get_run(result["train_run_id"])
        assert training.inputs.dataset_inputs[0].dataset.digest == result["dataset_digest"]
        split_provenance_path = client.download_artifacts(
            result["split_run_id"], "provenance.json", str(tmp_path / result["split_run_id"])
        )
        split_provenance = json.loads(Path(split_provenance_path).read_text())
        assert split_provenance["configuration"]["name"] == "self-supervised-holdout"
        provenance_path = client.download_artifacts(
            result["train_run_id"], "provenance.json", str(tmp_path / result["train_run_id"])
        )
        provenance = json.loads(Path(provenance_path).read_text())
        assert provenance["components"]["augmentation"] == ("dsio.train.augmentation:TwoView")
        assert provenance["components"]["objective"] == (
            "reference_projects.self_supervised.components:ContrastiveObjective"
        )
        assert provenance["configuration"]["augmentation"]["views"] == [
            "online",
            "target",
        ]
        assert provenance["configuration"]["augmentation_seed"] == 23
        execution = provenance["configuration"]["execution"]
        assert execution["requested_accelerator"] == "auto"
        assert execution["resolved_device"].split(":", maxsplit=1)[0] == views[0][0]["device"]
        assert execution["resolved_precision"] == "32-true"
        assert execution["strategy"]
        assert execution["torch_version"] == torch.__version__
        assert training.data.params["execution.resolved_device"] == execution["resolved_device"]
        assert training.data.params["execution.torch_version"] == torch.__version__
        assert provenance["configuration"]["objective_parameters"] == {"temperature": 0.2}
        evaluation = client.get_run(result["evaluation_run_id"])
        inference = client.get_run(result["inference_run_id"])
        model_id = result["model_uri"].removeprefix("models:/")
        assert [item.model_id for item in evaluation.inputs.model_inputs] == [model_id]
        assert [item.model_id for item in inference.inputs.model_inputs] == [model_id]
