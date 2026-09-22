"""The supervised example proves the whole public spine as consumer-owned code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from lightning import Trainer
from mlflow import MlflowClient

from dsio.data.loading import DsioDataModule
from dsio.model.module import DsioModule
from dsio.tracking import TrackingError, experiment


def test_supervised_reference_flow_replays_and_reevaluates_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_services: None,
) -> None:
    del reference_services
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.supervised.components import build_synthetic_store
    from reference_projects.supervised.flow import reevaluate, supervised_flow
    from reference_projects.supervised.tasks import infer, split_data

    observed_types: list[tuple[type[object], type[object]]] = []
    original_fit = Trainer.fit

    def recording_fit(
        trainer: Trainer,
        model: object,
        *args: object,
        **kwargs: object,
    ) -> Any:
        datamodule = kwargs.get("datamodule")
        observed_types.append((type(model), type(datamodule)))
        return original_fit(trainer, model, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", recording_fit)
    with prefect_test_harness():
        workspace = str(tmp_path / "workspace")
        first = supervised_flow(workspace, seed=19)
        second = supervised_flow(workspace, seed=19)

        def forbidden_fit(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("downstream-only reevaluation invoked training")

        monkeypatch.setattr(Trainer, "fit", forbidden_fit)
        downstream = reevaluate(first, metrics=("rmse",))

        other_store = build_synthetic_store(tmp_path / "other-store", seed=999)
        with pytest.raises(ValueError, match="store identity"):
            reevaluate({**first, "store_path": str(other_store.path)}, metrics=("rmse",))

    assert observed_types == [
        (DsioModule, DsioDataModule),
        (DsioModule, DsioDataModule),
    ]
    assert first["split_digest"] == second["split_digest"]
    assert first["assignments"] == second["assignments"]
    assert first["identities"] == second["identities"]
    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    assert first["inference_sample_id"] == first["test_sample_id"]

    assert downstream["metrics"] == {"rmse": first["metrics"]["rmse"]}
    assert downstream["evaluation_run_id"] != first["evaluation_run_id"]
    assert downstream["identity"] != first["identities"]["evaluation"]

    client = MlflowClient()
    for result in (first, second):
        assert client.get_run(result["parent_run_id"]).info.status == "FINISHED"
        for role in ("data", "split", "train", "export", "evaluation", "inference"):
            run = client.get_run(result[f"{role}_run_id"])
            assert run.info.status == "FINISHED"
            assert run.data.tags["mlflow.parentRunId"] == result["parent_run_id"]
            assert run.data.tags["dsio.execution_identity"] == result["identities"][role]

        training = client.get_run(result["train_run_id"])
        assert len(training.inputs.dataset_inputs) == 1
        assert training.inputs.dataset_inputs[0].dataset.digest == result["dataset_digest"]
        provenance_path = client.download_artifacts(
            result["train_run_id"], "provenance.json", str(tmp_path / result["train_run_id"])
        )
        provenance = json.loads(Path(provenance_path).read_text())
        expected_training_configuration = {
            "batch_size": 4,
            "fold": 0,
            "max_epochs": 3,
            "optimizer_parameters": {"lr": 0.05},
            "roles": {"train": "train", "validate": "test"},
        }
        for key, value in expected_training_configuration.items():
            assert provenance["configuration"][key] == value
        assert provenance["components"]["optimizer"] == "torch.optim:SGD"
        assert provenance["components"]["dataset_factory"] == (
            "reference_projects.supervised.components:regression_samples"
        )

        inference = client.get_run(result["inference_run_id"])
        assert inference.inputs.dataset_inputs[0].dataset.digest == result["dataset_digest"]
        assert [item.model_id for item in inference.inputs.model_inputs] == [
            result["model_uri"].removeprefix("models:/")
        ]

    evaluation = client.get_run(downstream["evaluation_run_id"])
    assert evaluation.data.tags["dsio.evaluation.dataset_source_run_id"] == first["split_run_id"]
    assert [item.model_id for item in evaluation.inputs.model_inputs] == [
        first["model_uri"].removeprefix("models:/")
    ]
    assert evaluation.data.params["evaluation.metrics"] == '["rmse"]'

    client.delete_run(first["split_run_id"])
    with prefect_test_harness(), experiment("dsio-supervised-reference") as parent:
        with pytest.raises(TrackingError, match="not reusable"):
            infer(
                store_path=first["store_path"],
                sample_ids=first["test_sample_id"],
                model_uri=first["model_uri"],
                dataset_run_id=first["split_run_id"],
                dataset_digest=first["dataset_digest"],
                checkpoint_digest=first["checkpoint_digest"],
                parent_run_id=parent.info.run_id,
            )

    with prefect_test_harness(), experiment("dsio-supervised-reference") as parent:
        invalid_parent_run_id = parent.info.run_id
        experiment_id = parent.info.experiment_id
        with pytest.raises(Exception, match="no store"):
            split_data(
                {"store_path": str(tmp_path / "missing-store")},
                invalid_parent_run_id,
                19,
            )
    failed_children = [
        run
        for run in client.search_runs([experiment_id])
        if run.data.tags.get("mlflow.parentRunId") == invalid_parent_run_id
    ]
    assert len(failed_children) == 1
    assert failed_children[0].info.status == "FAILED"
