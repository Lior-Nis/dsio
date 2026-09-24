"""The supervised example proves the whole public spine as consumer-owned code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from lightning import Trainer
from mlflow import MlflowClient

from dsio.data.loading import DsioDataModule
from dsio.model.module import DsioModule
from dsio.tracking import TrackingError, canonical_dataset_digest, experiment


def test_training_and_inference_share_channel_first_signal_layout(
    tmp_path: Path,
    reference_services: None,
) -> None:
    del reference_services
    from reference_projects.supervised.components import (
        RegressionSamples,
        TimeMajorToChannelFirst,
        evaluation_arrays,
    )

    from dsio.data.store import SignalStore

    values = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=np.float32,
    )
    path = tmp_path / "layout-store"
    with SignalStore.builder(path, channels=2, dtype="float32") as builder:
        builder.add("sample", values, group="group", attrs={"target": 0.0})
    store = SignalStore(path)
    raw, _ = evaluation_arrays(str(path), ["sample"])

    prepared = TimeMajorToChannelFirst(channels=2, time=3)(torch.from_numpy(raw["x"]))

    training_tensor = RegressionSamples(store, ["sample"])[0]["x"]
    assert prepared.shape == (1, 2, 3)
    assert prepared.is_contiguous()
    assert training_tensor.is_contiguous()
    torch.testing.assert_close(prepared[0], training_tensor)
    torch.testing.assert_close(
        prepared,
        torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]]),
    )


def test_regressor_rejects_time_major_input(reference_services: None) -> None:
    del reference_services
    from reference_projects.supervised.components import TinyRegressor

    with pytest.raises(ValueError, match=r"\[batch, channels, time\].*\(batch, 1, 4\)"):
        TinyRegressor()(torch.ones(2, 4, 1))


def test_supervised_reference_flow_replays_and_reevaluates_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_services: None,
) -> None:
    del reference_services
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.supervised.components import (
        RegressionObjective,
        TinyRegressor,
        build_synthetic_store,
    )
    from reference_projects.supervised.flow import reevaluate, supervised_flow
    from reference_projects.supervised.tasks import infer, split_data

    observed_runtime: list[dict[str, Any]] = []
    original_fit = Trainer.fit

    def recording_fit(
        trainer: Trainer,
        model: object,
        *args: object,
        **kwargs: object,
    ) -> Any:
        datamodule = kwargs.get("datamodule")
        assert isinstance(datamodule, DsioDataModule)
        assert isinstance(model, DsioModule)
        result = original_fit(trainer, model, *args, **kwargs)
        train_batches = list(datamodule.train_dataloader())
        validate_batches = list(datamodule.val_dataloader())
        observed_runtime.append(
            {
                "types": (type(model), type(datamodule)),
                "model_type": type(model.model),
                "objective_type": type(model.objective),
                "optimizer_factory": model.optimizer_factory,
                "optimizer_parameters": model.optimizer_parameters,
                "drop_last": datamodule.drop_last,
                "shuffle": datamodule.shuffle,
                "max_epochs": trainer.max_epochs,
                "limit_val_batches": trainer.limit_val_batches,
                "num_sanity_val_steps": trainer.num_sanity_val_steps,
                "num_devices": trainer.num_devices,
                "precision": str(trainer.precision),
                "checkpoint_callbacks": list(trainer.checkpoint_callbacks),
                "train_batch_sizes": [len(batch["sample_id"]) for batch in train_batches],
                "train_sample_ids": sorted(
                    sample_id
                    for batch in train_batches
                    for sample_id in batch["sample_id"]
                ),
                "validate_sample_ids": sorted(
                    sample_id
                    for batch in validate_batches
                    for sample_id in batch["sample_id"]
                ),
            }
        )
        return result

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

    expected_runtime = {
        "types": (DsioModule, DsioDataModule),
        "model_type": TinyRegressor,
        "objective_type": RegressionObjective,
        "optimizer_factory": torch.optim.SGD,
        "optimizer_parameters": {"lr": 0.05},
        "drop_last": {
            "train": False,
            "validate": False,
            "test": False,
            "predict": False,
        },
        "shuffle": {
            "train": True,
            "validate": False,
            "test": False,
            "predict": False,
        },
        "max_epochs": 3,
        "limit_val_batches": 1.0,
        "num_sanity_val_steps": 0,
        "num_devices": 1,
        "precision": "32-true",
        "checkpoint_callbacks": [],
        "train_batch_sizes": [4, 2],
        "train_sample_ids": sorted(first["assignments"]["train"]),
        "validate_sample_ids": sorted(first["assignments"]["test"]),
    }
    assert observed_runtime == [expected_runtime, expected_runtime]
    assert first["split_digest"] == second["split_digest"]
    assert first["assignments"] == second["assignments"]
    assert first["identities"] == second["identities"]
    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    assert first["prediction"].shape == (len(first["test_sample_id"]), 1)
    assert np.isfinite(first["prediction"]).all()
    assert set(first["metrics"]) == {"mae", "rmse"}
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
        assert canonical_dataset_digest(training.inputs.dataset_inputs[0]) == result[
            "dataset_digest"
        ]
        provenance_path = client.download_artifacts(
            result["train_run_id"], "provenance.json", str(tmp_path / result["train_run_id"])
        )
        provenance = json.loads(Path(provenance_path).read_text())
        expected_training_configuration = {
            "batch_size": 4,
            "drop_last": {
                "train": False,
                "validate": False,
                "test": False,
                "predict": False,
            },
            "fold": 0,
            "num_workers": 0,
            "optimizer_parameters": {"lr": 0.05},
            "roles": {"train": "train", "validate": "test"},
            "shuffle": {
                "train": True,
                "validate": False,
                "test": False,
                "predict": False,
            },
            "trainer": {
                "accelerator": "cpu",
                "accumulate_grad_batches": 1,
                "checkpoint": False,
                "deterministic": True,
                "devices": 1,
                "early_stopping_patience": None,
                "enable_progress_bar": False,
                "gradient_clip_val": None,
                "limit_val_batches": 1.0,
                "log_every_n_steps": 1,
                "max_epochs": 3,
                "monitor": "val/loss",
                "monitor_mode": "min",
                "num_sanity_val_steps": 0,
                "precision": "32-true",
            },
        }
        for key, value in expected_training_configuration.items():
            assert provenance["configuration"][key] == value
        assert provenance["components"]["optimizer"] == "torch.optim:SGD"
        assert provenance["components"]["dataset_factory"] == (
            "reference_projects.supervised.components:regression_samples"
        )
        execution = provenance["configuration"]["execution"]
        assert execution["requested_accelerator"] == "cpu"
        assert execution["resolved_device"] == "cpu"
        assert execution["resolved_devices"] == "1"
        assert execution["resolved_precision"] == "32-true"
        assert execution["resolved_deterministic"] == "warn"
        assert execution["strategy"]
        assert execution["torch_version"] == torch.__version__
        for key, value in execution.items():
            assert training.data.params[f"execution.{key}"] == value
        for removed in (
            "accelerator",
            "deterministic",
            "devices",
            "limit_val_batches",
            "max_epochs",
            "num_sanity_val_steps",
        ):
            assert removed not in provenance["configuration"]

        export_provenance_path = client.download_artifacts(
            result["export_run_id"],
            "provenance.json",
            str(tmp_path / result["export_run_id"]),
        )
        export_provenance = json.loads(Path(export_provenance_path).read_text())
        assert export_provenance["configuration"]["checkpoint_digest"] == result[
            "checkpoint_digest"
        ]
        assert export_provenance["configuration"]["dataset_digest"] == result[
            "dataset_digest"
        ]
        assert export_provenance["configuration"]["export_form"] == "pyfunc"
        assert export_provenance["configuration"]["split_digest"] == result["split_digest"]
        assert export_provenance["components"]["builder"] == (
            "dsio.inference.predictor:build_predictor"
        )
        assert export_provenance["components"]["model"] == (
            "reference_projects.supervised.components:TinyRegressor"
        )
        assert export_provenance["components"]["normalizer"] == (
            "dsio.inference.predictor:TensorOutput"
        )
        assert export_provenance["components"]["preprocessor"] == (
            "reference_projects.supervised.components:TimeMajorToChannelFirst"
        )
        assert export_provenance["components"]["validator"] == (
            "dsio.inference.predictor:validate_tensor_prediction"
        )
        assert export_provenance["configuration"]["preprocessor"] == {
            "reference": "reference_projects.supervised.components:TimeMajorToChannelFirst",
            "parameters": {"channels": 1, "time": 4},
        }

        inference = client.get_run(result["inference_run_id"])
        assert canonical_dataset_digest(inference.inputs.dataset_inputs[0]) == result[
            "dataset_digest"
        ]
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
