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
from dsio.tracking import canonical_dataset_digest
from dsio.train.augmentation import TwoView


def test_legacy_ssl_preprocessor_reference_remains_resolvable(
    reference_services: None,
) -> None:
    del reference_services
    from reference_projects.self_supervised.components import (
        TimeMajorToChannelFirst as legacy_preprocessor,
    )
    from reference_projects.supervised.components import TimeMajorToChannelFirst

    assert legacy_preprocessor is TimeMajorToChannelFirst


def test_time_major_preprocessor_preserves_values_and_input(reference_services: None) -> None:
    del reference_services
    from reference_projects.supervised.components import TimeMajorToChannelFirst

    source = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ]
    )
    original = source.clone()

    prepared = TimeMajorToChannelFirst(channels=2, time=3)(source)

    assert prepared.shape == (2, 2, 3)
    assert prepared.is_contiguous()
    torch.testing.assert_close(
        prepared,
        torch.tensor(
            [
                [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
            ]
        ),
    )
    torch.testing.assert_close(source, original)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (torch.ones(3, 2), r"\[batch, time, channels\]"),
        (torch.ones(1, 4, 2), "time extent 3"),
        (torch.ones(1, 3, 1), "channel extent 2"),
    ],
)
def test_time_major_preprocessor_rejects_wrong_external_shape(
    value: torch.Tensor,
    message: str,
    reference_services: None,
) -> None:
    del reference_services
    from reference_projects.supervised.components import TimeMajorToChannelFirst

    with pytest.raises(ValueError, match=message):
        TimeMajorToChannelFirst(channels=2, time=3)(value)


@pytest.mark.parametrize(
    "parameters",
    [
        {"channels": True, "time": 3},
        {"channels": 1.5, "time": 3},
        {"channels": 0, "time": 3},
        {"channels": 2, "time": False},
        {"channels": 2, "time": -1},
    ],
)
def test_time_major_preprocessor_requires_positive_integer_extents(
    parameters: dict[str, Any],
    reference_services: None,
) -> None:
    del reference_services
    from reference_projects.supervised.components import TimeMajorToChannelFirst

    with pytest.raises(ValueError, match="positive integer"):
        TimeMajorToChannelFirst(**parameters)


def test_predictor_preprocessing_matches_the_training_dataset_tensor(
    tmp_path: Path,
    reference_services: None,
) -> None:
    del reference_services
    from reference_projects.self_supervised.components import UnlabelledSamples
    from reference_projects.supervised.components import TimeMajorToChannelFirst, evaluation_arrays

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

    torch.testing.assert_close(prepared[0], UnlabelledSamples(store, ["sample"])[0]["x"])


def test_embedding_rejects_time_major_input(reference_services: None) -> None:
    del reference_services
    from reference_projects.self_supervised.components import TinyEmbedding

    with pytest.raises(ValueError, match=r"\[batch, channels, time\].*\(batch, 1, 4\)"):
        TinyEmbedding()(torch.ones(2, 4, 1))


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
    runtime_configuration: list[dict[str, Any]] = []
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
        data_module = kwargs.get("datamodule")
        fits.append((type(module), type(data_module), module.model, module.objective))
        assert isinstance(data_module, DsioDataModule)
        runtime_configuration.append(
            {
                "augmentation": module.augmentation_identity,
                "drop_last": data_module.drop_last,
                "limit_val_batches": trainer.limit_val_batches,
                "shuffle": data_module.shuffle,
            }
        )
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
    assert (
        runtime_configuration
        == [
            {
                "augmentation": {
                    "augmentor": {
                        "reference": "dsio.model.components:Jitter",
                        "parameters": {"sigma": 0.1},
                    },
                    "wrapper": {
                        "reference": "dsio.train.augmentation:TwoView",
                        "parameters": {"views": ["online", "target"]},
                    },
                },
                "drop_last": {
                    "train": True,
                    "validate": False,
                    "test": False,
                    "predict": False,
                },
                "limit_val_batches": 0,
                "shuffle": {
                    "train": True,
                    "validate": False,
                    "test": False,
                    "predict": False,
                },
            }
        ]
        * 2
    )
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
        assert left["source_x"].shape[0] == 4

    assert first["identities"] == second["identities"]
    assert first["assignments"] == second["assignments"]
    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    assert first["prediction"].shape == (len(first["test_sample_id"]), 1)
    assert np.isfinite(first["prediction"]).all()
    assert (first["prediction"] >= 0).all()
    assert first["inference_sample_id"] == first["test_sample_id"]

    client = MlflowClient()
    for result in (first, second):
        flow_run_ids: set[str] = set()
        for role in ("data", "split", "train", "export", "evaluation", "inference"):
            run = client.get_run(result[f"{role}_run_id"])
            assert run.info.experiment_id == result["experiment_id"]
            assert run.info.status == "FINISHED"
            assert "mlflow.parentRunId" not in run.data.tags
            flow_run_ids.add(run.data.tags["dsio.prefect.flow_run_id"])
            assert run.data.tags["dsio.execution_identity"] == result["identities"][role]
        assert len(flow_run_ids) == 1

        training = client.get_run(result["train_run_id"])
        assert (
            canonical_dataset_digest(training.inputs.dataset_inputs[0]) == result["dataset_digest"]
        )
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
        assert (
            provenance["configuration"]["augmentation"] == runtime_configuration[0]["augmentation"]
        )
        assert provenance["configuration"]["augmentation_seed"] == 23
        assert provenance["configuration"]["drop_last"] == runtime_configuration[0]["drop_last"]
        assert provenance["configuration"]["shuffle"] == runtime_configuration[0]["shuffle"]
        assert provenance["configuration"]["trainer"] == {
            "accelerator": "auto",
            "accumulate_grad_batches": 1,
            "checkpoint": False,
            "deterministic": True,
            "devices": 1,
            "early_stopping_patience": None,
            "enable_progress_bar": False,
            "gradient_clip_val": None,
            "limit_val_batches": 0,
            "log_every_n_steps": 1,
            "max_epochs": 3,
            "monitor": "val/loss",
            "monitor_mode": "min",
            "num_sanity_val_steps": None,
            "precision": "32-true",
        }
        for removed in (
            "accelerator",
            "deterministic",
            "devices",
            "limit_val_batches",
            "max_epochs",
        ):
            assert removed not in provenance["configuration"]
        execution = provenance["configuration"]["execution"]
        assert execution["requested_accelerator"] == "auto"
        assert execution["resolved_device"].split(":", maxsplit=1)[0] == views[0][0]["device"]
        assert execution["resolved_precision"] == "32-true"
        assert execution["resolved_deterministic"] == "warn"
        assert execution["strategy"]
        assert execution["torch_version"] == torch.__version__
        assert training.data.params["execution.resolved_device"] == execution["resolved_device"]
        assert training.data.params["execution.torch_version"] == torch.__version__
        assert provenance["configuration"]["objective_parameters"] == {"temperature": 0.2}
        export_provenance_path = client.download_artifacts(
            result["export_run_id"],
            "provenance.json",
            str(tmp_path / result["export_run_id"]),
        )
        export_provenance = json.loads(Path(export_provenance_path).read_text())
        assert export_provenance["configuration"]["split_digest"] == result["split_digest"]
        assert export_provenance["components"]["preprocessor"] == (
            "reference_projects.supervised.components:TimeMajorToChannelFirst"
        )
        assert export_provenance["configuration"]["preprocessor"] == {
            "reference": ("reference_projects.supervised.components:TimeMajorToChannelFirst"),
            "parameters": {"channels": 1, "time": 4},
        }
        evaluation = client.get_run(result["evaluation_run_id"])
        inference = client.get_run(result["inference_run_id"])
        model_id = result["model_uri"].removeprefix("models:/")
        assert [item.model_id for item in evaluation.inputs.model_inputs] == [model_id]
        assert [item.model_id for item in inference.inputs.model_inputs] == [model_id]
