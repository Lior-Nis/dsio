from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from mlflow import MlflowClient
from tests.kaggle_portfolio.assertions import (
    assert_downstream_evidence,
    assert_execution_evidence,
    assert_replay_run_ids_differ,
)

from dsio.contracts import sha256_of_bytes
from dsio.data.loading import DsioDataModule
from dsio.model.module import DsioModule


def test_rogii_boundary_joins_wells_and_reconstructs_hidden_tail(rogii_csvs: Path) -> None:
    from reference_projects.kaggle.rogii.components import pad_wells
    from reference_projects.kaggle.rogii.data import load_competition_data

    loaded = load_competition_data(rogii_csvs)

    assert len(loaded["train"]) == 8
    assert loaded["test_ids"] == ["well00", "well01"]
    assert loaded["excluded_train_ids"] == ["well00", "well01"]
    assert [len(well["tail"]) for well in loaded["test"]] == [4, 5]
    assert loaded["submission_ids"] == [
        *[f"well00_{index}" for index in range(5, 9)],
        *[f"well01_{index}" for index in range(6, 11)],
    ]
    batch = pad_wells(
        [
            {
                "sample_id": "short",
                "x": torch.ones(3, 13),
                "y": torch.ones(3),
            },
            {
                "sample_id": "long",
                "x": torch.ones(5, 13),
                "y": torch.ones(5),
            },
        ]
    )
    assert tuple(batch["x"].shape) == (2, 5, 13)
    assert batch["x"][0, 3:, -1].eq(0).all()

    paired = rogii_csvs / "train" / "well07__typewell.csv"
    paired.unlink()
    with pytest.raises(ValueError, match="paired typewell"):
        load_competition_data(rogii_csvs)


def test_rogii_flow_trains_masked_dense_regression_without_test_leakage(
    rogii_csvs: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    from lightning import Trainer
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.rogii.flow import rogii_flow

    observed: list[tuple[type[object], type[object], bool]] = []
    original = Trainer.fit

    def record(trainer: Trainer, model: object, *args: object, **kwargs: object) -> object:
        datamodule = kwargs["datamodule"]
        result = original(trainer, model, *args, **kwargs)
        batch = next(iter(datamodule.train_dataloader()))
        observed.append(
            (
                type(model),
                type(datamodule),
                bool(batch["x"][:, :, -1].eq(0).any()),
            )
        )
        return result

    monkeypatch.setattr(Trainer, "fit", record)

    with prefect_test_harness():
        result = rogii_flow(str(rogii_csvs), str(tmp_path / "work"), seed=17)
        replay = rogii_flow(str(rogii_csvs), str(tmp_path / "work"), seed=17)

    assert result["split_algorithm"] == "group_shuffle"
    assert result["split_parameters"] == {"test_size": 0.25}
    assert set(result["assignments"]["train"]).isdisjoint(result["assignments"]["validate"])
    assigned = [
        sample_id for role in ("train", "validate") for sample_id in result["assignments"][role]
    ]
    assert not any("well00" in value or "well01" in value for value in assigned)
    assert observed == [
        (DsioModule, DsioDataModule, True),
        (DsioModule, DsioDataModule, True),
    ]
    assert result["prediction_count"] == 9
    assert np.isfinite(result["prediction"]).all()
    assert np.isfinite(result["metrics"]["rmse"])
    assert np.isfinite(result["metrics"]["last_value_rmse"])
    assert result["metrics"]["rmse"] <= result["metrics"]["last_value_rmse"]
    assert_replay_run_ids_differ(result, replay)
    assert result["split_digest"] == replay["split_digest"]
    assert result["identities"] == replay["identities"]
    assert result["metrics"] == replay["metrics"]
    assert result["prediction"] == replay["prediction"]
    assert result["submission_bytes"] == replay["submission_bytes"]
    lines = result["submission_bytes"].decode().splitlines()
    assert lines[0] == "id,tvt"
    assert [line.split(",", 1)[0] for line in lines[1:]] == [
        *[f"well00_{index}" for index in range(5, 9)],
        *[f"well01_{index}" for index in range(6, 11)],
    ]
    assert result["submission_digest"] == sha256_of_bytes(result["submission_bytes"])
    run = MlflowClient().get_run(result["evaluation_run_id"])
    assert run.data.metrics["rmse"] == pytest.approx(result["metrics"]["rmse"])
    assert run.data.metrics["last_value_rmse"] == pytest.approx(
        result["metrics"]["last_value_rmse"]
    )
    assert_execution_evidence(
        result["train_run_id"],
        tmp_path / "rogii-provenance",
        optimizer="torch.optim:Adam",
        optimizer_parameters={"lr": 0.001},
        batch_size=8,
        num_workers=2,
    )
    assert_downstream_evidence(result, tmp_path / "rogii-downstream")
