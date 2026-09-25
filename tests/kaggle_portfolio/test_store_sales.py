from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mlflow import MlflowClient
from tests.kaggle_portfolio.assertions import (
    assert_downstream_evidence,
    assert_execution_evidence,
    assert_replay_run_ids_differ,
)

from dsio.contracts import sha256_of_bytes
from dsio.data.loading import DsioDataModule
from dsio.model.module import DsioModule


def test_store_sales_boundary_builds_complete_series_and_rejects_ragged_rows(
    store_sales_csvs: Path,
) -> None:
    from reference_projects.kaggle.store_sales.data import load_competition_data

    loaded = load_competition_data(store_sales_csvs)

    assert len(loaded["series"]) == 4
    assert {len(series["train"]) for series in loaded["series"].values()} == {110}
    assert {len(series["test"]) for series in loaded["series"].values()} == {16}
    assert loaded["train_count"] == 440
    assert "train_ids" not in loaded
    assert loaded["test_order"] == list(range(440, 504))

    path = store_sales_csvs / "train.csv"
    original = path.read_text()
    lines = original.splitlines()
    lines[1] += ",unexpected"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="exactly"):
        load_competition_data(store_sales_csvs)
    path.write_text(original)

    lines = original.splitlines()
    fields = lines[1].split(",")
    fields[4] = "-1"
    lines[1] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="sales"):
        load_competition_data(store_sales_csvs)


def test_store_sales_flow_trains_a_causal_multi_horizon_forecaster(
    store_sales_csvs: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    from lightning import Trainer
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.store_sales.flow import store_sales_flow

    observed: list[tuple[list[str], list[str]]] = []
    original = Trainer.fit

    def record(trainer: Trainer, model: object, *args: object, **kwargs: object) -> object:
        datamodule = kwargs["datamodule"]
        assert type(model) is DsioModule
        assert type(datamodule) is DsioDataModule
        result = original(trainer, model, *args, **kwargs)
        observed.append(
            (
                sorted(
                    item for batch in datamodule.train_dataloader() for item in batch["sample_id"]
                ),
                sorted(
                    item for batch in datamodule.val_dataloader() for item in batch["sample_id"]
                ),
            )
        )
        return result

    monkeypatch.setattr(Trainer, "fit", record)

    with prefect_test_harness():
        result = store_sales_flow(str(store_sales_csvs), str(tmp_path / "work"), seed=11)
        replay = store_sales_flow(str(store_sales_csvs), str(tmp_path / "work"), seed=11)

    assert result["split_algorithm"] == "purged_walk_forward"
    assert result["split_parameters"]["n_splits"] == 2
    assert len(result["fold_assignments"]) == 2
    assert all(
        boundary["max_train_target_end"] <= boundary["min_validate_target_start"]
        for boundary in result["fold_boundaries"]
    )
    assert result["training_fold"] == 1
    final = result["fold_assignments"][result["training_fold"]]
    assert observed == [(sorted(final["train"]), sorted(final["validate"]))] * 2
    assert result["forecast_shape"] == [4, 16]
    assert len(result["prediction"]) == 64
    assert np.isfinite(result["prediction"]).all()
    assert min(result["prediction"]) >= 0
    assert np.isfinite(result["metrics"]["rmsle"])
    assert np.isfinite(result["metrics"]["seasonal_naive_rmsle"])
    assert_replay_run_ids_differ(result, replay)
    assert result["split_digest"] == replay["split_digest"]
    assert result["identities"] == replay["identities"]
    assert result["metrics"] == replay["metrics"]
    assert result["prediction"] == replay["prediction"]
    assert result["submission_bytes"] == replay["submission_bytes"]
    lines = result["submission_bytes"].decode().splitlines()
    assert lines[0] == "id,sales"
    assert [int(line.split(",", 1)[0]) for line in lines[1:]] == list(range(440, 504))
    assert result["submission_digest"] == sha256_of_bytes(result["submission_bytes"])
    run = MlflowClient().get_run(result["evaluation_run_id"])
    assert run.data.metrics["rmsle"] == pytest.approx(result["metrics"]["rmsle"])
    assert run.data.metrics["seasonal_naive_rmsle"] == pytest.approx(
        result["metrics"]["seasonal_naive_rmsle"]
    )
    assert_execution_evidence(
        result["train_run_id"],
        tmp_path / "store-sales-provenance",
        optimizer="torch.optim:Adam",
        optimizer_parameters={"lr": 0.01},
        fold=1,
        batch_size=64,
    )
    assert_downstream_evidence(result, tmp_path / "store-sales-downstream")
