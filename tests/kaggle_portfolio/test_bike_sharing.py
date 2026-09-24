from __future__ import annotations

import csv
from pathlib import Path

import pytest
from tests.kaggle_portfolio.assertions import (
    assert_downstream_evidence,
    assert_execution_evidence,
    assert_replay_run_ids_differ,
)

from dsio.contracts import sha256_of_bytes
from dsio.data.loading import DsioDataModule
from dsio.model.module import DsioModule


def test_bike_csv_boundary_rejects_count_drift_and_preserves_datetime_order(
    bike_csvs: Path,
) -> None:
    from reference_projects.kaggle.bike_sharing.data import load_competition_data

    loaded = load_competition_data(bike_csvs)
    assert loaded["test_order"] == [f"2011-02-01 {hour:02d}:00:00" for hour in range(4)]

    path = bike_csvs / "train.csv"
    original = path.read_text()
    lines = original.splitlines()
    lines[1] += ",unexpected"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="exactly"):
        load_competition_data(bike_csvs)
    path.write_text(original)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["humidity"] = "101"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="humidity"):
        load_competition_data(bike_csvs)
    rows[0]["humidity"] = "40.0"
    rows[0]["count"] = "999"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=r"casual \+ registered"):
        load_competition_data(bike_csvs)


def test_bike_flow_has_one_causal_purged_holdout_and_nonnegative_ordered_submission(
    bike_csvs: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    from lightning import Trainer
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.bike_sharing.flow import bike_sharing_flow

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
        result = bike_sharing_flow(str(bike_csvs), str(tmp_path / "work"), seed=11)
        replay = bike_sharing_flow(str(bike_csvs), str(tmp_path / "work"), seed=11)

    assert result["split_algorithm"] == "purged_walk_forward"
    assert result["split_parameters"]["n_splits"] == 1
    assert result["split_parameters"]["label_horizon"] > 0
    assert result["split_parameters"]["embargo_fraction"] > 0
    assert result["discarded_count"] > 0
    assert_replay_run_ids_differ(result, replay)
    assert result["split_digest"] == replay["split_digest"]
    assert result["identities"] == replay["identities"]
    assert result["metrics"] == replay["metrics"]
    assert result["prediction"] == replay["prediction"]
    assert result["submission_bytes"] == replay["submission_bytes"]
    assert max(result["train_times"]) + result["split_parameters"]["label_horizon"] <= min(
        result["validate_times"]
    )
    assert (
        len(result["assignments"]["train"])
        + len(result["assignments"]["validate"])
        + result["discarded_count"]
        == 24
    )
    assert set(result["scaler_fit_ids"]) == set(result["assignments"]["train"])
    assert len(result["prediction"]) == 4
    assert all(value >= 0 for value in result["prediction"])
    lines = result["submission_bytes"].decode().splitlines()
    assert lines[0] == "datetime,count"
    assert [line.rsplit(",", 1)[0] for line in lines[1:]] == [
        f"2011-02-01 {hour:02d}:00:00" for hour in range(4)
    ]
    assert result["submission_digest"] == sha256_of_bytes(result["submission_bytes"])
    assert (
        observed
        == [
            (
                sorted(result["assignments"]["train"]),
                sorted(result["assignments"]["validate"]),
            )
        ]
        * 2
    )
    assert_execution_evidence(
        result["train_run_id"],
        tmp_path / "bike-provenance",
        optimizer="torch.optim:SGD",
        optimizer_parameters={"lr": 0.005},
    )
    assert_downstream_evidence(result, tmp_path / "bike-downstream")
