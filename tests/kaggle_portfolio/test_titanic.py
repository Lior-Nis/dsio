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


def test_titanic_csv_boundary_preserves_ids_and_rejects_duplicate_passengers(
    titanic_csvs: Path,
) -> None:
    from reference_projects.kaggle.titanic.data import load_competition_data

    loaded = load_competition_data(titanic_csvs)
    assert loaded["train_ids"] == [str(value) for value in range(1, 13)]
    assert loaded["test_ids"] == [str(value) for value in range(101, 105)]
    assert loaded["test_order"] == loaded["test_ids"]

    path = titanic_csvs / "test.csv"
    original = path.read_text()
    lines = original.splitlines()
    lines[1] += ",unexpected"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="exactly"):
        load_competition_data(titanic_csvs)
    path.write_text(original)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[1]["PassengerId"] = rows[0]["PassengerId"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="duplicate PassengerId"):
        load_competition_data(titanic_csvs)


def test_titanic_flow_is_replayable_and_ticket_groups_do_not_leak(
    titanic_csvs: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    from lightning import Trainer
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.titanic.flow import titanic_flow

    observed: list[tuple[type[object], type[object], list[str], list[str]]] = []
    original = Trainer.fit

    def record(trainer: Trainer, model: object, *args: object, **kwargs: object) -> object:
        datamodule = kwargs["datamodule"]
        assert type(model) is DsioModule
        assert type(datamodule) is DsioDataModule
        result = original(trainer, model, *args, **kwargs)
        observed.append(
            (
                type(model),
                type(datamodule),
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
        first = titanic_flow(str(titanic_csvs), str(tmp_path / "work"), seed=7)
        second = titanic_flow(str(titanic_csvs), str(tmp_path / "work"), seed=7)

    assert len(observed) == 2
    assert observed[0][2] == sorted(first["assignments"]["train"])
    assert observed[0][3] == sorted(first["assignments"]["validate"])
    assert first["split_digest"] == second["split_digest"]
    assert first["assignments"] == second["assignments"]
    assert first["identities"] == second["identities"]
    assert first["metrics"] == second["metrics"]
    assert first["prediction"] == second["prediction"]
    assert first["submission_bytes"] == second["submission_bytes"]
    assert_replay_run_ids_differ(first, second)
    train_tickets = set(first["tickets"][sample] for sample in first["assignments"]["train"])
    validate_tickets = set(first["tickets"][sample] for sample in first["assignments"]["validate"])
    assert train_tickets.isdisjoint(validate_tickets)
    assert set(first["assignments"]["train"] + first["assignments"]["validate"]) == {
        str(value) for value in range(1, 13)
    }
    assert first["submission_bytes"].decode().splitlines()[0] == "PassengerId,Survived"
    assert [line.split(",")[0] for line in first["submission_bytes"].decode().splitlines()[1:]] == [
        str(value) for value in range(101, 105)
    ]
    assert first["submission_digest"] == sha256_of_bytes(first["submission_bytes"])
    assert_execution_evidence(
        first["train_run_id"],
        tmp_path / "titanic-provenance",
        optimizer="torch.optim:SGD",
        optimizer_parameters={"lr": 0.02},
    )
    assert_downstream_evidence(first, tmp_path / "titanic-downstream")
