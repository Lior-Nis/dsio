from __future__ import annotations

import csv
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


def test_essay_boundary_and_quadratic_weighted_kappa(essay_scoring_csvs: Path) -> None:
    from reference_projects.kaggle.essay_scoring.components import (
        MAX_TOKENS,
        pad_essays,
        quadratic_weighted_kappa,
        tokenize,
    )
    from reference_projects.kaggle.essay_scoring.data import load_competition_data

    loaded = load_competition_data(essay_scoring_csvs)

    assert len(loaded["train"]) == 36
    assert loaded["train"][0]["full_text"] == str(loaded["train"][0]["full_text"]).strip()
    assert loaded["test_ids"] == ["essay-000", *[f"test-{index:03d}" for index in range(1, 5)]]
    assert {row["score"] for row in loaded["train"]} == set(range(1, 7))
    assert tokenize("Same words, same IDs!") == tokenize("Same words, same IDs!")
    assert 0 < len(tokenize("word " * (MAX_TOKENS + 10))) == MAX_TOKENS
    batch = pad_essays(
        [
            {"sample_id": "short", "x": torch.ones(2, 2), "y": torch.tensor(1)},
            {"sample_id": "long", "x": torch.ones(5, 2), "y": torch.tensor(2)},
        ]
    )
    assert tuple(batch["x"].shape) == (2, 5, 2)
    assert batch["x"][0, 2:, 1].eq(0).all()
    assert quadratic_weighted_kappa(
        np.asarray([1, 2, 3, 4, 5, 6]),
        np.asarray([1, 2, 4, 3, 6, 5]),
    ) == pytest.approx(0.8857142857142857)

    path = essay_scoring_csvs / "train.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["score"] = "7"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["essay_id", "full_text", "score"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="score"):
        load_competition_data(essay_scoring_csvs)


def test_essay_flow_trains_variable_length_ordinal_predictions(
    essay_scoring_csvs: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    from lightning import Trainer
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.essay_scoring.flow import essay_scoring_flow

    observed: list[tuple[type[object], type[object], tuple[int, ...], bool]] = []
    original = Trainer.fit

    def record(trainer: Trainer, model: object, *args: object, **kwargs: object) -> object:
        datamodule = kwargs["datamodule"]
        result = original(trainer, model, *args, **kwargs)
        batch = next(iter(datamodule.train_dataloader()))
        observed.append(
            (
                type(model),
                type(datamodule),
                tuple(batch["x"].shape),
                bool(batch["x"][:, :, 1].eq(0).any()),
            )
        )
        return result

    monkeypatch.setattr(Trainer, "fit", record)

    with prefect_test_harness():
        result = essay_scoring_flow(str(essay_scoring_csvs), str(tmp_path / "work"), seed=13)
        replay = essay_scoring_flow(str(essay_scoring_csvs), str(tmp_path / "work"), seed=13)

    assert result["split_algorithm"] == "stratified_group_kfold"
    assert result["split_parameters"] == {"n_splits": 3, "target": "score"}
    assert set(result["assignments"]) == {"train", "validate"}
    assert set(result["assignments"]["train"]).isdisjoint(result["assignments"]["validate"])
    assert observed == [
        (DsioModule, DsioDataModule, observed[0][2], True),
        (DsioModule, DsioDataModule, observed[1][2], True),
    ]
    assert all(shape[1] <= 512 and shape[2] == 2 for _, _, shape, _ in observed)
    assert result["prediction"] and set(result["prediction"]) <= set(range(1, 7))
    assert len(result["prediction"]) == 5
    assert np.isfinite(result["metrics"]["quadratic_weighted_kappa"])
    assert_replay_run_ids_differ(result, replay)
    assert result["split_digest"] == replay["split_digest"]
    assert result["identities"] == replay["identities"]
    assert result["metrics"] == replay["metrics"]
    assert result["prediction"] == replay["prediction"]
    assert result["submission_bytes"] == replay["submission_bytes"]
    lines = result["submission_bytes"].decode().splitlines()
    assert lines[0] == "essay_id,score"
    assert [line.split(",", 1)[0] for line in lines[1:]] == [
        "essay-000",
        *[f"test-{index:03d}" for index in range(1, 5)],
    ]
    assert result["submission_digest"] == sha256_of_bytes(result["submission_bytes"])
    run = MlflowClient().get_run(result["evaluation_run_id"])
    assert run.data.metrics["quadratic_weighted_kappa"] == pytest.approx(
        result["metrics"]["quadratic_weighted_kappa"]
    )
    assert_execution_evidence(
        result["train_run_id"],
        tmp_path / "essay-provenance",
        optimizer="torch.optim:Adam",
        optimizer_parameters={"lr": 0.01},
        batch_size=128,
        num_workers=2,
    )
    assert_downstream_evidence(result, tmp_path / "essay-downstream")
