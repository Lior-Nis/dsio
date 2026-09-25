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


def test_parkinsons_boundary_streams_windows_and_excludes_test_subjects(
    parkinsons_fog_csvs: Path,
) -> None:
    from reference_projects.kaggle.parkinsons_fog.data import (
        WINDOW_SIZE,
        iter_recording_windows,
        load_competition_data,
    )

    loaded = load_competition_data(parkinsons_fog_csvs)

    assert loaded["excluded_train_ids"] == ["same-subject", "shared"]
    assert loaded["test_subjects"] == ["held-out-subject", "test-subject"]
    assert loaded["submission_ids"][:2] == ["shared_0", "shared_1"]
    recording = next(value for value in loaded["train"] if value["recording_id"] == "defog0")
    windows = list(iter_recording_windows(recording, window_size=WINDOW_SIZE))
    assert sum(len(window["data"]) for window in windows) == 17
    assert max(len(window["data"]) for window in windows) <= WINDOW_SIZE
    mask = np.concatenate([window["data"][:, -1] for window in windows])
    assert mask.tolist() == [0.0, 0.0, *([1.0] * 13), 0.0, 0.0]

    (parkinsons_fog_csvs / "tdcsfog_metadata.csv").unlink()
    with pytest.raises(ValueError, match="metadata"):
        load_competition_data(parkinsons_fog_csvs)


def test_parkinsons_flow_trains_masked_dense_prediction_with_subject_split(
    parkinsons_fog_csvs: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    from lightning import Trainer
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.parkinsons_fog.flow import parkinsons_fog_flow

    observed: list[tuple[type[object], type[object], bool]] = []
    original = Trainer.fit

    def record(trainer: Trainer, model: object, *args: object, **kwargs: object) -> object:
        datamodule = kwargs["datamodule"]
        result = original(trainer, model, *args, **kwargs)
        batch = next(iter(datamodule.train_dataloader()))
        observed.append((type(model), type(datamodule), bool((~batch["mask"]).any())))
        return result

    monkeypatch.setattr(Trainer, "fit", record)

    with prefect_test_harness():
        result = parkinsons_fog_flow(str(parkinsons_fog_csvs), str(tmp_path / "work"), seed=17)
        replay = parkinsons_fog_flow(str(parkinsons_fog_csvs), str(tmp_path / "work"), seed=17)

    assert result["split_algorithm"] == "group_shuffle"
    assert result["split_parameters"] == {"test_size": 0.34}
    assert set(result["split_groups"]["train"]).isdisjoint(result["split_groups"]["validate"])
    assigned = [
        sample_id for role in ("train", "validate") for sample_id in result["assignments"][role]
    ]
    assert not any("shared" in value or "same-subject" in value for value in assigned)
    assert observed == [
        (DsioModule, DsioDataModule, True),
        (DsioModule, DsioDataModule, True),
    ]
    assert result["prediction_count"] == 20
    assert np.isfinite(result["prediction"]).all()
    assert all(0 <= value <= 1 for row in result["prediction"] for value in row)
    assert set(result["metrics"]) == {
        "StartHesitation_average_precision",
        "Turn_average_precision",
        "Walking_average_precision",
        "mean_average_precision",
        "StartHesitation_positive_rate",
        "Turn_positive_rate",
        "Walking_positive_rate",
        "mean_positive_rate",
    }
    assert all(np.isfinite(value) for value in result["metrics"].values())
    assert result["metrics"]["mean_average_precision"] >= result["metrics"]["mean_positive_rate"]
    assert result["ignored_points"] > 0
    assert_replay_run_ids_differ(result, replay)
    assert result["split_digest"] == replay["split_digest"]
    assert result["identities"] == replay["identities"]
    assert result["metrics"] == replay["metrics"]
    assert result["prediction"] == replay["prediction"]
    assert result["submission_bytes"] == replay["submission_bytes"]
    lines = result["submission_bytes"].decode().splitlines()
    assert lines[0] == "Id,StartHesitation,Turn,Walking"
    assert [line.split(",", 1)[0] for line in lines[1:]] == [
        *[f"shared_{index}" for index in range(9)],
        *[f"test-defog_{index}" for index in range(11)],
    ]
    assert result["submission_digest"] == sha256_of_bytes(result["submission_bytes"])
    run = MlflowClient().get_run(result["evaluation_run_id"])
    for name, value in result["metrics"].items():
        assert run.data.metrics[name] == pytest.approx(value)
    assert_execution_evidence(
        result["train_run_id"],
        tmp_path / "parkinsons-provenance",
        optimizer="torch.optim:Adam",
        optimizer_parameters={"lr": 0.001},
        batch_size=16,
        num_workers=2,
    )
    assert_downstream_evidence(result, tmp_path / "parkinsons-downstream")
