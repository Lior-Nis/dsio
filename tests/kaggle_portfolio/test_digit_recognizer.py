from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

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
from dsio.tracking import TrackingError, record_provenance
from dsio.train.artifacts import ArtifactIntegrityError, save_artifact


def test_digit_csv_boundary_requires_all_784_bounded_pixels(digit_csvs: Path) -> None:
    from reference_projects.kaggle.digit_recognizer.data import load_competition_data

    loaded = load_competition_data(digit_csvs)
    assert loaded["train_ids"] == [f"train-{value}" for value in range(1, 21)]
    assert loaded["test_ids"] == [str(value) for value in range(1, 5)]

    path = digit_csvs / "test.csv"
    original = path.read_text()
    lines = original.splitlines()
    lines[1] += ",unexpected"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="exactly"):
        load_competition_data(digit_csvs)
    path.write_text(original)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["pixel783"] = "256"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="pixel783"):
        load_competition_data(digit_csvs)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"label_fields": ["label"]}, "label-tainted"),
        ({"dataset_digest": "other"}, "different dataset"),
        ({"split_digest": "other"}, "different split"),
    ],
)
def test_digit_encoder_handoff_rejects_tainted_or_mismatched_lineage(
    change: dict[str, object], message: str
) -> None:
    from reference_projects.kaggle.digit_recognizer.tasks.training import (
        validate_encoder_handoff,
    )

    pretraining = {
        "label_fields": [],
        "dataset_digest": "dataset",
        "split_digest": "split",
        **change,
    }
    with pytest.raises(ValueError, match=message):
        validate_encoder_handoff(
            pretraining,
            {"dataset_digest": "dataset"},
            {"split_digest": "split"},
        )


@pytest.mark.parametrize(
    "state", ["failed", "deleted", "corrupted", "mismatched", "foreign", "label-tainted"]
)
def test_digit_encoder_handoff_rejects_unusable_artifact_evidence(
    state: str, kaggle_services: None
) -> None:
    del kaggle_services
    from reference_projects.kaggle.digit_recognizer.components import FrozenDigitClassifier
    from reference_projects.kaggle.digit_recognizer.tasks.training import _verified_encoder

    client = MlflowClient()
    experiment_id = client.create_experiment(f"digit-evidence-{state}")
    run = client.create_run(experiment_id)
    source_dataset = "other" if state == "foreign" else "dataset"
    label_fields = ["label"] if state == "label-tainted" else []
    identity = record_provenance(
        run.info.run_id,
        {
            "dataset_digest": source_dataset,
            "split_digest": "split",
            "label_fields_consumed": label_fields,
        },
        components={"encoder": "FrozenDigitClassifier.encoder"},
    )
    buffer = io.BytesIO()
    torch.save(FrozenDigitClassifier().encoder.state_dict(), buffer)
    reference = save_artifact(buffer.getvalue(), run_id=run.info.run_id, name="encoder")
    client.set_terminated(run.info.run_id, "FINISHED")

    expected_identity = identity
    if state == "failed":
        client.set_terminated(run.info.run_id, "FAILED")
    elif state == "deleted":
        client.delete_run(run.info.run_id)
    elif state == "corrupted":
        artifact_root = Path(unquote(urlparse(run.info.artifact_uri).path))
        (artifact_root / reference.path).write_bytes(b"corrupt")
    else:
        expected_identity = "f" * 64

    with pytest.raises((TrackingError, ArtifactIntegrityError, ValueError)):
        _verified_encoder(
            reference.model_dump(mode="json"),
            identity=expected_identity,
            dataset_digest="dataset",
            split_digest="split",
        )


def test_digit_prediction_rejects_invalid_logits_and_fractional_classes() -> None:
    from reference_projects.kaggle.digit_recognizer.components import (
        DigitPrediction,
        validate_digit_prediction,
    )

    for logits in (
        torch.zeros(2, 9),
        torch.full((2, 10), torch.nan),
        torch.full((2, 10), torch.inf),
    ):
        with pytest.raises(ValueError, match="finite.*batch, 10"):
            DigitPrediction()(logits)
    with pytest.raises(ValueError, match="int64"):
        validate_digit_prediction({"prediction": torch.tensor([1.5])})


def test_digit_flow_verifies_label_free_encoder_then_trains_a_frozen_classifier(
    digit_csvs: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    from lightning import Trainer
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.digit_recognizer.components import FrozenDigitClassifier
    from reference_projects.kaggle.digit_recognizer.flow import digit_recognizer_flow

    observed: list[tuple[type[object], list[bool], list[str], list[str], frozenset[str]]] = []
    original = Trainer.fit

    def record(trainer: Trainer, model: object, *args: object, **kwargs: object) -> object:
        assert type(model) is DsioModule
        datamodule = kwargs["datamodule"]
        assert type(datamodule) is DsioDataModule
        encoder_before = (
            [parameter.detach().clone() for parameter in model.model.encoder.parameters()]
            if isinstance(model.model, FrozenDigitClassifier)
            else []
        )
        result = original(trainer, model, *args, **kwargs)
        if encoder_before:
            assert isinstance(model.model, FrozenDigitClassifier)
            for before, after in zip(encoder_before, model.model.encoder.parameters(), strict=True):
                torch.testing.assert_close(before, after)
        train_batches = list(datamodule.train_dataloader())
        validate_batches = list(datamodule.val_dataloader())
        observed.append(
            (
                type(model.model),
                [parameter.requires_grad for parameter in model.model.parameters()],
                sorted(item for batch in train_batches for item in batch["sample_id"]),
                sorted(item for batch in validate_batches for item in batch["sample_id"]),
                frozenset().union(*(batch.keys() for batch in train_batches)),
            )
        )
        return result

    monkeypatch.setattr(Trainer, "fit", record)
    with prefect_test_harness():
        result = digit_recognizer_flow(str(digit_csvs), str(tmp_path / "work"), seed=13)
        replay = digit_recognizer_flow(str(digit_csvs), str(tmp_path / "work"), seed=13)

    assert len(observed) == 4
    for position in (1, 3):
        assert observed[position][0] is FrozenDigitClassifier
        assert observed[position][1][:4] == [False, False, False, False]
        assert any(observed[position][1][4:])
        assert observed[position][4] == {"sample_id", "x", "y"}
    for position in (0, 2):
        assert observed[position][4] == {"sample_id", "x"}
    assert result["pretraining_label_fields"] == []
    assert len(result["encoder_digest"]) == 64
    assert result["encoder_verified"] is True
    assert_replay_run_ids_differ(result, replay)
    assert result["split_digest"] == replay["split_digest"]
    assert result["encoder_digest"] == replay["encoder_digest"]
    assert result["identities"] == replay["identities"]
    assert result["metrics"] == replay["metrics"]
    assert result["prediction"] == replay["prediction"]
    assert result["submission_bytes"] == replay["submission_bytes"]
    assert len(result["prediction"]) == 4
    assert all(0 <= value <= 9 for value in result["prediction"])
    lines = result["submission_bytes"].decode().splitlines()
    assert lines[0] == "ImageId,Label"
    assert [line.split(",")[0] for line in lines[1:]] == ["1", "2", "3", "4"]
    assert result["submission_digest"] == sha256_of_bytes(result["submission_bytes"])
    expected_membership = (
        sorted(result["assignments"]["train"]),
        sorted(result["assignments"]["validate"]),
    )
    assert [(entry[2], entry[3]) for entry in observed] == [expected_membership] * 4
    assert_execution_evidence(
        result["pretrain_run_id"],
        tmp_path / "digit-pretrain-provenance",
        optimizer="torch.optim:Adam",
        optimizer_parameters={"lr": 0.001},
    )
    assert_execution_evidence(
        result["train_run_id"],
        tmp_path / "digit-train-provenance",
        optimizer="torch.optim:Adam",
        optimizer_parameters={"lr": 0.002},
    )
    client = MlflowClient()
    provenance_path = client.download_artifacts(
        result["train_run_id"], "provenance.json", str(tmp_path / "digit-lineage")
    )
    training = json.loads(Path(provenance_path).read_text())["configuration"]
    assert training["encoder_digest"] == result["encoder_digest"]
    assert training["encoder_source_identity"] == result["identities"]["pretrain"]
    encoder_path = client.download_artifacts(
        result["train_run_id"], "inputs/encoder.json", str(tmp_path / "digit-lineage-input")
    )
    encoder = json.loads(Path(encoder_path).read_text())
    assert encoder["run_id"] == result["pretrain_run_id"]
    assert encoder["digest"] == result["encoder_digest"]
    assert encoder["path"].endswith("/artifact.bin")
    assert_downstream_evidence(result, tmp_path / "digit-downstream")
