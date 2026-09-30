"""A diverging replay must say which provenance field moved, not just print two hashes."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.replay import assert_same_identities


def _run_with_provenance(config: dict[str, object]) -> tuple[str, str]:
    from mlflow import MlflowClient

    from dsio.tracking import record_provenance

    client = MlflowClient()
    experiment = client.get_experiment_by_name("replay-diff")
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment("replay-diff")
    )
    run = client.create_run(experiment_id)
    identity = record_provenance(run.info.run_id, config)
    client.set_terminated(run.info.run_id, "FINISHED")
    return run.info.run_id, identity


def test_identical_identities_pass() -> None:
    run_id, identity = _run_with_provenance({"seed": 7})
    result = {"identities": {"train": identity}, "train_run_id": run_id}
    assert_same_identities(result, dict(result))


def test_mismatch_reports_the_changed_provenance_field() -> None:
    first_run, first_identity = _run_with_provenance({"checkpoint_digest": "aaa", "seed": 7})
    second_run, second_identity = _run_with_provenance({"checkpoint_digest": "bbb", "seed": 7})
    first = {"identities": {"export": first_identity}, "export_run_id": first_run}
    second = {"identities": {"export": second_identity}, "export_run_id": second_run}

    with pytest.raises(AssertionError) as failure:
        assert_same_identities(first, second)

    message = str(failure.value)
    assert f"export: {first_identity} != {second_identity}" in message
    assert "configuration.checkpoint_digest: 'aaa' != 'bbb'" in message
    assert "configuration.seed" not in message


def test_mismatch_names_files_that_dirtied_the_checkout(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    first_run, first_identity = _run_with_provenance({"seed": 7})
    (git_repo / "stray-export-output.txt").write_text("written by a flow\n")
    second_run, second_identity = _run_with_provenance({"seed": 7})
    first = {"identities": {"export": first_identity}, "export_run_id": first_run}
    second = {"identities": {"export": second_identity}, "export_run_id": second_run}

    with pytest.raises(AssertionError) as failure:
        assert_same_identities(first, second)

    message = str(failure.value)
    assert "execution.git.code_hash" in message
    assert "git.patch files (first): []" in message
    assert "stray-export-output.txt" in message


def test_missing_run_ids_still_report_the_stage() -> None:
    with pytest.raises(AssertionError, match="train: a != b"):
        assert_same_identities({"identities": {"train": "a"}}, {"identities": {"train": "b"}})


def _train_and_export(weight: float) -> dict[str, object]:
    import io

    import torch
    from mlflow import MlflowClient

    from dsio.train.artifacts import save_artifact

    client = MlflowClient()
    experiment = client.get_experiment_by_name("replay-diff")
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment("replay-diff")
    )
    train = client.create_run(experiment_id)
    buffer = io.BytesIO()
    torch.save({"state_dict": {"w": torch.tensor([1.0, weight])}, "epoch": 1}, buffer)
    reference = save_artifact(buffer.getvalue(), run_id=train.info.run_id, name="checkpoint")
    client.log_dict(train.info.run_id, reference.model_dump(mode="json"), "outputs/checkpoint.json")
    client.set_terminated(train.info.run_id, "FINISHED")
    export_run, export_identity = _run_with_provenance({"checkpoint_digest": reference.digest})
    return {
        "identities": {"export": export_identity},
        "export_run_id": export_run,
        "train_run_id": train.info.run_id,
    }


def test_checkpoint_digest_mismatch_reports_weight_drift() -> None:
    first = _train_and_export(2.0)
    second = _train_and_export(2.5)

    with pytest.raises(AssertionError) as failure:
        assert_same_identities(first, second)

    message = str(failure.value)
    assert "configuration.checkpoint_digest" in message
    assert "checkpoint state_dict.w: max abs diff 5.000e-01" in message
    assert "checkpoint epoch" not in message
