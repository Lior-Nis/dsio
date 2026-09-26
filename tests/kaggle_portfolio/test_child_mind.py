from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from tests.kaggle_portfolio.assertions import (
    assert_downstream_evidence,
    assert_execution_evidence,
    assert_replay_run_ids_differ,
)

from dsio.contracts import sha256_of_bytes
from dsio.data.loading import DsioDataModule
from dsio.model.module import DsioModule


def test_child_mind_boundary_preserves_optional_modalities_and_excludes_leakage(
    child_mind_data: Path,
) -> None:
    from reference_projects.kaggle.child_mind.data import (
        PACKED_FEATURES,
        SAFE_FEATURE_COLUMNS,
        load_competition_data,
        pack_participant,
        summarize_partition,
    )

    loaded = load_competition_data(child_mind_data)

    assert len(loaded["train"]) == 48
    assert loaded["dropped_missing_targets"] == 4
    assert loaded["test_ids"] == [f"test-{index:03d}" for index in range(4)]
    assert not any(field.startswith("PCIAT-") or field == "sii" for field in SAFE_FEATURE_COLUMNS)

    with_series = loaded["train"][0]
    without_series = loaded["train"][4]
    summary = summarize_partition(Path(with_series["series_path"]), str(with_series["id"]))
    assert np.array_equal(
        summary.values,
        summarize_partition(Path(with_series["series_path"]), str(with_series["id"])).values,
    )
    present = pack_participant(with_series, summary)
    absent = pack_participant(without_series, None)
    assert present.shape == absent.shape == (1, PACKED_FEATURES)
    assert present[0, -1] == 1.0
    assert absent[0, -1] == 0.0
    assert np.isfinite(present).all()
    assert np.isfinite(absent).all()


def test_child_mind_boundary_rejects_finite_values_outside_float32_range(
    child_mind_data: Path,
) -> None:
    from reference_projects.kaggle.child_mind.data import (
        load_competition_data,
        pack_participant,
    )

    row = dict(load_competition_data(child_mind_data)["train"][0])
    row["Basic_Demos-Age"] = "3.5e38"

    with pytest.raises(ValueError, match="train-000.*Basic_Demos-Age.*float32"):
        pack_participant(row, None)


def test_child_mind_boundary_rejects_schema_drift_and_empty_sensor_partition(
    child_mind_data: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.child_mind.data import (
        SENSOR_PARQUET_COLUMNS,
        load_competition_data,
        summarize_partition,
    )

    empty_id = "train-000"
    partition = child_mind_data / "series_train.parquet" / f"id={empty_id}" / "part-0.parquet"
    pq.write_table(
        pa.table(
            {
                name: pa.array([], type=pa.type_for_alias(field_type))
                for name, field_type in SENSOR_PARQUET_COLUMNS
            }
        ),
        partition,
    )
    with pytest.raises(ValueError, match=f"{empty_id}.*empty"):
        summarize_partition(partition, empty_id)

    path = child_mind_data / "test.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    write_fields = list(rows[0])[:-1]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=write_fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in write_fields} for row in rows])
    with pytest.raises(ValueError, match="test.csv columns"):
        load_competition_data(child_mind_data)


def test_child_mind_sensor_boundary_requires_exact_arrow_types(
    child_mind_data: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.child_mind.data import (
        SENSOR_PARQUET_COLUMNS,
        summarize_partition,
    )

    participant_id = "train-000"
    partition = (
        child_mind_data
        / "series_train.parquet"
        / f"id={participant_id}"
        / "part-0.parquet"
    )
    columns = {
        name: pa.array(
            [0, 1],
            type=pa.float64() if name == "X" else pa.type_for_alias(field_type),
        )
        for name, field_type in SENSOR_PARQUET_COLUMNS
    }
    pq.write_table(pa.table(columns), partition)

    with pytest.raises(ValueError, match=f"{participant_id}.*X.*float32"):
        summarize_partition(partition, participant_id)


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([0, None, 2], "non-null"),
        ([0, 2, 1], "strictly increasing"),
        ([*range(65_536), 65_535], "strictly increasing"),
    ],
)
def test_child_mind_sensor_boundary_requires_valid_steps(
    child_mind_data: Path,
    steps: list[int | None],
    message: str,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.child_mind.data import (
        SENSOR_PARQUET_COLUMNS,
        summarize_partition,
    )

    participant_id = "train-000"
    partition = (
        child_mind_data
        / "series_train.parquet"
        / f"id={participant_id}"
        / "part-0.parquet"
    )
    columns = {
        name: pa.array(
            steps if name == "step" else [0] * len(steps),
            type=pa.type_for_alias(field_type),
        )
        for name, field_type in SENSOR_PARQUET_COLUMNS
    }
    pq.write_table(pa.table(columns), partition)

    with pytest.raises(ValueError, match=f"{participant_id}.*step.*{message}"):
        summarize_partition(partition, participant_id)


def test_child_mind_sensor_summary_has_stable_variance_and_finite_float32_output(
    child_mind_data: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.child_mind.data import (
        SENSOR_COLUMNS,
        summarize_partition,
    )

    participant_id = "train-000"
    partition = (
        child_mind_data
        / "series_train.parquet"
        / f"id={participant_id}"
        / "part-0.parquet"
    )
    table = pq.ParquetFile(partition).read()
    time_index = table.schema.get_field_index("time_of_day")
    base = 1_000_000_000_000
    table = table.set_column(
        time_index,
        "time_of_day",
        pa.array([base + offset for offset in (0, 2, 4, 6, 8)], type=pa.int64()),
    )
    float_index = table.schema.get_field_index("X")
    limit = np.finfo(np.float32).max
    table = table.set_column(
        float_index,
        "X",
        pa.array([limit, -limit, limit, -limit, 0], type=pa.float32()),
    )
    pq.write_table(table, partition)

    summary = summarize_partition(partition, participant_id)
    time_offset = 1 + SENSOR_COLUMNS.index("time_of_day") * 5
    assert summary.values[time_offset + 1] == pytest.approx(np.sqrt(8), rel=1e-6)
    assert np.isfinite(summary.values).all()


def test_child_mind_sensor_summary_rejects_infinite_source_values(
    child_mind_data: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.child_mind.data import summarize_partition

    participant_id = "train-000"
    partition = (
        child_mind_data
        / "series_train.parquet"
        / f"id={participant_id}"
        / "part-0.parquet"
    )
    table = pq.ParquetFile(partition).read()
    field_index = table.schema.get_field_index("X")
    table = table.set_column(
        field_index,
        "X",
        pa.array([0, 1, np.inf, 3, 4], type=pa.float32()),
    )
    pq.write_table(table, partition)

    with pytest.raises(ValueError, match=f"{participant_id}.*X.*infinity"):
        summarize_partition(partition, participant_id)


def test_child_mind_fusion_keeps_missing_sensor_explicit_and_ablation_local() -> None:
    from reference_projects.kaggle.child_mind.components import (
        CmiFusionClassifier,
        CmiPrediction,
        ablate_sensor,
        validate_cmi_prediction,
    )
    from reference_projects.kaggle.child_mind.data import (
        PACKED_FEATURES,
        SENSOR_FEATURES,
        SENSOR_MASK_SLICE,
        SENSOR_PRESENT_INDEX,
        SENSOR_VALUE_SLICE,
        TABULAR_FEATURES,
    )

    batch = torch.zeros(2, 1, PACKED_FEATURES)
    batch[:, 0, :TABULAR_FEATURES] = torch.arange(TABULAR_FEATURES)
    batch[:, 0, TABULAR_FEATURES : 2 * TABULAR_FEATURES] = 1
    batch[0, 0, SENSOR_VALUE_SLICE] = 2
    batch[0, 0, SENSOR_MASK_SLICE] = 1
    batch[0, 0, SENSOR_PRESENT_INDEX] = 1
    batch[1, 0, SENSOR_VALUE_SLICE] = 999

    model = CmiFusionClassifier(
        tabular_center=[0.0] * TABULAR_FEATURES,
        tabular_scale=[1.0] * TABULAR_FEATURES,
        sensor_center=[0.0] * SENSOR_FEATURES,
        sensor_scale=[1.0] * SENSOR_FEATURES,
        use_sensor=True,
        hidden=8,
    )
    logits = model(batch)
    outputs = CmiPrediction()(logits)
    validate_cmi_prediction(outputs)
    assert logits.shape == (2, 4)
    assert outputs["prediction"].shape == (2,)
    assert outputs["probability"].shape == (2, 4)

    ablated = ablate_sensor(batch)
    assert torch.equal(
        ablated[..., : SENSOR_VALUE_SLICE.start],
        batch[..., : SENSOR_VALUE_SLICE.start],
    )
    assert ablated[..., SENSOR_VALUE_SLICE].eq(0).all()
    assert ablated[..., SENSOR_MASK_SLICE].eq(0).all()
    assert ablated[..., SENSOR_PRESENT_INDEX].eq(0).all()

    tabular_only = CmiFusionClassifier(
        tabular_center=[0.0] * TABULAR_FEATURES,
        tabular_scale=[1.0] * TABULAR_FEATURES,
        sensor_center=[0.0] * SENSOR_FEATURES,
        sensor_scale=[1.0] * SENSOR_FEATURES,
        use_sensor=False,
        hidden=8,
    )
    assert torch.equal(tabular_only(batch), tabular_only(ablated))


def test_child_mind_fusion_ignores_masked_nonfinite_values_but_rejects_observed_ones() -> None:
    from reference_projects.kaggle.child_mind.components import CmiFusionClassifier
    from reference_projects.kaggle.child_mind.data import (
        PACKED_FEATURES,
        SENSOR_VALUE_SLICE,
        TABULAR_FEATURES,
    )

    model = CmiFusionClassifier(
        tabular_center=[0.0] * TABULAR_FEATURES,
        tabular_scale=[1.0] * TABULAR_FEATURES,
        sensor_center=[0.0] * (SENSOR_VALUE_SLICE.stop - SENSOR_VALUE_SLICE.start),
        sensor_scale=[1.0] * (SENSOR_VALUE_SLICE.stop - SENSOR_VALUE_SLICE.start),
        hidden=8,
    )
    clean = torch.zeros(1, 1, PACKED_FEATURES)
    masked = clean.clone()
    masked[0, 0, 0] = torch.nan
    masked[0, 0, SENSOR_VALUE_SLICE.start] = torch.inf

    assert torch.equal(model(masked), model(clean))

    observed = masked.clone()
    observed[0, 0, TABULAR_FEATURES] = 1
    with pytest.raises(ValueError, match="observed tabular.*finite"):
        model(observed)


@pytest.mark.parametrize(
    ("prediction", "probability", "message"),
    [
        (torch.empty(0, dtype=torch.long), torch.empty(0, 4), "non-empty"),
        (torch.tensor([1.0]), torch.tensor([[0.0, 1.0, 0.0, 0.0]]), "integer"),
        (
            torch.tensor([1]),
            torch.tensor([[-0.1, 1.1, 0.0, 0.0]]),
            r"\[0, 1\]",
        ),
    ],
)
def test_child_mind_prediction_validation_rejects_invalid_output_values(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    message: str,
) -> None:
    from reference_projects.kaggle.child_mind.components import validate_cmi_prediction

    with pytest.raises(ValueError, match=message):
        validate_cmi_prediction(
            {"prediction": prediction, "probability": probability}
        )


def test_child_mind_flow_trains_tabular_and_fused_models_with_replayable_evidence(
    child_mind_data: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    from lightning import Trainer
    from mlflow import MlflowClient
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.child_mind.flow import child_mind_flow

    observed: list[tuple[type[object], type[object]]] = []
    original = Trainer.fit

    def record(trainer: Trainer, model: object, *args: object, **kwargs: object) -> object:
        observed.append((type(model), type(kwargs["datamodule"])))
        return original(trainer, model, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", record)
    with prefect_test_harness():
        result = child_mind_flow(str(child_mind_data), str(tmp_path / "work"), seed=17)
        replay = child_mind_flow(str(child_mind_data), str(tmp_path / "work"), seed=17)

    assert result["dataset_digest"] == replay["dataset_digest"]
    assert result["split_digest"] == replay["split_digest"]
    assert result["assignments"] == replay["assignments"]
    assert result["labelled_participants"] == 48
    assert result["dropped_missing_targets"] == 4
    assert result["train_sensor_participants"] == 24
    assert set(result["assignments"]["train"]).isdisjoint(result["assignments"]["validate"])
    assert observed == [(DsioModule, DsioDataModule)] * 4

    for mode in ("tabular", "fused"):
        model = result["models"][mode]
        replay_model = replay["models"][mode]
        assert_replay_run_ids_differ(model, replay_model)
        assert model["identities"] == replay_model["identities"]
        assert model["metrics"] == replay_model["metrics"]
        assert model["prediction"] == replay_model["prediction"]
        assert model["submission_bytes"] == replay_model["submission_bytes"]
        assert model["prediction_count"] == 4
        assert set(model["metrics"]) == {
            "accuracy",
            "quadratic_weighted_kappa",
            "qwk.sensor_present",
            "qwk.sensor_missing",
            "qwk.sensor_ablated",
            "qwk.sensor_ablation_delta",
        }
        assert all(np.isfinite(value) for value in model["metrics"].values())
        assert model["ablation_sample_ids"] == model["sensor_present_sample_ids"]
        assert model["submission_digest"] == sha256_of_bytes(model["submission_bytes"])
        lines = model["submission_bytes"].decode().splitlines()
        assert lines[0] == "id,sii"
        assert [line.split(",", 1)[0] for line in lines[1:]] == [
            f"test-{index:03d}" for index in range(4)
        ]
        assert all(int(line.rsplit(",", 1)[1]) in range(4) for line in lines[1:])
        evaluation = MlflowClient().get_run(model["evaluation_run_id"])
        for name, value in model["metrics"].items():
            assert evaluation.data.metrics[name] == pytest.approx(value)
        assert_execution_evidence(
            model["train_run_id"],
            tmp_path / f"{mode}-provenance",
            optimizer="torch.optim:AdamW",
            optimizer_parameters={"lr": 0.003, "weight_decay": 0.0001},
            batch_size=16,
            num_workers=0,
        )
        assert_downstream_evidence(model, tmp_path / f"{mode}-downstream")
