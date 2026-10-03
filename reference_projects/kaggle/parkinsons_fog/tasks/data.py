"""Bounded window staging and subject-disjoint split generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mlflow import MlflowClient
from prefect import task

from dsio.data.adapters import TableExamples
from dsio.data.splits import generate
from dsio.data.store import SignalStore
from dsio.experimental.telemetry import log_phase_evidence, measure_phase
from dsio.tracking import attempt, record_provenance, record_split_evidence
from reference_projects.kaggle.parkinsons_fog.components import CHANNEL_COUNT
from reference_projects.kaggle.parkinsons_fog.data import (
    WINDOW_SIZE,
    iter_recording_windows,
    load_competition_data,
)
from reference_projects.kaggle.parkinsons_fog.scale import (
    resolve_official_inventory,
    scan_non_supervised,
)

SPLIT_PARAMETERS = {"test_size": 0.34}


def labelled_examples(store: SignalStore) -> TableExamples:
    labelled = [entity for entity in store.entities if entity.attrs["source"] == "train"]
    return TableExamples(
        name=store.path.name,
        sample_ids=[entity.entity_id for entity in labelled],
        groups=[entity.group for entity in labelled],
        digest=store.identity,
    )


@task(persist_result=False)
def scan_scale_sources(
    metadata_root: str,
    labelled_root: str,
    daily_root: str,
    experiment_id: str,
    *,
    batch_rows: int = 65_536,
    memory_budget_bytes: int = 4 * 1024**3,
) -> dict[str, Any]:
    """Resolve completeness first, then time a bounded read of non-supervised sources."""
    if (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, int)
        or memory_budget_bytes < 1
    ):
        raise ValueError(
            f"memory_budget_bytes must be a positive integer, got {memory_budget_bytes!r}"
        )
    with attempt(experiment_id) as run:
        inventory = resolve_official_inventory(metadata_root, labelled_root, daily_root)
        with measure_phase("scan") as telemetry:
            scan = scan_non_supervised(inventory, batch_rows=batch_rows)
        elapsed = float(telemetry["elapsed_seconds"])
        measurements: dict[str, int | float] = {
            "files": scan["files"],
            "rows": scan["rows"],
            "source_bytes": scan["source_bytes"],
            "max_batch_rows": scan["max_batch_rows"],
            "rows_per_second": scan["rows"] / elapsed,
            "bytes_per_second": scan["source_bytes"] / elapsed,
        }
        identity = record_provenance(
            run.info.run_id,
            {
                "manifest_digest": inventory["manifest_digest"],
                "scan_checksum": scan["checksum"],
                "batch_rows": batch_rows,
                "memory_budget_bytes": memory_budget_bytes,
                "lanes": scan["lanes"],
            },
            components={
                "inventory": (
                    "reference_projects.kaggle.parkinsons_fog.scale.inventory:"
                    "resolve_official_inventory"
                ),
                "scanner": (
                    "reference_projects.kaggle.parkinsons_fog.scale.scanning:scan_non_supervised"
                ),
            },
        )
        client = MlflowClient()
        client.log_dict(run.info.run_id, inventory, "outputs/source-inventory.json")
        log_phase_evidence(run.info.run_id, "scan", telemetry, measurements)
        if int(telemetry["peak_process_tree_rss_bytes"]) >= memory_budget_bytes:
            raise RuntimeError(
                "Parkinson scale scan exceeded its process-tree RSS budget: "
                f"{telemetry['peak_process_tree_rss_bytes']} >= {memory_budget_bytes}"
            )
        return {
            "scan_run_id": run.info.run_id,
            "inventory": inventory,
            "manifest_digest": inventory["manifest_digest"],
            "scan": scan,
            "telemetry": telemetry,
            "identity": identity,
        }


@task(persist_result=False)
def ingest(
    data_dir: str,
    workspace: str,
    experiment_id: str,
    source_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        telemetry_context = (
            measure_phase("ingest") if source_inventory is not None else _unmeasured_phase()
        )
        with telemetry_context as telemetry:
            loaded = load_competition_data(data_dir, source_inventory=source_inventory)
            excluded = set(loaded["excluded_train_ids"])
            path = Path(workspace) / run.info.run_id / "parkinsons-fog"
            path.parent.mkdir(parents=True, exist_ok=True)
            test_sample_ids: list[str] = []
            ignored_points = 0
            staged_points = 0
            dropped_windows = 0
            with SignalStore.builder(path, channels=CHANNEL_COUNT, dtype="float32") as builder:
                for source in ("train", "test"):
                    for recording in loaded[source]:
                        recording_id = str(recording["recording_id"])
                        if source == "train" and recording_id in excluded:
                            continue
                        for window in iter_recording_windows(recording):
                            values = np.asarray(window["data"], dtype=np.float32)
                            valid = values[:, -1].astype(bool)
                            ignored_points += int((~valid).sum()) if source == "train" else 0
                            if source == "train" and not bool(valid.any()):
                                dropped_windows += 1
                                continue
                            start = int(window["start"])
                            sample_id = f"{source}:{recording['kind']}:{recording_id}:{start}"
                            attrs: dict[str, Any] = {
                                "source": source,
                                "kind": str(recording["kind"]),
                                "recording_id": recording_id,
                                "subject": str(recording["subject"]),
                                "start": start,
                            }
                            if source == "test":
                                attrs["submission_ids"] = [
                                    f"{recording_id}_{time}" for time in window["times"]
                                ]
                                test_sample_ids.append(sample_id)
                            builder.add(
                                sample_id,
                                values,
                                group=str(recording["subject"]),
                                attrs=attrs,
                            )
                            staged_points += len(values)
            store = SignalStore(path)
            store.verify()
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "window_size": WINDOW_SIZE,
                "excluded_test_subjects": loaded["test_subjects"],
                "excluded_train_recordings": sorted(excluded),
                "ignored_points": ignored_points,
                "dropped_windows": dropped_windows,
                "staged_points": staged_points,
                "schema": "kaggle-parkinsons-fog-v2",
                **(
                    {"source_manifest_digest": source_inventory["manifest_digest"]}
                    if source_inventory is not None
                    else {}
                ),
            },
            components={
                "loader": ("reference_projects.kaggle.parkinsons_fog.data:iter_recording_windows")
            },
        )
        MlflowClient().log_dict(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "test_sample_ids": test_sample_ids,
                "submission_ids": loaded["submission_ids"],
            },
            "outputs/dataset.json",
        )
        if source_inventory is not None:
            source_items = [
                *source_inventory["train_defog"],
                *source_inventory["train_tdcsfog"],
                *source_inventory["test"],
            ]
            elapsed = float(telemetry["elapsed_seconds"])
            log_phase_evidence(
                run.info.run_id,
                "ingest",
                telemetry,
                {
                    "files": len(source_items),
                    "source_bytes": sum(int(item["bytes"]) for item in source_items),
                    "staged_points": staged_points,
                    "staged_windows": len(store.entities),
                    "dropped_windows": dropped_windows,
                    "ignored_points": ignored_points,
                    "points_per_second": staged_points / elapsed,
                },
            )
        return {
            "store_path": str(path),
            "dataset_digest": store.identity,
            "data_run_id": run.info.run_id,
            "identity": identity,
            "test_sample_ids": test_sample_ids,
            "submission_ids": loaded["submission_ids"],
            "ignored_points": ignored_points,
            "dropped_windows": dropped_windows,
            "staged_points": staged_points,
            "telemetry": telemetry or None,
            "source_manifest_digest": (
                source_inventory["manifest_digest"] if source_inventory is not None else None
            ),
        }


def _unmeasured_phase() -> Any:
    from contextlib import nullcontext

    return nullcontext({})


@task(persist_result=False)
def split_data(data: dict[str, Any], experiment_id: str, seed: int) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = generate(
            examples,
            "group_shuffle",
            name="parkinsons-subject-holdout",
            seed=seed,
            roles=("train", "validate"),
            parameters=SPLIT_PARAMETERS,
        )
        fold = manifest.fold(0)
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "algorithm": "group_shuffle",
                "parameters": SPLIT_PARAMETERS,
                "seed": seed,
            },
            components={"splitter": "dsio.data.splits.generate:generate"},
        )
        uri = record_split_evidence(run.info.run_id, examples, manifest, source=store.path.name)
        return {
            "split_run_id": run.info.run_id,
            "split_uri": uri,
            "split_digest": manifest.digest,
            "assignments": fold.assignments,
            "split_groups": fold.parts,
            "identity": identity,
        }
