"""The data, splits and eval command surfaces.

Driven as subprocesses through the real entrypoint, because the thing under test is the
envelope contract as a caller actually sees it — an in-process call would bypass the
argument parsing and the exit code, which is where two of the CLI's past bugs lived.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dsio.config.schema import RunConfig
from dsio.data import SignalStore, entity_examples
from dsio.runs import RunLedger
from dsio.runs.record import RUNS_ROOT_ENV
from dsio.splits import SplitFile
from dsio.train import execute, load_runners


def dsio(*args: str, cwd: Path, env_extra: dict[str, str] | None = None) -> tuple[int, dict]:
    env = {**os.environ, **(env_extra or {})}
    completed = subprocess.run(
        [sys.executable, "-m", "dsio.cli.main", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(completed.stdout) if completed.stdout.strip() else {}
    return completed.returncode, payload


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A scratch project with one store under ``stores/``."""
    root = tmp_path / "project"
    root.mkdir()
    rng = np.random.default_rng(0)
    with SignalStore.builder(root / "stores" / "cohort", channels=3) as builder:
        for group in range(9):
            for session in range(2):
                builder.add(
                    f"p{group}_s{session}",
                    rng.standard_normal((1500, 3)).astype("float32"),
                    group=f"p{group}",
                    attrs={"t_start": 0.0, "sample_rate": 100.0, "events": group * 10},
                )
    return root


# --- dsio data ----------------------------------------------------------------------


def test_data_ls_finds_the_store(workdir: Path) -> None:
    code, payload = dsio("data", "ls", cwd=workdir)
    assert code == 0 and payload["ok"] is True
    assert payload["count"] == 1
    assert payload["stores"][0]["name"] == "cohort"
    assert payload["stores"][0]["groups"] == 9


def test_data_ls_on_an_empty_root_is_success_not_failure(tmp_path: Path) -> None:
    """Nothing to list is an answer, not an error. An agent branching on `ok` needs that."""
    code, payload = dsio("data", "ls", cwd=tmp_path)
    assert code == 0 and payload["ok"] is True and payload["count"] == 0


def test_data_show_projects_by_default(workdir: Path) -> None:
    code, payload = dsio("data", "show", "stores/cohort", cwd=workdir)
    assert code == 0
    assert payload["entities"] == 18
    assert "entity_list" not in payload, "detail must be opt-in"


def test_data_show_can_include_entities(workdir: Path) -> None:
    code, payload = dsio("data", "show", "stores/cohort", "--entities", "--limit", "5", cwd=workdir)
    assert code == 0
    assert len(payload["entity_list"]) == 5
    assert payload["entity_list_truncated"] is True


def test_data_verify_passes_on_an_intact_store(workdir: Path) -> None:
    code, payload = dsio("data", "verify", "stores/cohort", cwd=workdir)
    assert code == 0 and payload["verified"] is True


def test_data_verify_fails_closed_on_corruption(workdir: Path) -> None:
    """Existence is not integrity. A store that silently lost bytes must not read as valid."""
    signal = workdir / "stores" / "cohort" / "signal.bin"
    data = bytearray(signal.read_bytes())
    data[1000:1010] = b"\x00" * 10
    signal.write_bytes(bytes(data))

    code, payload = dsio("data", "verify", "stores/cohort", cwd=workdir)
    assert code == 1
    assert payload["ok"] is False
    assert payload["code"] == "integrity"


def test_data_index_builds_a_view(workdir: Path) -> None:
    code, payload = dsio(
        "data", "index", "stores/cohort", "--length", "500", "--stride", "200", cwd=workdir
    )
    assert code == 0
    assert payload["windows"] == 108
    assert payload["written"] is True
    assert (workdir / payload["path"]).is_file()


def test_data_index_dry_run_writes_nothing(workdir: Path) -> None:
    code, payload = dsio("data", "index", "stores/cohort", "--dry-run", cwd=workdir)
    assert code == 0 and payload["written"] is False
    assert not (workdir / payload["path"]).exists()


# --- dsio data push / pull ----------------------------------------------------------


def test_push_then_pull_round_trips_through_the_cli(workdir: Path, tmp_path: Path) -> None:
    """The fresh-clone path, driven the way a user or an agent would drive it."""
    remote = tmp_path / "remote"
    remote.mkdir()

    code, payload = dsio(
        "data", "push", "stores/cohort", "--remote", str(remote), cwd=workdir
    )
    assert code == 0 and payload["transferred"] == 3

    clone = tmp_path / "clone"
    (clone / "stores" / "cohort").mkdir(parents=True)
    (clone / "stores" / "cohort" / "manifest.yaml").write_bytes(
        (workdir / "stores" / "cohort" / "manifest.yaml").read_bytes()
    )

    code, payload = dsio("data", "pull", "cohort", "--remote", str(remote), cwd=clone)
    assert code == 0 and payload["transferred"] == 3

    code, payload = dsio("data", "verify", "stores/cohort", cwd=clone)
    assert code == 0 and payload["verified"] is True


def test_push_is_idempotent_through_the_cli(workdir: Path, tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    dsio("data", "push", "stores/cohort", "--remote", str(remote), cwd=workdir)
    code, payload = dsio("data", "push", "stores/cohort", "--remote", str(remote), cwd=workdir)
    assert code == 0
    assert payload["transferred"] == 0 and payload["skipped"] == 3
    assert payload["bytes"] == 0


def test_status_reports_before_and_after_a_push(workdir: Path, tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    code, payload = dsio("data", "status", "cohort", "--remote", str(remote), cwd=workdir)
    assert code == 0 and payload["complete"] is True and payload["pushed"] is False

    dsio("data", "push", "stores/cohort", "--remote", str(remote), cwd=workdir)
    _, payload = dsio("data", "status", "cohort", "--remote", str(remote), cwd=workdir)
    assert payload["pushed"] is True


def test_an_unconfigured_remote_is_its_own_error_code(workdir: Path) -> None:
    """`remote` rather than `internal`: it is a setup problem with an obvious fix."""
    code, payload = dsio("data", "status", "cohort", cwd=workdir)
    assert code == 1
    assert payload["code"] == "remote"
    assert payload["retryable"] is False


# --- dsio splits --------------------------------------------------------------------


def _write_k3_splits(workdir: Path) -> None:
    """Split files are a project's own output now; the CLI only reads them.

    Writing the fixture directly (rather than through a removed ``splits make``) is what
    ``show`` and ``check`` below actually exercise: reading and proving a committed split.
    Three hand-picked folds over the store's nine groups, each covering all of them.
    """
    store = SignalStore(workdir / "stores" / "cohort")
    digest = entity_examples(store).digest
    folds = [
        {"test": ["p0", "p1", "p2"], "val": ["p3"], "train": ["p4", "p5", "p6", "p7", "p8"]},
        {"test": ["p3", "p4", "p5"], "val": ["p6"], "train": ["p0", "p1", "p2", "p7", "p8"]},
        {"test": ["p6", "p7", "p8"], "val": ["p0"], "train": ["p1", "p2", "p3", "p4", "p5"]},
    ]
    for fold, parts in enumerate(folds):
        SplitFile(
            store=store.path.name,
            store_manifest_sha256=digest,
            name="k3",
            fold=fold,
            counts={part: len(members) for part, members in parts.items()},
            parts=parts,
        ).save(workdir / "splits" / "k3" / f"fold{fold}.yaml")


def test_splits_check_proves_no_row_overlap(workdir: Path) -> None:
    """The guarantee the whole layer exists for, verified rather than asserted."""
    _write_k3_splits(workdir)
    code, payload = dsio(
        "splits", "check", "stores/cohort", "--name", "k3",
        "--length", "500", "--stride", "200", cwd=workdir,
    )
    assert code == 0
    assert payload["ok"] is True
    assert payload["rows_proved_disjoint"] is True
    assert payload["coverage"] == 1.0
    assert len(payload["per_fold"]) == 3


def test_splits_check_catches_a_hand_edited_overlap(workdir: Path) -> None:
    """Split files are meant to be read and edited by humans, so an edit that breaks the
    guarantee is a realistic failure — and it must be caught before a model is fitted.

    Here a group tested by fold 0 is also given to fold 1's test part. Each file is still
    individually valid, so only comparing them across folds reveals the double-count.
    """
    import yaml

    _write_k3_splits(workdir)
    first_path = workdir / "splits" / "k3" / "fold0.yaml"
    second_path = workdir / "splits" / "k3" / "fold1.yaml"

    borrowed = yaml.safe_load(first_path.read_text())["parts"]["test"][0]
    second = yaml.safe_load(second_path.read_text())
    second["parts"]["train"] = [g for g in second["parts"]["train"] if g != borrowed]
    second["parts"]["val"] = [g for g in second["parts"]["val"] if g != borrowed]
    second["parts"]["test"] = [*second["parts"]["test"], borrowed]
    second_path.write_text(yaml.safe_dump(second))

    code, payload = dsio(
        "splits", "check", "stores/cohort", "--name", "k3",
        "--length", "500", "--stride", "200", cwd=workdir,
    )
    assert code == 1
    assert payload["ok"] is False
    assert payload["code"] == "leakage"
    assert "test part of both" in payload["error"]


def test_splits_show_projects_group_lists_by_default(workdir: Path) -> None:
    _write_k3_splits(workdir)
    code, payload = dsio("splits", "show", "splits/k3/fold0.yaml", cwd=workdir)
    assert code == 0
    assert payload["parts"]["test"] == 3
    assert "group_lists" not in payload


def test_splits_check_says_how_to_make_missing_files(workdir: Path) -> None:
    code, payload = dsio("splits", "check", "stores/cohort", "--name", "nope", cwd=workdir)
    assert code == 1
    assert "commit a split file" in payload["error"]


# --- dsio eval ----------------------------------------------------------------------


@pytest.fixture
def two_runs(tmp_path: Path) -> tuple[Path, str, str]:
    """Two runs over identical folds, so the paired comparison is available."""
    root = tmp_path / "evalproject"
    root.mkdir()
    runs_root = root / "runs"
    os.environ[RUNS_ROOT_ENV] = str(runs_root)
    load_runners()
    from dsio.train.tabular import TabularTask

    ledger = RunLedger(runs_root)
    ids = []
    for estimator in ("logreg", "random_forest"):
        config = RunConfig(
            name=estimator,
            task=TabularTask(
                dataset="breast_cancer",
                estimator=estimator,
                folds=5,
                metrics=("accuracy", "average_precision"),
                keep_model=False,
            ),
        )
        run = ledger.start(
            name=config.name,
            config=config.to_dict(),
            config_hash=config.config_hash,
            seed=config.seed,
        )
        with run:
            run.finish(metrics=execute(config, run))
        ids.append(run.run_id)
    return root, ids[0], ids[1]


def test_eval_show_reports_coverage_and_spread(two_runs) -> None:
    root, first, _ = two_runs
    code, payload = dsio(
        "eval", "show", first, cwd=root, env_extra={RUNS_ROOT_ENV: str(root / "runs")}
    )
    assert code == 0
    assert payload["coverage"] == 1.0
    assert payload["per_fold_std"]["accuracy"] > 0
    assert payload["fold_fingerprint"]


def test_eval_verdict_pairs_identical_folds(two_runs) -> None:
    """Both runs used the same splitter and seed, so the sharper test is available."""
    root, first, second = two_runs
    code, payload = dsio(
        "eval", "verdict", second, "--baseline", first, "--metric", "accuracy",
        cwd=root, env_extra={RUNS_ROOT_ENV: str(root / "runs")},
    )
    assert code == 0
    assert payload["method"] == "paired"
    assert payload["outcome"] in {"win", "neutral", "regression"}
    assert payload["noise_floor"] is not None


def test_eval_rank_orders_runs(two_runs) -> None:
    root, first, _ = two_runs
    code, payload = dsio(
        "eval", "rank", "--metric", "accuracy", "--baseline", first,
        cwd=root, env_extra={RUNS_ROOT_ENV: str(root / "runs")},
    )
    assert code == 0
    assert payload["count"] == 2
    values = [row["value"] for row in payload["runs"]]
    assert values == sorted(values, reverse=True)
    assert "not a claim" in payload["note"]


def test_eval_noise_answers_before_the_experiment(tmp_path: Path) -> None:
    code, payload = dsio("eval", "noise", "--n-rows", "569", "--delta", "0.005", cwd=tmp_path)
    assert code == 0
    assert payload["detectable"] is False
    assert payload["rows_needed"] == 10_000


def test_eval_noise_needs_something_to_compute(tmp_path: Path) -> None:
    code, payload = dsio("eval", "noise", cwd=tmp_path)
    assert code == 1
    assert "--n-rows" in payload["error"]


def test_eval_show_on_a_run_without_a_report_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    ledger = RunLedger(root / "runs")
    run = ledger.start(name="bare", config={}, config_hash="0" * 64, seed=1)
    code, payload = dsio(
        "eval", "show", run.run_id, cwd=tmp_path,
        env_extra={RUNS_ROOT_ENV: str(root / "runs")},
    )
    assert code == 1
    assert payload["ok"] is False
