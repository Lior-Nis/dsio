"""Ledger, provenance and determinism invariants."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dsio.config import RunConfig
from dsio.contracts import NonCanonicalValueError, sha256_of
from dsio.runs import RunLedger, RunStatus, capture_env, capture_git, seed_everything
from dsio.runs.record import CONFIG_FILE, PATCH_FILE, REPRODUCE_FILE, RUN_FILE
from dsio.train import execute, load_runners


def _start(ledger: RunLedger, config: RunConfig, **kwargs: object) -> object:
    return ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
        **kwargs,  # type: ignore[arg-type]
    )


def test_record_is_written_before_any_work(ledger: RunLedger, config: RunConfig) -> None:
    """A run that crashes immediately must still have an identity on disk."""
    run = _start(ledger, config)
    assert (run.dir / RUN_FILE).is_file()
    assert (run.dir / CONFIG_FILE).is_file()
    assert (run.dir / REPRODUCE_FILE).is_file()


def test_identical_configs_get_distinct_run_ids(ledger: RunLedger, config: RunConfig) -> None:
    """Run ids carry a one-second timestamp, so a fast rerun must not collide."""
    ids = {_start(ledger, config).run_id for _ in range(3)}  # type: ignore[attr-defined]
    assert len(ids) == 3


def test_failed_run_is_recorded_not_lost(ledger: RunLedger, config: RunConfig) -> None:
    with pytest.raises(RuntimeError), _start(ledger, config) as run:
        raise RuntimeError("boom")
    assert ledger.load(run.run_id).record.status is RunStatus.FAILED
    assert "boom" in (ledger.load(run.run_id).record.error or "")


def test_non_finite_metrics_are_not_written(ledger: RunLedger, config: RunConfig) -> None:
    """A NaN in the stream breaks every downstream comparison."""
    run = _start(ledger, config)
    run.log_metrics({"good": 1.0, "bad": float("nan"), "worse": float("inf")})
    rows = run.read_metrics()
    assert rows and "good" in rows[0]
    assert "bad" not in rows[0] and "worse" not in rows[0]


def test_non_finite_values_cannot_be_hashed() -> None:
    """Two different configs must never collide onto one cache key via NaN."""
    with pytest.raises(NonCanonicalValueError):
        sha256_of({"lr": float("nan")})


def test_git_provenance_on_a_clean_tree(git_repo: Path) -> None:
    state = capture_git(cwd=git_repo)
    assert state.sha is not None
    assert state.dirty is False
    assert state.code_hash == state.sha


def test_dirty_tree_is_allowed_and_still_reconstructible(git_repo: Path) -> None:
    """Never block; capture enough that the run reproduces anyway."""
    (git_repo / "tracked.txt").write_text("modified\n")
    (git_repo / "untracked.txt").write_text("new file\n")

    state = capture_git(cwd=git_repo)
    assert state.dirty is True
    assert state.code_hash is not None
    assert state.code_hash.startswith(f"{state.sha}-dirty-")
    assert state.patch_sha256 is not None


def test_dirty_hash_tracks_content_not_just_filenames(git_repo: Path) -> None:
    """Editing a file's contents must change the code hash, not only touching it."""
    (git_repo / "tracked.txt").write_text("first change\n")
    first = capture_git(cwd=git_repo).code_hash
    (git_repo / "tracked.txt").write_text("second change\n")
    second = capture_git(cwd=git_repo).code_hash
    assert first != second


def test_missing_git_yields_none_not_a_wrong_stamp(tmp_path: Path) -> None:
    """A confident-but-wrong provenance value is worse than a missing one."""
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    state = capture_git(cwd=plain)
    assert state.sha is None
    assert state.code_hash is None
    assert state.available is False


def test_patch_file_is_written_for_dirty_runs(
    git_repo: Path, config: RunConfig, ledger: RunLedger
) -> None:
    (git_repo / "tracked.txt").write_text("dirty\n")
    run = _start(ledger, config, repo_root=git_repo)
    assert (run.dir / PATCH_FILE).is_file()
    assert b"dirty" in (run.dir / PATCH_FILE).read_bytes()


def test_reproduce_script_pins_the_commit(
    git_repo: Path, config: RunConfig, ledger: RunLedger
) -> None:
    run = _start(ledger, config, repo_root=git_repo, command=("dsio", "run", "x"))
    script = (run.dir / REPRODUCE_FILE).read_text()
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=git_repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert f"git checkout {sha}" in script
    assert "uv sync --locked" in script
    assert "dsio run x" in script


def test_env_capture_records_the_lockfile() -> None:
    env = capture_env(lock_path=Path("uv.lock"))
    assert env.python
    assert env.lock_sha256 is not None, "uv.lock should be hashed for reproducibility"


def test_same_seed_gives_identical_metrics(ledger: RunLedger, config: RunConfig) -> None:
    load_runners()
    results = []
    for _ in range(2):
        seed_everything(config.seed)
        with _start(ledger, config) as run:
            results.append(execute(config, run))
    assert results[0] == results[1]


def test_different_seed_gives_different_metrics(ledger: RunLedger, config: RunConfig) -> None:
    """Guards against a seed that is recorded but never actually wired through."""
    load_runners()
    outcomes = []
    for seed in (1, 999):
        variant = config.model_copy(update={"seed": seed})
        seed_everything(seed)
        with _start(ledger, variant) as run:
            outcomes.append(execute(variant, run))
    assert outcomes[0] != outcomes[1]


def test_seed_is_range_checked() -> None:
    with pytest.raises(ValueError, match="seed must be in"):
        seed_everything(-1)
