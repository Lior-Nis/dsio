"""The provenance stamper, MLflow's status guarantee, and determinism invariants.

This file used to be titled "Ledger, provenance and determinism invariants" and tested
``RunLedger``: a run's directory allocation, its persisted ``run.json``, its ``status``
field. Decision 7 of the lean design deletes all three -- MLflow is the source of truth
for run identity and status now (`dsio.runs.record`'s own module docstring has the full
argument). Tests of that deleted behaviour are gone with it; each one below that replaces
one says explicitly what happened to the property it used to check.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient

from dsio.config import RunConfig
from dsio.contracts import NonCanonicalValueError, sha256_of
from dsio.eval.metrics import MetricError
from dsio.runs.provenance import capture_env, capture_git
from dsio.runs.record import CONFIG_FILE, PATCH_FILE, REPRODUCE_FILE, Run, start_run
from dsio.runs.seeding import seed_everything
from dsio.train import load_runners
from dsio.train.runner import execute
from dsio.train.tracking import resolve_tracking_uri


def _start(config: RunConfig, **kwargs: object) -> Run:
    return start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
        **kwargs,  # type: ignore[arg-type]
    )


# --- what `start_run` still writes locally, before any work begins ---------------------


def test_provenance_is_written_before_any_work(config: RunConfig) -> None:
    """A run that crashes immediately must still have provenance captured.

    There is no ``run.json`` any more (removed test: ``test_record_is_written_before_
    any_work`` used to also check one existed here) -- there is nothing left to persist
    locally once ``status``/``metrics``/``error`` are gone (MLflow owns them), and a
    ``RunRecord`` that carries only git/env/config identity has no reason to be written
    to disk before a runner exists to log it anywhere. What "before any work begins"
    still guarantees is that the two files a reproduction needs are already on disk the
    moment ``start_run`` returns.
    """
    run = _start(config)
    assert (run.dir / CONFIG_FILE).is_file()
    assert (run.dir / REPRODUCE_FILE).is_file()


def test_fold_is_stamped_on_the_run_record(config: RunConfig) -> None:
    """Under fold-as-process (decision 6), fold is part of what identifies a run, so it
    is hoisted to a top-level record field the same way `seed` already is, rather than
    left only inside the nested `config` dict.

    The old second half of this test (``ledger.load(run.run_id).record.fold ==
    ...`` -- reloading the record from disk) is gone with ``RunLedger.load``: there is no
    on-disk record to reload. The property it was really checking -- that the fold
    survives past construction and is *searchable*, not just held in memory -- now lives
    in MLflow as the ``fold`` tag ``dsio.train.tracking.stamp_provenance`` sets;
    ``tests/cli/test_cli.py::test_the_run_record_stamps_which_fold_ran`` proves that end
    to end.
    """
    run = _start(config, fold=config.task.fold)  # type: ignore[attr-defined]
    assert run.record.fold == config.task.fold  # type: ignore[union-attr,attr-defined]


def test_fold_is_none_when_the_task_kind_carries_none(config: RunConfig) -> None:
    """A caller that never passes `fold` -- any task kind with no notion of one -- gets
    `None`, not a stamped value it never claimed."""
    run = _start(config)
    assert run.record.fold is None


def test_two_starts_always_get_distinct_scratch_directories(config: RunConfig) -> None:
    """Replaces ``test_identical_configs_get_distinct_run_ids``, which used to assert
    ``run_id``s were distinct: ``RunLedger._allocate_run_dir`` retried an ``os.mkdir``
    until it claimed a collision-free slot, so a fast rerun's *directory name*
    (``run_id``) could not collide. There is no slot to claim any more --
    ``start_run``'s scratch directory is ``tempfile.mkdtemp()`` -- so this checks the
    property that actually needs to hold: two starts never share a scratch directory,
    which ``tempfile.mkdtemp()`` guarantees on its own, without a retry loop.
    """
    directories = {_start(config).dir for _ in range(5)}
    assert len(directories) == 5


def test_run_id_is_a_label_not_a_claimed_identity(config: RunConfig) -> None:
    """The other half of what ``test_identical_configs_get_distinct_run_ids`` used to
    prove, made explicit rather than silently dropped: ``run_id`` is a human-readable
    label now (timestamp plus config-hash prefix), used only as MLflow's
    ``mlflow.runName`` tag (``dsio.train.tracking.build_mlflow_logger``) -- which MLflow
    does not require to be unique -- not a claimed, collision-free identity. Two starts
    in the same second *can* carry the same ``run_id``; what actually distinguishes them
    is MLflow's own run id, allocated when a runner builds its ``MLFlowLogger`` and
    recorded onto ``Run.mlflow_run_id``.
    ``tests/train/test_torch_runner.py::test_two_folds_of_one_config_land_in_one_mlflow_
    experiment`` proves that real distinctness end to end, with two actual training runs.
    """
    run = _start(config)
    assert run.run_id.endswith(config.config_hash[:12])
    assert run.mlflow_run_id is None, "Run itself never talks to MLflow -- see its docstring"


# --- run status: MLflow's job now, not `Run.__exit__`'s ---------------------------------


def test_run_exit_no_longer_records_anything(config: RunConfig) -> None:
    """Replaces ``test_failed_run_is_recorded_not_lost``, which used to assert
    ``ledger.load(run.run_id).record.status is RunStatus.FAILED`` after an exception
    propagated out of ``with ledger.start(...) as run: raise ...``. ``Run.__exit__`` is a
    no-op now (see its docstring): there is no local status field to flip, and MLflow is
    what is authoritative for status, so recording a failure is not this module's job at
    all any more -- it belongs to whichever runner built the MLflow run in the first
    place. The two tests below prove *that* property holds, against a real MLflow run
    reached through ``execute()``, which is the only path that can ever produce one.
    """
    with pytest.raises(RuntimeError), _start(config) as run:
        raise RuntimeError("boom")
    assert run.mlflow_run_id is None


def test_a_successful_run_is_recorded_finished_in_mlflow(config: RunConfig) -> None:
    """The acceptance half of the pair below: a run that completes normally must not be
    left looking like it is still running, or like nothing happened."""
    load_runners()
    run = _start(config)
    execute(config, run)

    assert run.mlflow_run_id is not None
    status = MlflowClient(resolve_tracking_uri()).get_run(run.mlflow_run_id).info.status
    assert status == RunStatus.to_string(RunStatus.FINISHED)


def test_a_failure_after_predict_still_marks_the_mlflow_run_failed(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal half: the property ``test_failed_run_is_recorded_not_lost`` used to
    check -- a crash leaves a *recorded* failure, not a silently successful-looking run
    -- moved to ``run_torch``'s own ``try``/``except`` around the MLflow run
    (``dsio.train.torch_task``). That guard exists specifically because Lightning's own
    ``Trainer`` already marks the MLflow run FINISHED the instant ``trainer.predict``
    *returns* -- before the scoring guards below it run -- so a failure there would look
    like a completed run in MLflow without it. This monkeypatches ``compute`` to fail
    exactly there, after Lightning's own finalize has already fired, and checks the
    status itself rather than merely that the exception propagates: a refusal-only
    version of this test could not tell a run correctly marked FAILED apart from one left
    stuck at FINISHED with an exception merely raised on top.
    """
    import dsio.train.torch_task as torch_task

    def explode(*args: object, **kwargs: object) -> dict[str, float]:
        raise MetricError("boom")

    monkeypatch.setattr(torch_task, "compute", explode)
    load_runners()

    run = _start(config)
    with pytest.raises(Exception, match="check stratification"):
        execute(config, run)

    assert run.mlflow_run_id is not None
    status = MlflowClient(resolve_tracking_uri()).get_run(run.mlflow_run_id).info.status
    assert status == RunStatus.to_string(RunStatus.FAILED)


# --- metrics filtering moved out of this module entirely --------------------------------

# ``test_non_finite_metrics_are_not_written`` used to live here, against ``Run.
# log_metrics``/``Run.read_metrics`` -- both deleted, along with every other metrics- or
# status-carrying field ``RunRecord`` used to have (see this module's docstring: MLflow is
# authoritative for metrics now). The NaN/inf-filtering property itself was not dropped,
# just rehomed: ``dsio.train.tracking.finite_metrics`` does the identical filtering,
# called from ``run_torch``/``run_ssl_pretrain`` right before the held-out metrics reach
# ``mlflow_logger.log_metrics`` -- ``tests/train/test_tracking.py::
# test_finite_metrics_drops_non_finite_values`` is its test now.


# --- provenance actually lands in MLflow, byte-identical --------------------------------


def test_provenance_lands_in_mlflow_byte_identical(config: RunConfig, git_repo: Path) -> None:
    """Task 3's own fake-backed verification: a run's diff, resolved config and
    reproduce script are logged into MLflow as artifacts before training starts
    (``dsio.train.tracking.stamp_provenance``, called first thing inside ``run_torch``),
    and read back exactly the bytes ``start_run`` wrote to local scratch. The tree is
    made dirty here so the patch -- the artifact ADR 0002 said MLflow could not hold at
    all -- is exercised too, not just the two files that always exist.
    """
    (git_repo / "tracked.txt").write_text("dirty for this test\n")
    load_runners()

    run = _start(config, repo_root=git_repo)
    execute(config, run)
    assert run.mlflow_run_id is not None

    client = MlflowClient(resolve_tracking_uri())
    tags = client.get_run(run.mlflow_run_id).data.tags
    assert tags["config_hash"] == config.config_hash

    for filename in (CONFIG_FILE, REPRODUCE_FILE, PATCH_FILE):
        local = (run.dir / filename).read_bytes()
        downloaded = Path(client.download_artifacts(run.mlflow_run_id, filename))
        assert downloaded.read_bytes() == local


# --- git and environment capture (unchanged by decision 7) ------------------------------


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


def test_patch_file_is_written_for_dirty_runs(git_repo: Path, config: RunConfig) -> None:
    (git_repo / "tracked.txt").write_text("dirty\n")
    run = _start(config, repo_root=git_repo)
    assert (run.dir / PATCH_FILE).is_file()
    assert b"dirty" in (run.dir / PATCH_FILE).read_bytes()


def test_reproduce_script_pins_the_commit(git_repo: Path, config: RunConfig) -> None:
    run = _start(config, repo_root=git_repo, command=("dsio", "run", "x"))
    script = (run.dir / REPRODUCE_FILE).read_text()
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=git_repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert f"git checkout {sha}" in script
    assert "uv sync --locked" in script
    assert "dsio run x" in script


def test_reproduce_script_syncs_the_extra_that_produced_this_run(
    git_repo: Path, config: RunConfig
) -> None:
    """Bug 2: a bare ``uv sync --locked`` uninstalls its own dependencies. ``torch``,
    ``lightning`` and ``mlflow-skinny`` live only in the ``cpu``/``gpu`` extras
    (``pyproject.toml``), never in the base ``dependencies`` set, so a bare sync
    *removes* them and the very next line of the script crashes with
    ``ModuleNotFoundError`` before rerunning anything. This test suite runs under
    ``uv run --extra cpu``, so the environment that captured this run's provenance really
    does have the ``cpu`` extra installed (``dsio.runs.provenance.capture_env`` detects
    it from ``torch.__version__``'s ``+cpu`` local segment) -- the script must sync that
    same extra, not a bare, dependency-stripping sync.
    """
    run = _start(config, repo_root=git_repo, command=("dsio", "run", "x"))
    assert run.record.env.extra == "cpu"
    script = (run.dir / REPRODUCE_FILE).read_text()
    assert "uv sync --locked --extra cpu" in script
    # And never the bare, dependency-stripping form on its own line.
    assert "\nuv sync --locked\n" not in script


def test_reproduce_script_falls_back_to_a_bare_sync_with_a_warning_when_the_extra_is_unknown(
    git_repo: Path, config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal-adjacent half of the pair above: when the extra genuinely cannot be
    determined (no torch, or a torch build this heuristic does not recognize --
    ``dsio.runs.provenance._detect_extra`` returns ``None`` rather than guessing), the
    script must not silently emit a wrong ``--extra`` and must not pretend it knows --
    it falls back to the old bare sync, with a warning a reader (or a script) can see,
    rather than a confident-but-wrong flag.
    """
    import dsio.runs.record as record

    monkeypatch.setattr(
        record, "capture_env", lambda: capture_env().model_copy(update={"extra": None})
    )
    run = _start(config, repo_root=git_repo, command=("dsio", "run", "x"))
    assert run.record.env.extra is None
    script = (run.dir / REPRODUCE_FILE).read_text()
    assert "\nuv sync --locked\n" in script
    assert "WARNING: could not determine which cpu/gpu extra" in script


def test_env_capture_records_the_lockfile() -> None:
    env = capture_env(lock_path=Path("uv.lock"))
    assert env.python
    assert env.lock_sha256 is not None, "uv.lock should be hashed for reproducibility"


def test_detect_extra_reads_the_torch_local_version_segment() -> None:
    """Unit-level pin on the heuristic itself (``dsio.runs.provenance._detect_extra``):
    the ``cpu`` extra's wheel carries a ``+cpu`` local version, the ``gpu`` extra's
    carries a ``+cu...`` one (``pyproject.toml``'s ``[tool.uv.sources]``/index comments),
    and anything else is honestly unrecognized rather than guessed at.
    """
    from dsio.runs.provenance import _detect_extra

    assert _detect_extra("2.13.0+cpu") == "cpu"
    assert _detect_extra("2.13.0+cu130") == "gpu"
    assert _detect_extra("2.13.0+cu121") == "gpu"
    assert _detect_extra("2.13.0") is None
    assert _detect_extra(None) is None


# --- determinism (unchanged by decision 7) -----------------------------------------------


@pytest.fixture(autouse=True)
def _dirty_ambient_rng() -> None:
    """Every test process starts from the same fixed, default RNG state. A direct call to
    ``execute()`` can then look reproducible, or look seed-sensitive, purely by accident of
    that shared starting point -- not because ``execute()`` actually reseeds anything from
    ``config.seed``. Perturbing every global RNG to a fixed, non-default value before each
    test removes that accident; the two tests below build on it further."""
    random.seed(20260821)
    np.random.seed(20260821)
    torch.manual_seed(20260821)


def _reseed_ambient(value: int = 20260821) -> None:
    """Reset every global RNG to an identical, fixed state.

    Called immediately before each of two ``execute()`` calls in
    ``test_different_seed_gives_different_metrics`` so the usual source of difference
    between two training runs -- whatever the first run happened to leave in the global
    RNG -- is cancelled out. With ambient state pinned identical for both, any difference
    in outcomes can only come from ``execute()`` using each variant's distinct
    ``config.seed``, which is exactly what that test exists to prove.
    """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _with_seed_sensitive_metrics(config: RunConfig) -> RunConfig:
    """``accuracy`` saturates to 1.0 on this trivially separable toy corpus regardless of
    the backbone's random initialisation, so a match (or mismatch) on accuracy alone would
    prove nothing about whether the seed reached the weights. ``log_loss`` is continuous
    and is what actually makes these tests sensitive to initialisation."""
    return config.model_copy(
        update={"task": config.task.model_copy(update={"metrics": ("accuracy", "log_loss")})}
    )


def test_same_seed_gives_identical_metrics(config: RunConfig) -> None:
    """Same config, same seed, twice -- through ``execute()`` alone, with no external
    ``seed_everything`` call to lean on. A freshly constructed backbone draws its initial
    weights from torch's global RNG, so a match here can only come from ``execute()``
    resetting that RNG itself from ``config.seed``, not from an accident of ambient state:
    the loop advances the RNGs between the two calls so they provably start from different
    places."""
    load_runners()
    seed_config = _with_seed_sensitive_metrics(config)
    results = []
    for i in range(2):
        if i == 1:
            torch.rand(97)
            np.random.random(97)
        with _start(seed_config) as run:
            results.append(execute(seed_config, run))
    assert results[0] == results[1]


def test_different_seed_gives_different_metrics(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards against a seed that is recorded but never actually wired through.

    No external ``seed_everything`` call backs this up either: the ambient RNGs are reset
    to an *identical* fixed state right before each run (see ``_reseed_ambient``). That
    alone is not enough here, though: each dataloader's shuffle order is *also* seeded
    directly from ``config.seed`` (``dsio.dataset.dataset.make_loader`` ->
    ``dataloader_kwargs``), completely independently of ``execute()``'s own seeding -- so
    the two variants would still shuffle their batches differently, and their metrics
    would still differ, even if ``execute()`` reseeded nothing at all. Pinning the
    dataloader seed identical for both runs removes that confound, so a surviving
    difference in outcomes can only come from ``execute()`` itself carrying
    ``config.seed`` into the backbone's initial weights.
    """
    import dsio.dataset.dataset as dataset_module

    real_dataloader_kwargs = dataset_module.dataloader_kwargs
    monkeypatch.setattr(
        dataset_module, "dataloader_kwargs", lambda seed: real_dataloader_kwargs(0)
    )

    load_runners()
    outcomes = []
    for seed in (1, 999):
        variant = _with_seed_sensitive_metrics(config.model_copy(update={"seed": seed}))
        _reseed_ambient()
        with _start(variant) as run:
            outcomes.append(execute(variant, run))
    assert outcomes[0] != outcomes[1]


def test_seed_is_range_checked() -> None:
    with pytest.raises(ValueError, match="seed must be in"):
        seed_everything(-1)
