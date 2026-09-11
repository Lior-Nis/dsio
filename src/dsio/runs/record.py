"""The provenance stamper: git rev, dirty-diff capture, the reproduce script.

Decision 7 of the lean design supersedes ADR 0002 ("the ledger is authoritative; trackers
are sinks"): MLflow is the source of truth for run identity, status and metrics now, and a
run that cannot write to it does not start (``dsio.train.tracking.require_mlflow``). This
module no longer allocates a permanent, append-only ledger directory -- there is no
``RunLedger``, no ``run.json``, no ``list_runs``/``iter_runs``, and nothing here is ever
read back from disk. What it still owns, because MLflow has no notion of any of it:

* **Capturing git and environment state** (``dsio.runs.provenance``), unchanged.
* **A local scratch directory** (:attr:`Run.dir`) a runner writes into -- the dirty-tree
  diff, the resolved config, the reproduce script, and later whatever training artifacts
  (predictions, checkpoints, an exported encoder) the runner produces -- before those
  become MLflow artifacts. It lives under the OS temp directory by default (or
  ``DSIO_RUNS_ROOT`` when set, which is how the test suite keeps it under ``tmp_path``),
  never under the source tree.
* **The reproduce script**, which pins the commit (and the dirty-tree patch, when there
  is one) so a run started from an uncommitted change still reproduces exactly.

Two things that used to live here no longer do, on purpose:

**Run-id allocation.** ``RunRecord.run_id`` is still computed (:func:`_run_id`) and still
used as the MLflow run's ``mlflow.runName`` tag, but it is a *human-readable label* now,
not a claimed, collision-free identity -- there is no directory-allocation loop to make it
one, and there does not need to be. MLflow allocates the real identity itself, a UUID, the
moment a runner's ``MLFlowLogger`` creates its run; that id is what actually makes two runs
of the same config distinguishable, and it is recorded back onto :attr:`Run.mlflow_run_id`
once the runner has it (``Run`` itself never talks to MLflow -- see below).

**Run status.** There is no ``RunStatus`` here any more. Whether a run completed or failed
is answered by MLflow's own run status, set by the runner (``dsio.train.torch_task``,
``dsio.train.ssl_task``) once it holds the MLflow run that ``Run`` cannot construct itself.

Why ``Run`` cannot construct or hold an MLflow run: ``dsio.runs`` is a foundation module
(see the import-linter contract in ``pyproject.toml``, "Foundation modules import no other
dsio module") specifically so it stays testable and reasoned-about in isolation. MLflow
reachability, the ``MLFlowLogger`` and everything that logs to it belong to
``dsio.train.tracking``, which is allowed to depend on this module -- not the other way
around. So ``Run`` hands the runner everything it captured (git state, the resolved
config, the local scratch directory) and the runner is who actually talks to MLflow.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import yaml
from pydantic import Field

from dsio.contracts import DsioModel, atomic_write
from dsio.runs.provenance import EnvState, GitState, capture_env, capture_git, working_tree_patch

#: Optional: where a run's local scratch directory is created. Unset by default, which
#: puts it under the OS temp directory -- decision 7's whole complaint about ADR 0002 was
#: generated run data mixing with the source tree, so the default must not be a path under
#: the repo. Tests point this at `tmp_path` (`tests/conftest.py`) so scratch directories
#: are cleaned up with everything else pytest already cleans up.
RUNS_ROOT_ENV = "DSIO_RUNS_ROOT"

CONFIG_FILE = "config.resolved.yaml"
PATCH_FILE = "git.patch"
REPRODUCE_FILE = "reproduce.sh"
ARTIFACTS_DIR = "artifacts"


class RunRecord(DsioModel):
    """Everything captured about a run before any work begins.

    Deliberately does not know about config *schemas* -- it stores serialized data and a
    hash, which keeps it a generic record that outlives any particular config layout. No
    ``status``, ``metrics`` or ``error`` any more: MLflow is authoritative for all three
    now (decision 7), and a second, locally-tracked copy of them here would be exactly the
    "three parallel comparison mechanisms" ADR 0002 itself was written to avoid.
    """

    run_id: str
    name: str
    created_at: str

    config_hash: str
    config: dict[str, Any]

    seed: int
    seeds: dict[str, int] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()

    git: GitState = GitState()
    env: EnvState

    command: tuple[str, ...] = ()
    # Which fold this run trains, for task kinds that carry one (SslPretrainTask,
    # TorchTask). Hoisted to a top-level field the same way `seed` is: under
    # fold-as-process (decision 6) the fold is part of what identifies a run. `None` for a
    # task kind with no notion of a fold.
    fold: int | None = None

    @property
    def reproducible(self) -> bool:
        """Whether this run carries enough provenance to be reconstructed.

        ADR 0003's clean-tree property, unchanged by decision 7: a dirty tree is still
        reproducible as long as its diff was captured (``git.code_hash`` covers both the
        clean and the dirty case -- see ``dsio.runs.provenance.capture_git``), it is just
        not eligible for *promotion*, a stricter and separate question that has no
        implementation today (see ``docs/adr/0003-never-block-gate-at-promotion.md``).
        """
        return self.git.code_hash is not None and self.env.lock_sha256 is not None


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id(config_hash: str, when: datetime | None = None) -> str:
    """A sortable, human-readable label: timestamp plus config digest.

    Not a claimed, collision-free identity -- see this module's docstring. Two runs
    started in the same second carry the *same* label, on purpose; MLflow's own run id is
    what actually distinguishes them (:attr:`Run.mlflow_run_id`).
    """
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_hash[:12]}"


class Run:
    """A live run: captured provenance, plus a local scratch directory to write into.

    ``mlflow_run_id`` starts ``None`` and is filled in by the runner once it holds the
    MLflow run this dsio run is logging to (``dsio.train.tracking.build_mlflow_logger``,
    called from ``dsio.train.torch_task.run_torch`` / ``dsio.train.ssl_task.
    run_ssl_pretrain``) -- ``Run`` itself never talks to MLflow (see the module
    docstring), so this is the one field it does not set.

    Supports ``with run:`` for compatibility with every existing call site, but the
    context manager does nothing: there is no local state whose correctness depends on
    exiting the block (no status to flip, no file to close), and deleting the scratch
    directory on exit would break every caller -- test and CLI alike -- that inspects
    ``run.artifacts_dir`` after the run completes, which is the ordinary way to look at
    what a run just wrote. Nothing here cleans the scratch directory up automatically;
    it lives under the OS temp directory (or ``DSIO_RUNS_ROOT``), not the source tree, so
    leaving it for the OS's own temp-file hygiene is the same tradeoff every other
    ``tempfile.mkdtemp()`` caller makes.
    """

    def __init__(self, directory: Path, record: RunRecord) -> None:
        self.dir = directory
        self.record = record
        self.mlflow_run_id: str | None = None

    @property
    def run_id(self) -> str:
        return self.record.run_id

    @property
    def artifacts_dir(self) -> Path:
        path = self.dir / ARTIFACTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def provenance_files(self) -> list[Path]:
        """The provenance files written at :func:`start_run` time, in log order.

        ``config.resolved.yaml`` and ``reproduce.sh`` always exist; ``git.patch`` only
        when the tree was dirty. What a runner logs to MLflow as artifacts
        (``dsio.train.tracking.stamp_provenance``) is exactly this list.
        """
        candidates = (self.dir / CONFIG_FILE, self.dir / REPRODUCE_FILE, self.dir / PATCH_FILE)
        return [path for path in candidates if path.is_file()]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def start_run(
    *,
    name: str,
    config: dict[str, Any],
    config_hash: str,
    seed: int,
    seeds: dict[str, int] | None = None,
    tags: tuple[str, ...] = (),
    command: tuple[str, ...] = (),
    fold: int | None = None,
    repo_root: Path | None = None,
    root: Path | str | None = None,
) -> Run:
    """Capture provenance and stage a run's local scratch directory.

    Git and environment state are captured, and the resolved config, the reproduce
    script, and (if the tree is dirty) the diff are written to disk immediately -- the
    same "written before any work begins" guarantee ``RunLedger.start`` used to give, just
    no longer into a permanent ledger. A runner logs these into MLflow as its first act
    (``dsio.train.tracking.stamp_provenance``), so they land there before training starts
    too, not merely on local disk.

    ``root`` overrides ``DSIO_RUNS_ROOT``; both are optional, and the default is the OS
    temp directory (see :data:`RUNS_ROOT_ENV`).
    """
    git = capture_git(cwd=repo_root)
    env = capture_env()
    run_id = _run_id(config_hash)

    record = RunRecord(
        run_id=run_id,
        name=name,
        created_at=_utc_now(),
        config_hash=config_hash,
        config=config,
        seed=seed,
        seeds=seeds or {},
        tags=tags,
        fold=fold,
        git=git,
        env=env,
        command=command,
    )

    scratch_root = Path(root) if root is not None else _default_scratch_root()
    if scratch_root is not None:
        scratch_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"dsio-run-{run_id}-", dir=scratch_root))

    atomic_write(
        directory / CONFIG_FILE,
        yaml.safe_dump(config, sort_keys=True, default_flow_style=False).encode(),
    )
    if git.dirty:
        atomic_write(directory / PATCH_FILE, working_tree_patch(cwd=repo_root))
    atomic_write(directory / REPRODUCE_FILE, _reproduce_script(record).encode())
    (directory / REPRODUCE_FILE).chmod(0o755)

    return Run(directory, record)


def _default_scratch_root() -> Path | None:
    root = os.environ.get(RUNS_ROOT_ENV)
    return Path(root) if root else None


def _reproduce_script(record: RunRecord) -> str:
    """Emit a self-contained script that rebuilds the environment and reruns."""
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by dsio. Reconstructs the environment for this run and reruns it.",
        "set -euo pipefail",
        "",
        f"# run_id      {record.run_id}",
        f"# config_hash {record.config_hash}",
        "",
    ]
    if record.git.sha:
        lines += [
            f'echo "checking out {record.git.sha}"',
            f"git checkout {record.git.sha}",
        ]
        if record.git.dirty:
            lines += [
                "",
                "# This run had uncommitted changes. The patch reproduces them exactly.",
                f'git apply "$(dirname "$0")/{PATCH_FILE}"',
            ]
    else:
        lines.append('echo "WARNING: no git provenance was captured for this run" >&2')

    if record.env.extra:
        # `torch`/`lightning`/`mlflow-skinny` live only in the `cpu`/`gpu` extras
        # (`pyproject.toml`), not in `dependencies` -- a bare `uv sync --locked` would
        # *remove* them (they are not in the base dependency set at all) and every line
        # after it would crash with `ModuleNotFoundError` before rerunning anything. The
        # extra this run actually used was captured once, at run time, by
        # `dsio.runs.provenance.capture_env` (see `EnvState.extra`); replaying with the
        # same extra is what makes this script self-contained on a fresh clone.
        lines += ["", f"uv sync --locked --extra {record.env.extra}", ""]
    else:
        lines += [
            "",
            'echo "WARNING: could not determine which cpu/gpu extra this run used;'
            ' pass one explicitly, e.g. uv sync --locked --extra cpu" >&2',
            "uv sync --locked",
            "",
        ]
    if record.command:
        lines.append(" ".join(_quote(part) for part in record.command))
    else:
        lines.append('echo "WARNING: no command was recorded for this run" >&2')
    return "\n".join(lines) + "\n"


def _quote(token: str) -> str:
    return token if all(ch.isalnum() or ch in "-_=./:" for ch in token) else f"'{token}'"
