"""The Run ledger: dsio's authoritative record of what happened.

Authority lives here, on disk, in plain files — not in MLflow or W&B, which are sinks.
The failure this prevents is run identity never being modelled at all, so that which run
produced a published number has to be recovered later from filenames and timestamps.
A run record is written the moment a run starts, so identity exists even for a run that
crashes.

The ledger deliberately knows nothing about config *schemas* — it stores serialized data
and a hash. That keeps it a generic ledger that outlives any particular config layout.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import yaml
from pydantic import Field

from dsio.contracts import DsioModel, atomic_write
from dsio.runs.provenance import EnvState, GitState, capture_env, capture_git, working_tree_patch

RUNS_ROOT_ENV = "DSIO_RUNS_ROOT"
DEFAULT_RUNS_ROOT = Path("runs")

RUN_FILE = "run.json"
CONFIG_FILE = "config.resolved.yaml"
METRICS_FILE = "metrics.jsonl"
PATCH_FILE = "git.patch"
REPRODUCE_FILE = "reproduce.sh"
ARTIFACTS_DIR = "artifacts"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunRecord(DsioModel):
    """Everything needed to identify, compare, and reconstruct a run."""

    run_id: str
    name: str
    status: RunStatus
    created_at: str
    finished_at: str | None = None

    config_hash: str
    config: dict[str, Any]

    seed: int
    seeds: dict[str, int] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()

    git: GitState = GitState()
    env: EnvState

    command: tuple[str, ...] = ()
    data_snapshot_ids: tuple[str, ...] = ()
    split_manifest_sha: str | None = None

    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None

    @property
    def reproducible(self) -> bool:
        """Whether this run carries enough provenance to be reconstructed."""
        return self.git.code_hash is not None and self.env.lock_sha256 is not None


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id(config_hash: str, when: datetime | None = None) -> str:
    """Sortable, unique, and self-identifying: timestamp plus config digest."""
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_hash[:12]}"


def _allocate_run_dir(root: Path, config_hash: str) -> tuple[str, Path]:
    """Claim a unique run directory, disambiguating same-second reruns.

    The timestamp has one-second resolution, so re-running one config twice in the same
    second collides. Creating the directory with ``exist_ok=False`` is the claim: whoever
    wins the mkdir owns the id, which keeps this correct across concurrent processes
    rather than merely across a single loop.
    """
    base = _run_id(config_hash)
    for ordinal in range(1, 1000):
        run_id = base if ordinal == 1 else f"{base}-{ordinal}"
        directory = root / run_id
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_id, directory
    raise RuntimeError(
        f"could not allocate a run directory under {root} for config {config_hash[:12]}"
    )


class Run:
    """A live run: a directory, an append-only metric stream, and a record."""

    def __init__(self, directory: Path, record: RunRecord) -> None:
        self.dir = directory
        self.record = record

    @property
    def run_id(self) -> str:
        return self.record.run_id

    @property
    def artifacts_dir(self) -> Path:
        path = self.dir / ARTIFACTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Append a metric row to the run's stream.

        Non-finite values are dropped rather than written: a NaN in the stream breaks
        every downstream comparison, and silently poisoning a hash is worse than a gap.
        """
        finite = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float) and float(value) == float(value)
            and float(value) not in (float("inf"), float("-inf"))
        }
        if not finite:
            return
        row: dict[str, Any] = {"t": _utc_now(), **finite}
        if step is not None:
            row["step"] = step
        with (self.dir / METRICS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def read_metrics(self) -> list[dict[str, Any]]:
        """Return every metric row logged so far."""
        path = self.dir / METRICS_FILE
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def update(self, **fields: Any) -> None:
        """Replace record fields and persist. The record is frozen, so this rebuilds it."""
        self.record = self.record.model_copy(update=fields)
        self._persist()

    def finish(
        self,
        status: RunStatus = RunStatus.COMPLETED,
        metrics: dict[str, float] | None = None,
        error: str | None = None,
    ) -> RunRecord:
        """Close the run out, recording final metrics and status."""
        self.update(
            status=status,
            finished_at=_utc_now(),
            metrics={**self.record.metrics, **(metrics or {})},
            error=error,
        )
        return self.record

    def _persist(self) -> None:
        atomic_write(
            self.dir / RUN_FILE,
            json.dumps(self.record.model_dump(mode="json"), indent=2, sort_keys=True).encode(),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            if self.record.status is RunStatus.RUNNING:
                self.finish(RunStatus.COMPLETED)
            return
        self.finish(RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


class RunLedger:
    """Reads and writes runs under a root directory."""

    def __init__(self, root: Path | str | None = None) -> None:
        if root is not None:
            self.root = Path(root)
        else:
            self.root = Path(os.environ.get(RUNS_ROOT_ENV, DEFAULT_RUNS_ROOT))

    def start(
        self,
        *,
        name: str,
        config: dict[str, Any],
        config_hash: str,
        seed: int,
        seeds: dict[str, int] | None = None,
        tags: tuple[str, ...] = (),
        command: tuple[str, ...] = (),
        repo_root: Path | None = None,
    ) -> Run:
        """Create a run directory and write its initial record.

        The record is written *before* any work happens, so a run that crashes still has
        an identity and a config on disk.
        """
        git = capture_git(cwd=repo_root)
        env = capture_env()
        run_id, directory = _allocate_run_dir(self.root, config_hash)

        record = RunRecord(
            run_id=run_id,
            name=name,
            status=RunStatus.RUNNING,
            created_at=_utc_now(),
            config_hash=config_hash,
            config=config,
            seed=seed,
            seeds=seeds or {},
            tags=tags,
            git=git,
            env=env,
            command=command,
        )
        run = Run(directory, record)
        run._persist()

        atomic_write(
            directory / CONFIG_FILE,
            yaml.safe_dump(config, sort_keys=True, default_flow_style=False).encode(),
        )
        if git.dirty:
            atomic_write(directory / PATCH_FILE, working_tree_patch(cwd=repo_root))
        atomic_write(directory / REPRODUCE_FILE, _reproduce_script(record).encode())
        (directory / REPRODUCE_FILE).chmod(0o755)
        return run

    def load(self, run_id: str) -> Run:
        """Load an existing run by id."""
        directory = self.root / run_id
        path = directory / RUN_FILE
        if not path.is_file():
            raise FileNotFoundError(f"no run record at {path}")
        record = RunRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
        return Run(directory, record)

    def list_runs(self) -> list[RunRecord]:
        """Return every run record, newest first."""
        if not self.root.is_dir():
            return []
        records: list[RunRecord] = []
        for path in sorted(self.root.glob(f"*/{RUN_FILE}"), reverse=True):
            records.append(RunRecord.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return records

    def iter_runs(self) -> Iterator[RunRecord]:
        yield from self.list_runs()


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

    lines += ["", "uv sync --locked", ""]
    if record.command:
        lines.append(" ".join(_quote(part) for part in record.command))
    else:
        lines.append('echo "WARNING: no command was recorded for this run" >&2')
    return "\n".join(lines) + "\n"


def _quote(token: str) -> str:
    return token if all(ch.isalnum() or ch in "-_=./:" for ch in token) else f"'{token}'"
