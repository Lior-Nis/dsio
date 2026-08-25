"""ops/backup-mlflow.sh's one testable property: the archive is append-only.

The script's other half -- the Postgres dump -- needs the compose stack and is exercised
manually against the running stack (see the task report), not here: this suite must not
need Docker, matching the rest of this plan's "the suite must not need Docker" constraint
(docs/superpowers/plans/2026-08-25-mlflow-source-of-truth.md).

`rclone copy`, never `rclone sync`, is the one property in the Backup section
(docs/superpowers/specs/2026-08-20-dsio-lean-design.md) that turns a backup into a
liability if it regresses: `sync` makes the destination match the source, so it deletes
from the archive whatever disappeared locally -- propagating a wiped or corrupted volume
straight into the backup at exactly the moment the backup matters. `copy` never deletes.

This is fully testable without Google Drive: `rclone` supports a plain local directory as
a remote, no config needed. The test below runs the real script (`SKIP_DB=1`, so no Docker
or Postgres is involved -- only the artifacts `rclone copy` line executes) against a local
source directory and a local "archive" directory standing in for Drive, deletes a file from
the source, reruns the script, and asserts the file is still in the archive. Swapping
`copy` for `sync` in the script must make this fail; that substitution is exercised by hand
(see the task report) rather than as an automated mutation test, since the mutation would
need to touch the checked-in script itself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "ops" / "backup-mlflow.sh"

requires_rclone = pytest.mark.skipif(
    shutil.which("rclone") is None,
    reason=(
        "rclone is not installed on this machine (checked via `shutil.which('rclone')`). "
        "Skipping explicitly rather than silently passing: a backup test that no-ops "
        "reads as coverage it doesn't have. Install rclone to exercise this test."
    ),
)


def _run_backup(*, artifacts_volume: Path, remote_dir: Path, backup_dir: Path) -> None:
    subprocess.run(
        ["bash", str(SCRIPT)],
        check=True,
        env={
            "PATH": os.environ["PATH"],
            "SKIP_DB": "1",
            "ARTIFACTS_VOLUME": str(artifacts_volume),
            "REMOTE": str(remote_dir),
            "BACKUP_DIR": str(backup_dir),
        },
        capture_output=True,
        text=True,
    )


@requires_rclone
def test_deleting_the_source_does_not_delete_it_from_the_archive(tmp_path: Path) -> None:
    """The append-only property: `rclone copy` must never remove anything from the
    archive, even when the thing it copied from has since vanished locally."""
    source = tmp_path / "mlartifacts"
    source.mkdir()
    (source / "run-1-model.bin").write_text("weights")

    remote = tmp_path / "gdrive-stub" / "dsio-backup"
    backup_dir = tmp_path / "var-backups-mlflow"

    _run_backup(artifacts_volume=source, remote_dir=remote, backup_dir=backup_dir)
    archived = remote / "artifacts" / "run-1-model.bin"
    assert archived.exists(), "first backup did not copy the file into the archive"
    assert archived.read_text() == "weights"

    # The volume is wiped -- the exact scenario the spec calls out ("a mistaken flag" or
    # a corrupted/wiped volume must not propagate into the archive).
    (source / "run-1-model.bin").unlink()
    assert not list(source.iterdir()), "test setup: source should be empty now"

    _run_backup(artifacts_volume=source, remote_dir=remote, backup_dir=backup_dir)

    assert archived.exists(), (
        "the file that disappeared locally is gone from the archive too -- this is the "
        "exact failure `rclone sync` would cause and `rclone copy` must not"
    )
    assert archived.read_text() == "weights"


@requires_rclone
def test_a_new_file_is_added_on_the_next_run(tmp_path: Path) -> None:
    """The accepted-case complement to the deletion test above: `copy` must still pick up
    genuinely new files, not just refuse to delete old ones (a script that copied nothing
    at all would pass the deletion test for the wrong reason)."""
    source = tmp_path / "mlartifacts"
    source.mkdir()
    (source / "run-1-model.bin").write_text("weights")

    remote = tmp_path / "gdrive-stub" / "dsio-backup"
    backup_dir = tmp_path / "var-backups-mlflow"

    _run_backup(artifacts_volume=source, remote_dir=remote, backup_dir=backup_dir)

    (source / "run-2-model.bin").write_text("more weights")
    _run_backup(artifacts_volume=source, remote_dir=remote, backup_dir=backup_dir)

    archive_dir = remote / "artifacts"
    assert (archive_dir / "run-1-model.bin").exists()
    assert (archive_dir / "run-2-model.bin").read_text() == "more weights"


def test_script_uses_copy_not_sync() -> None:
    """A cheap static guard alongside the functional test above: greps the script for the
    two `rclone` invocations and asserts neither is a `sync`. This does not require
    rclone itself, so it runs even when the functional tests above are skipped."""
    text = SCRIPT.read_text()
    rclone_lines = [
        line
        for line in text.splitlines()
        if "rclone " in line and "#" not in line.split("rclone", 1)[0]
    ]
    assert rclone_lines, "expected at least one rclone invocation in the backup script"
    for line in rclone_lines:
        assert "rclone copy" in line, f"expected `rclone copy`, found: {line!r}"
        assert "rclone sync" not in line
