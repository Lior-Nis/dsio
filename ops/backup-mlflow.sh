#!/usr/bin/env bash
# Nightly backup of the MLflow source of truth (decision 7 of
# docs/superpowers/specs/2026-08-20-dsio-lean-design.md, "Backup" section). Run this via
# ops/mlflow-backup.timer -- see that file and ops/mlflow-backup.service for installation.
#
# MLflow being the source of truth makes this a correctness concern, not hygiene: Postgres
# holds run metadata and artifact *URIs*, while the files live in the mlartifacts volume.
# Restoring the database alone yields an index pointing at files that no longer exist, so
# both halves are captured every run.
#
# The archive is push-only and append-only. Data moves local -> Drive and never back;
# nothing in the system reads from the archive, and restore is a deliberate manual act
# (see the RESTORE section at the bottom of this file).
#
# `rclone copy`, never `rclone sync`: sync makes the destination match the source, so it
# would delete from the archive whatever disappeared locally -- propagating a wiped or
# corrupted volume straight into the backup at exactly the moment the backup matters.
# `copy` never deletes, and with timestamped dumps the archive is append-only by
# construction. This property is covered by tests/ops/test_backup.py, which breaks this
# script (copy -> sync) and confirms the test catches it -- do not "simplify" the two
# `rclone copy` calls below into a `sync` without re-reading that test.
#
# The `gdrive` remote (see REMOTE below) must be configured with the `drive.file` OAuth
# scope -- access only to files this application created -- so the one-way property is
# enforced by the credential, not by discipline. `rclone config` cannot be done for you;
# see RESTORE section for the one-time setup this script assumes already happened.
set -euo pipefail

# --- configuration (env-overridable for testing; production defaults match the spec) ------
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/mlflow}"
ARTIFACTS_VOLUME="${ARTIFACTS_VOLUME:-/var/lib/docker/volumes/mlartifacts/_data}"
REMOTE="${REMOTE:-gdrive:dsio-backup}"
# SKIP_DB=1 exercises only the artifacts half, which needs no Docker and no Postgres --
# used by tests/ops/test_backup.py to verify the append-only property in isolation.
SKIP_DB="${SKIP_DB:-0}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BACKUP_DIR"

if [ "$SKIP_DB" != "1" ]; then
  # DB: small, full dump, timestamped, keep many. pg_dump is atomic; the artifact copy
  # below is not taken at the same instant, so a run writing during the backup can
  # straddle the two. At nightly cadence the exposure is one in-flight run.
  docker compose -f "$REPO_DIR/compose.yaml" exec -T postgres pg_dump -U mlflow mlflow \
    | gzip >"$BACKUP_DIR/db-$STAMP.sql.gz"
  rclone copy "$BACKUP_DIR" "$REMOTE/db"
fi

# Artifacts: incremental. MLflow artifacts are immutable once a run finishes, so nothing
# already in the archive is ever re-uploaded.
rclone copy "$ARTIFACTS_VOLUME" "$REMOTE/artifacts"

# --- RESTORE (deliberate manual act; nothing in this script or in dsio does this for you) -
#
# 1. Stop the stack so nothing writes while you restore:
#      docker compose down
#
# 2. Pick the dump to restore from the archive (they are timestamped and never deleted):
#      rclone lsf gdrive:dsio-backup/db | sort | tail -1
#
# 3. Recreate the volumes empty, then bring Postgres up alone:
#      docker volume rm pgdata mlartifacts   # only if actually restoring onto a fresh volume
#      docker compose up -d postgres
#
# 4. Restore the database from the chosen dump:
#      rclone copy gdrive:dsio-backup/db/db-<STAMP>.sql.gz /tmp/restore/
#      gunzip -c /tmp/restore/db-<STAMP>.sql.gz | \
#        docker compose exec -T postgres psql -U mlflow mlflow
#
# 5. Restore the artifacts volume (rclone copy, not sync -- same reasoning as above; the
#    destination is a fresh empty volume so there is nothing to preserve by not deleting,
#    but `copy` is still correct and `sync` is not needed):
#      rclone copy gdrive:dsio-backup/artifacts /var/lib/docker/volumes/mlartifacts/_data
#
# 6. Bring MLflow back up and confirm the UI shows the restored runs:
#      docker compose up -d
#
# The DB dump and the artifact copy are not taken at the same instant (see the comment
# above the artifacts `rclone copy` call), so after a restore, check the newest run in the
# MLflow UI -- if it has metadata but a broken artifact link, it was the in-flight run at
# backup time and its artifact should be treated as lost, not restored.
