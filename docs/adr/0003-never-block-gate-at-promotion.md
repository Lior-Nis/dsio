# 3. Never block a run; gate at promotion

Status: accepted (2026-08-15)

## Context

"Reproduction is core" invites a strict rule: refuse to run with a dirty working tree.
That rule fails in practice. It fights hardest exactly when iteration matters most, and
the `--allow-dirty` escape hatch becomes the default habit within a week — at which point
you have the friction and none of the guarantee.

The opposite failure is also real. FORGE captured a git SHA in exactly one place,
`utils/wandb_model_registry.py`, wrapped in a try/except and only for registry artifacts.
Nothing else recorded a commit or dirty state.

## Decision

A dirty tree never blocks a run. Instead the run captures enough to be reconstructed
anyway: the commit SHA, a patch of every uncommitted change including untracked files, the
lockfile hash, all seeds, and the environment. `code_hash` is either the SHA or
`{sha}-dirty-{digest}` where the digest covers status *and* patch contents.

When git is unavailable, `code_hash` is `None` — never a guess. A confident-but-wrong
provenance value is worse than a missing one, because it looks trustworthy.

The gate lives at model-registry promotion. `promotion_blockers()` (`dsio.artifacts.store`,
survived Plan 3b's Task 4 migration to MLflow's registry unchanged) refuses a run with a
dirty tree or missing provenance, reading `git`/`env` off whatever record it is given —
`RunRecord.git`/`RunRecord.env` (`dsio.runs.record`), captured by `dsio.runs.provenance`'s
`capture_git`/`capture_env`. It exists today as a policy function with its own tests
(`tests/artifacts/test_registry.py`); no CLI command calls it yet — `dsio run` is
presently the CLI's only command (`dsio.cli.main`) — so `dsio registry promote` is
aspirational rather than current. The envelope shape it would need is already in place:
`BlockedError` and `ErrorCode.BLOCKED = "blocked"` (`dsio.cli.envelope`) exist, unused,
waiting for a caller.

## Consequences

Exploration stays frictionless while the audit trail stays intact. A dirty run is tagged
and still reproducible, because the diff is an artifact.

Promotion is the one place dsio says no. That is deliberate: a registered model outlives
the session that produced it, so it must name a commit.
