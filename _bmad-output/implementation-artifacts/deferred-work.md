# Deferred Work

## Deferred from: code review of 1-1-install-dsio-and-run-a-project-owned-flow (2026-09-21)

- Resolve `uv.lock` relative to `repo_root` when capturing run provenance. The mismatch between Git and environment roots predates Story 1.1 and belongs with deterministic identity work.
- Define supported replay behavior for consumer projects that do not use `uv.lock`. The unconditional locked sync predates Story 1.1 and belongs with reproducibility design.
- Replace the handwritten recorded-command quoting with `shlex.quote()` and cover shell metacharacters. The quoting defect predates Story 1.1 and is independent of the orchestration removal.

## Deferred from: canonical supervised reference cleanup (2026-09-23)

- Define how project-owned export reconstructs versioned model and preprocessor components from checkpoint/run evidence instead of assuming the currently imported class still matches an older checkpoint.

## Deferred from: Kaggle consumer portfolio (2026-09-24)

- Extend DSIO execution provenance so consumer commit/dirty-patch identity, dependency-lock digest, package digest, relevant command, and environment are captured consistently by the shared spine rather than reimplemented by each project flow.
