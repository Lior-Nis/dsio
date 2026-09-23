# Deferred Work

## Deferred from: code review of 1-1-install-dsio-and-run-a-project-owned-flow (2026-09-21)

- Resolve `uv.lock` relative to `repo_root` when capturing run provenance. The mismatch between Git and environment roots predates Story 1.1 and belongs with deterministic identity work.
- Define supported replay behavior for consumer projects that do not use `uv.lock`. The unconditional locked sync predates Story 1.1 and belongs with reproducibility design.
- Replace the handwritten recorded-command quoting with `shlex.quote()` and cover shell metacharacters. The quoting defect predates Story 1.1 and is independent of the orchestration removal.

## Deferred from: canonical supervised reference cleanup (2026-09-23)

- Build a small portfolio of Kaggle competition consumer projects that run end to end through DSIO, covering grouped classification, temporal regression, and self-supervised representation reuse. Keep competition-specific ingestion, components, and Prefect DAGs outside `src/dsio`; admit only independently reusable gaps through the governed experimental process.
- Bind export inputs to the originating training evidence so a checkpoint cannot be paired with unrelated dataset or split metadata while producing plausible provenance. Treat this as cross-cutting lineage hardening for every project-owned export task, not a supervised-only check.
- Define how project-owned export reconstructs versioned model and preprocessor components from checkpoint/run evidence instead of assuming the currently imported class still matches an older checkpoint.
