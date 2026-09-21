# Deferred Work

## Deferred from: code review of 1-1-install-dsio-and-run-a-project-owned-flow (2026-09-21)

- Resolve `uv.lock` relative to `repo_root` when capturing run provenance. The mismatch between Git and environment roots predates Story 1.1 and belongs with deterministic identity work.
- Define supported replay behavior for consumer projects that do not use `uv.lock`. The unconditional locked sync predates Story 1.1 and belongs with reproducibility design.
- Replace the handwritten recorded-command quoting with `shlex.quote()` and cover shell metacharacters. The quoting defect predates Story 1.1 and is independent of the orchestration removal.
