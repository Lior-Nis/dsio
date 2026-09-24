---
status: accepted
date: 2026-09-24
amends: ADR-0016
---

# Execution provenance is part of evidence identity

Scientific configuration alone does not identify an execution. The same dataset, split,
seed, and component references can produce different evidence when consumer code, the
dependency lock, the installed DSIO package, the command, or the numerical environment
changes. Project flows should not reimplement those captures, and requiring every caller
to pass them would make the provenance interface as complicated as its implementation.

`execution_identity()` and `record_provenance()` therefore capture execution context
themselves. Provenance schema 2 adds the consumer Git commit and dirty-patch identity, the
repository-root `uv.lock` digest when present, a digest of the installed DSIO package
sources, the redacted process command, and Python/platform/PyTorch/accelerator facts. The
document remains the single hashed identity record. A dirty patch is also stored as an
MLflow artifact and verified before evidence can be reused. Searchable MLflow tags mirror
the code, lock, and package digests without becoming a second source of truth.

Schema 1 evidence remains readable. New records use schema 2. A project without `uv.lock`
records a missing lock rather than inventing one; the legacy reproduce script correspondingly
uses `uv sync` instead of the unsupported `uv sync --locked`. Shell commands are rendered
with Python's standard `shlex` quoting.

## Consequences

Every project-owned MLflow Attempt gets the same execution identity without adding fields
to project task interfaces. Dirty experimentation remains allowed but reconstructible.
Moving to different code, dependencies, DSIO bytes, commands, or runtime environments
produces different evidence identity. Existing schema 1 Runs remain reusable, while schema
2 Runs fail closed if their execution tags or dirty patch disagree with the hashed record.
