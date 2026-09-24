# 25. Execution capture belongs to tracking

Date: 2026-09-25

## Decision

Delete the `dsio.runs` package. Code, environment, command, and installed-package capture
live in the `dsio.tracking.execution` package that records and validates that evidence.
DataLoader worker seeding lives in `dsio.data.loading.loaders`, beside the only production
consumer.

Project flows use Lightning's `seed_everything` for process-level training seeds. DSIO does
not retain its unused parallel implementation.

## Why

After project-owned Prefect flows replaced fixed runners, `dsio.runs` owned no coherent
concept. Its provenance functions were private dependencies of tracking, while one seeding
helper configured loaders. The package name implied a run lifecycle abstraction that no
longer existed.

## Consequences

- Tracking evidence capture is split by concern into `capture`, `git`, and `environment`.
- Loader determinism has one implementation at the loading boundary.
- The unused global RNG mutator is deleted rather than maintained as an alternate API.
- Existing schema-2 MLflow provenance documents remain readable and unchanged in shape.
