---
status: accepted
date: 2026-09-25
supersedes: ADR-0010, ADR-0017
depends-on: ADR-0019
---

# Project flows are the only training entrypoints

ADR 0019 moved orchestration into project-owned Prefect flows, but the earlier fixed
runner system remained importable. `RunConfig`, `TorchTask`, `SslPretrainTask`, their
registries, and the local `Run` lifecycle duplicated configuration assembly, MLflow
lifecycle, and artifact upload beside the new flow path. Maintaining both contradicted
the single-path training decision and made the old architecture look supported.

The fixed runners and their lifecycle are removed. A training task now composes the
shared `DsioModule`, `DsioDataModule`, `TrainerConfig`, callbacks, capability checks,
artifact references, and tracking evidence directly inside its project-owned flow. DSIO
owns those reusable components and contracts; the project owns task ordering and
lifecycle.

Shared MLflow URI handling moved from the deleted runner tracking module to
`dsio.tracking.client`, beside the native MLflow attempt and evidence APIs. There is no
runner registry, task-kind registry, configuration root, or compatibility loader.

## Consequences

There is one supported training architecture and no silent fallback to the retired
fold-as-process runner. Projects can express different DAGs without asking DSIO to model
their lifecycle, while every actual fit still passes through DSIO's battle-tested
Lightning module, data module, trainer construction, and provenance boundary.
