# 26. Window datasets belong to data loading

Date: 2026-09-25

## Decision

Move `WindowDataset` from the separate `dsio.dataset` namespace into
`dsio.data.loading.windows`. Export it with `DsioDataModule` and `build_loader` from
`dsio.data.loading`.

Delete the train/validation dataset aliases and the specialized loader wrappers. A
`WindowDataset` does not change by phase; train-only augmentation belongs to `DsioModule`.
All datasets use the same validated `build_loader` implementation.

## Why

The extra top-level package made one data-loading concern look like a second subsystem and
kept two loader constructors alive. The aliases added names without adding invariants: each
called the same constructor with the same arguments. This contradicted the agreed layout,
where Torch dataset implementations live beside the data module and loader boundary.

## Consequences

- Windowed signal, text, and fixed-size-item datasets have one canonical import path.
- Worker seeding, identity-preserving collation, and loader validation have one owner.
- Dataset construction remains phase-neutral and modality-neutral.
- Project-owned flows still train only through the reusable DSIO Lightning spine.
