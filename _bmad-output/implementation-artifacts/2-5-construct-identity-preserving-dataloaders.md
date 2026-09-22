---
baseline_commit: a62680d
---

# Story 2.5: Construct Identity-Preserving DataLoaders

Status: in-progress

## Story

As a training-task author,
I want `DsioDataModule` to construct DataLoaders from a store and split manifest,
so that every training paradigm shares one replayable CPU data path.

## Acceptance Criteria

1. Given a validated store, `Examples`, governed `SplitFile`, fold, and explicit Lightning-phase-to-role mapping, `DsioDataModule.setup()` constructs only the requested train, validation, test, and prediction datasets/loaders, with exactly the stable sample identities assigned to each mapped role.
2. Dataset construction is injected through one ordinary callable boundary while DSIO owns assignment replay, identity checking, deterministic sampling, batching, worker setup, and collation under `dsio.data.loading`; no runtime registry, loader-result model, or parallel split abstraction is added.
3. Every dataset item and produced batch is a mapping containing `sample_id`; DSIO verifies identity alignment before collation and after custom collation so data, targets, and identity cannot silently drift.
4. The same validated inputs, seed, and deterministic components produce the same sample membership and order for supported worker counts. Worker-unsafe datasets or collators fail during setup with an actionable error when multiprocessing is requested.
5. Missing phase mappings or roles, invalid folds or manifests, incompatible factory output, malformed/empty batches, invalid loader arguments, and identity-changing collators fail at the narrowest boundary without substituting a default path.
6. CPU loading performs no accelerator-side or stochastic training augmentation. `DsioDataModule` does not customize `on_after_batch_transfer`; accelerator augmentation remains a later `DsioModule.training_step()` concern.
7. `DsioDataModule` is the one public Lightning data-module class, existing consumers remain functional, and root `import dsio` remains inert.

## Tasks / Subtasks

- [ ] Add the cohesive `dsio.data.loading` package (AC: 2-7)
  - [ ] Add one identity-checking dataset wrapper and one canonical stored-sample factory.
  - [ ] Add one identity-preserving collator and one deterministic DataLoader builder.
  - [ ] Add the exact `DsioDataModule` composition root and public exports.
- [ ] Replay governed assignments into Lightning phases (AC: 1, 3, 5)
  - [ ] Validate the manifest against concrete examples and resolve the declared fold by index.
  - [ ] Map only explicit `train`, `validate`, `test`, and `predict` phases to named roles.
  - [ ] Build only the datasets requested by Lightning's setup stage and refuse missing loaders.
- [ ] Preserve identity and deterministic CPU behavior (AC: 2-6)
  - [ ] Require factory length/order to match the exact role assignment and check each item at access time.
  - [ ] Preserve ordered `sample_id` values through default and custom collation.
  - [ ] Seed shuffle/workers explicitly and preflight multiprocessing picklability.
- [ ] Verify and review (AC: 1-7)
  - [ ] Cover phase/role replay, identity alignment, custom collation, deterministic worker counts, and exact class behavior.
  - [ ] Cover invalid roles/stages/folds, malformed items/batches, worker-unsafe components, and loader arguments.
  - [ ] Run full pytest, Ruff, mypy, import contracts, lock validation, build, and diff checks.
  - [ ] Complete independent blind, edge-case, and acceptance reviews before merge.

## Dev Notes

### Minimal shape

- Use a package now because loading already has four cohesive concerns: datasets, collation, loader construction, and the Lightning composition root.
- Use ordinary callables and native PyTorch `Dataset`/`DataLoader` values. Story 3.2 will govern named importable component configuration; this story does not add a registry or component model.
- Keep phase mapping as a plain mapping and loader options as constructor arguments. Do not add `LoaderConfig`, phase result dataclasses, or a sampler hierarchy.
- The factory owns modality-specific decoding/windowing into per-example mappings. DSIO owns which exact identities it may expose and rejects factories that reorder, omit, duplicate, or replace them.

### Identity boundary

- Manifest assignments are the source of role membership and order.
- A dataset item must expose the expected string `sample_id` at its position. The wrapper checks this lazily so a bad decoder fails where it produces the contradiction.
- The collator derives expected identity from the items, delegates to native `default_collate` or the configured collator, then requires the batch identity to match exactly.

### Lightning mapping

- Public phases are `train`, `validate`, `test`, and `predict`; Lightning setup stages are `fit`, `validate`, `test`, and `predict`.
- `setup("fit")` constructs train and validation when they are mapped; other stages construct only their corresponding phase. `setup(None)` constructs every mapped phase.
- A loader method without an explicit mapping/setup result raises an actionable loading error rather than returning `None` or borrowing another role.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 2.5, FR16, FR17, FR27, FR28]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Training system, augmentation ownership, data and splits]
- [Source: `src/dsio/data/examples.py`]
- [Source: `src/dsio/data/splits/models/`]
- [Source: `src/dsio/data/store/`]
- [Source: `src/dsio/dataset/dataset.py` and `src/dsio/runs/seeding.py` — proven predecessor behavior to reuse or deepen]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 2.4 at `a62680d`; reconciled the generic spine with the canonical store, exact split assignments, existing window loader, and Lightning module contracts.

### Completion Notes List

### File List

- `_bmad-output/implementation-artifacts/2-5-construct-identity-preserving-dataloaders.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

### Change Log

- 2026-09-22: Created Story 2.5 and started implementation.
