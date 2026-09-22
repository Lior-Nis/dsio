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
3. Every dataset item and produced batch is a mapping containing `sample_id`; DSIO verifies item identity, exact post-collation identity order, batch-field cardinality, and CPU placement. A configured custom collator is governed trusted code for the semantic association of transformed payload/target values, because an arbitrary callable can lie while returning structurally valid output.
4. The same validated inputs, seed, and deterministic components produce the same sample membership and order for supported worker counts. Worker-unsafe datasets or collators fail during setup with an actionable error when multiprocessing is requested.
5. Missing phase mappings or roles, invalid folds or manifests, incompatible factory output, malformed/empty batches, invalid loader arguments, and identity-changing collators fail at the narrowest boundary without substituting a default path.
6. CPU loading performs no accelerator-side or stochastic training augmentation. `DsioDataModule` does not customize `on_after_batch_transfer`; accelerator augmentation remains a later `DsioModule.training_step()` concern.
7. `DsioDataModule` is the one public Lightning data-module class, existing consumers remain functional, and root `import dsio` remains inert.

## Tasks / Subtasks

- [x] Add the cohesive `dsio.data.loading` package (AC: 2-7)
  - [x] Add one identity-checking dataset wrapper and one canonical stored-sample factory.
  - [x] Add one identity-preserving collator and one deterministic DataLoader builder.
  - [x] Add the exact `DsioDataModule` composition root and public exports.
- [x] Replay governed assignments into Lightning phases (AC: 1, 3, 5)
  - [x] Validate the manifest against concrete examples and resolve the declared fold by index.
  - [x] Map only explicit `train`, `validate`, `test`, and `predict` phases to named roles.
  - [x] Build only the datasets requested by Lightning's setup stage and refuse missing loaders.
- [x] Preserve identity and deterministic CPU behavior (AC: 2-6)
  - [x] Require factory length/order to match the exact role assignment and check each item at access time.
  - [x] Preserve ordered `sample_id` values through default and custom collation.
  - [x] Seed shuffle/workers explicitly and preflight multiprocessing picklability.
- [ ] Verify and review (AC: 1-7)
  - [x] Cover phase/role replay, identity alignment, custom collation, deterministic worker counts, and exact class behavior.
  - [x] Cover invalid roles/stages/folds, malformed items/batches, worker-unsafe components, and loader arguments.
  - [x] Run full pytest, Ruff, mypy, import contracts, lock validation, build, and diff checks.
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
- DSIO can prove identity order, field cardinality, container safety, and CPU placement. It cannot infer whether arbitrary transformed tensor values still mean what their IDs claim; custom collators therefore enter through the same reviewed component-governance boundary as other executable components in Story 3.2.

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
- 2026-09-22: Implemented `dsio.data.loading` as four cohesive modules: dataset identity, collation, native loader construction, and the exact Lightning data composition root.
- 2026-09-22: Self-audit removed `drop_last` because it contradicts exact assignment membership, bound the canonical factory to store content identity, and rejected non-CPU collated tensors.
- 2026-09-22: Candidate gate passed: 804 tests passed (3 deselected), Ruff, mypy, all three import contracts, lock validation, build, and diff checks.
- 2026-09-22: Independent review exposed a topology-blind corpus digest, cross-epoch worker-count shuffle drift, unchecked batch cardinality/containers, `IterableDataset` admission, and lax runtime option validation; each was reproduced and fixed.
- 2026-09-22: Clarified the only unprovable boundary: a governed custom collator is trusted executable code for the semantics of same-shaped transformed values, while DSIO proves identity order, cardinality, container safety, and CPU placement.

### Completion Notes List

- `DsioDataModule` validates the concrete governed split and maps explicit Lightning phases to exact named role assignments without a phase/result model.
- The injected factory remains modality-specific, while `IdentityDataset` makes its length and per-position identity checkable before any batch reaches a model.
- Native default or custom collation is wrapped by one identity and CPU-device guard; `sample_id` remains aligned with all payload/target fields.
- Loader construction is seeded, exact-membership (`drop_last=False`), and preflights dataset/collator picklability when workers are requested.
- Canonical corpus identity now covers full signal, index, and entity-metadata digests, so changing ID-to-row topology invalidates examples, splits, loaders, and recorded fold evidence even when payload bytes are identical.
- No registry, loader configuration hierarchy, accelerator augmentation hook, or second split abstraction was added.

### File List

- `_bmad-output/implementation-artifacts/2-5-construct-identity-preserving-dataloaders.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `src/dsio/data/loading/__init__.py`
- `src/dsio/data/loading/collation.py`
- `src/dsio/data/loading/datasets.py`
- `src/dsio/data/loading/loaders.py`
- `src/dsio/data/loading/module.py`
- `tests/data/loading/test_data_module.py`

### Change Log

- 2026-09-22: Created Story 2.5 and started implementation.
- 2026-09-22: Added the identity-preserving `DsioDataModule` loading path and passed the complete local quality gate.
