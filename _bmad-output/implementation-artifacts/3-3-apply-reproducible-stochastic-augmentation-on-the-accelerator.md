---
baseline_commit: 32c42dea276a53ad0a32b8e6174f87f948e131bf
---

# Story 3.3: Apply Reproducible Stochastic Augmentation on the Accelerator

Status: review

## Story

As a training-task author,
I want stochastic training augmentation applied after device transfer,
so that augmentation is fast while remaining exactly replayable.

## Acceptance Criteria

1. A configured accelerator augmentation executes inside `DsioModule.training_step()` after Lightning transfers the batch and before the objective/model; no stochastic training work remains in datasets, workers, or collation.
2. Augmentation randomness is derived from the execution seed, epoch, global step, ordered source `sample_id` values, augmentation identity, and view identity. Replaying those inputs reproduces the result independently of worker scheduling; changing seed or view changes the deterministic stream.
3. Validation, test, prediction, and deterministic preprocessing never invoke stochastic training augmentation.
4. Multi-view training creates declared, reproducible views from the same source batch while preserving the source `sample_id` association for every view.
5. Existing supervised, masked-reconstruction, contrastive, provenance, and root-import contracts remain intact.

## Tasks / Subtasks

- [x] Add one training-only augmentation seam to `DsioModule` (AC: 1-3)
  - [x] Accept one optional native batch augmentation and execution seed.
  - [x] Invoke it only from `training_step()`, before the existing objective boundary.
  - [x] Record its exact component identity in Lightning hyperparameters without adding a registry or result model.
- [x] Implement reproducible accelerator batch augmentations (AC: 2, 4)
  - [x] Derive device-local `torch.Generator` instances from canonical execution and batch identity without mutating global RNG state.
  - [x] Provide masked-reconstruction and two-view adapters that preserve the current native loss contracts.
  - [x] Carry source `sample_id` and stable view identity through multi-view batches.
- [x] Move SSL stochastic work out of the data layer (AC: 1, 3-5)
  - [x] Make SSL workers and collation return raw windows only.
  - [x] Remove dataset masking and two-view collation paths rather than retaining duplicate compatibility lanes.
  - [x] Do not manufacture a stochastic pretext validation loss; run representation-quality callbacks from the training lifecycle over their separate deterministic loaders.
- [x] Prove replay, lifecycle isolation, migration, and compatibility through tests and full quality gates (AC: 1-5)

## Dev Notes

### Minimal shape

- Add one optional callable module to `DsioModule`; do not add an augmentation registry, context dataclass, result dataclass, custom RNG wrapper, or objective hierarchy.
- Keep the existing mutually exclusive `mask` / `augmentor` task configuration. Task assembly privately adapts either native component to the one batch augmentation seam.
- Keep `LossObjective`, `MaskedMSE`, `NTXent`, and `VICReg` unchanged by producing their existing `x` / `y` batch shapes.
- Use fixed, explicit two-view identities for this proven two-view contract. General arbitrary-view composition is not required by this story.

### Determinism and ownership

- Hash canonical seed material into a valid PyTorch seed, then create a fresh generator on the transferred tensor's device for each logical view.
- Pass generators explicitly through built-in stochastic components. Do not seed or fork the process-global PyTorch RNG.
- Ordered batch membership is deliberately part of execution identity: worker count cannot change a collated batch, while changing batching or distributed topology may change the stochastic stream and therefore belongs in provenance.
- Lightning-restored epoch and global step plus stateless derivation provide checkpoint replay without another checkpointed RNG object.
- `training_step()` owns stochastic train-only augmentation. Validation/test/predict/forward and the data module do not provide alternate hooks.

### Validation behavior

- A held-out stochastic pretext loss would violate the training-only lane, so SSL does not create an augmented validation loader in this iteration.
- Online probe and RankMe remain deterministic evaluation consumers of their existing raw loaders and run at train-epoch end.
- SSL checkpointing therefore uses the existing no-validation behavior rather than inventing a replacement monitored result.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 3.3, FR18]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Training system and execution identity]
- [Source: `src/dsio/model/module.py`]
- [Source: `src/dsio/train/ssl_task.py`]
- [PyTorch Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html)
- [Lightning training loop](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop)

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 3.2 at `32c42de`; compared three interface designs and selected one training-step seam with explicit device-local generators and no new objective abstraction.
- 2026-09-22: The complete repository suite passed with 855 tests and 3 live tests deselected; affected augmentation, SSL, callback, dataset, component, and module suites passed 120 tests.
- 2026-09-22: First independent review blocked the candidate on retained identity-augmentor compatibility, in-place aliasing, length-one normalization, malformed metadata boundaries, test-split RankMe use, and silently accepted validation-dependent controls. All findings were reproduced and fixed test-first; 128 affected tests passed.

### Completion Notes List

- `DsioModule.training_step()` is the sole stochastic training hook. It accepts one native, non-learnable augmentation module and records canonical seed/component identity.
- Masked reconstruction and two-view contrastive training now run on the transferred batch with explicit device-local generators; no process-global RNG state is modified.
- SSL loaders return raw identity-bearing windows. Dataset masking, seeded mask state, and `TwoViewCollate` were deleted rather than retained as a second path.
- SSL no longer manufactures a stochastic pretext validation result. OnlineProbe and RankMe run from train-epoch end over their own deterministic raw loaders.
- Existing MAE, SimCLR, VICReg, supervised training, checkpoint, MLflow, and encoder handoff flows remain green.
- Review hardening isolates in-place augmentors, preserves dtype/row/view contracts, normalizes canonicalization failures, keeps unlabeled RankMe on training evidence, and rejects SSL early stopping or direct plateau scheduling without a validation objective.

### File List

- `_bmad-output/implementation-artifacts/3-3-apply-reproducible-stochastic-augmentation-on-the-accelerator.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/adr/0010-lightning-lives-inside-a-fold.md`
- `docs/adr/0011-ssl-is-first-class.md`
- `src/dsio/batches.py`
- `src/dsio/dataset/dataset.py`
- `src/dsio/model/components.py`
- `src/dsio/model/module.py`
- `src/dsio/train/augmentation.py`
- `src/dsio/train/callbacks.py`
- `src/dsio/train/ssl_task.py`
- `src/dsio/train/trainer.py`
- `tests/dataset/test_dataset.py`
- `tests/model/test_components.py`
- `tests/train/test_augmentation.py`
- `tests/train/test_callbacks.py`
- `tests/train/test_ssl_runner.py`
- `tests/typing/batch_contracts.py`

### Change Log

- 2026-09-22: Created Story 3.3 and started implementation.
- 2026-09-22: Moved reproducible stochastic augmentation to the accelerator-side training step and removed the data-layer implementation.
