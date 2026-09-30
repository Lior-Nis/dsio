# Story 6.3: One batch-field vocabulary with consistent mask polarity

Status: done

## Story

As a consumer composing blocks,
I want every DSio component to use the same batch-field names, one mask polarity and one weighting rule,
so that datasets, collators, objectives and evaluation compose without adapters.

## Acceptance Criteria

1. Given the published conventions (`docs/component-warehouse/conventions.md`, sourced from the spec's `conventions.md`), when a consumer reads them, then they define:
   - the batch fields (with `x` as the only model input);
   - `mask` (True = valid, objectives and evaluation only);
   - `hidden`;
   - declared validity and presence channels of `x`;
   - the weighting rule;
   - layouts and target conventions.
2. Given the masking strategies and `MaskedReconstruction`, when they emit a corruption tensor into a batch, then it is named `hidden` (True = hidden), and no batch field named `mask` ever carries True = hidden. A test guards that polarity at every DSio producer and consumer of `mask`.
3. Given the self-supervised reference flow, when it replays after the rename, then its contract-tier metrics are unchanged.

## Tasks / Subtasks

- [x] Task 1: Publish `docs/component-warehouse/conventions.md`, stating which rules are enforced now and which arrive with Stories 7.1, 7.4 and 9.3 (AC 1).
- [x] Task 2: Rename the True-means-hidden tensor to `hidden` (AC 2).
  - [x] 2.1 `dsio.model.masking`: strategies build and return `hidden`, and `apply_mask(x, hidden, value)` takes `hidden`. The module docstring states the polarity against batch `mask`, and the stale "a new mode is a decorator" claim is corrected.
  - [x] 2.2 `dsio.train.augmentation.MaskedReconstruction(strategy, ...)` replaces the ambiguous `mask` constructor parameter with `strategy` (type `HiddenStrategy`). It documents that it keeps `hidden` internal and emits no batch `mask`.
- [x] Task 3: Polarity guards (AC 2).
  - [x] 3.1 **Consumer.** `tests/eval/test_execution.py::test_mask_true_marks_the_scored_positions`: predictions are wrong exactly where `mask` is False, so accuracy is 1.0 only under True = valid (0.0 if inverted). The existing mask test used predictions equal to targets and could not detect inversion.
  - [x] 3.2 **Producer.** `tests/train/test_augmentation.py::test_masked_reconstruction_never_emits_or_overwrites_a_batch_mask`: output keys equal input keys, a consumer's True = valid `mask` passes through as the same object, and no `hidden` field leaks. No other DSio component emits a batch `mask` today; the Story 8.1 pad collator will be the next producer and must add its own guard.
- [x] Task 4: Self-supervised reference replay unchanged (AC 3). `tests/reference_flows/`: 16 passed, with no expected-value edits. Caveat from review: that reference uses only `TwoView`, so no reference flow exercises the masking code. The rename is covered by the unit suites below instead.
- [x] Task 5: Review follow-ups.
  - [x] 5.1 The published and spec conventions now match reality:
    - `row` and `view_id` are documented;
    - the `mask` shape matches `evaluate`'s exact-shape rule;
    - `hidden` is added to the reserved table;
    - the dropped spec rules are restored;
    - every rule not yet enforced is marked with the story that enforces it.
  - [x] 5.2 `dsio.batches.BatchInputs` gains `mask`, `sample_weight` and `group` as optional fields, so the static contract matches the vocabulary.
  - [x] 5.3 `test_masked_reconstruction_hides_exactly_what_the_strategy_hides` guards the internal polarity: `CausalMask` hides the tail, which is zeroed in `x` and is the only finite part of `y`. The review showed that swapping `hidden` and `~hidden` previously went unnoticed by the main test.
  - [x] 5.4 `tests/model/test_masking.py` names the tensor `hidden`. `MaskedMSE`'s docstring credits `MaskedReconstruction`, not `WindowDataset`, and its error reads "no hidden positions".

## Dev Notes

- **Finding.** No DSio component emitted a batch field named `mask`. The conflict was naming: the masking API called a True-means-hidden tensor `mask`, while `dsio.eval` uses `mask` for True = valid. The rename is API-only. `apply_mask`'s positional use is unchanged, and `MaskedReconstruction` is constructed positionally everywhere in the repo (pre-1.0, no consumers).
- These components move to `dsio.experimental` as legacy in Story 6.5.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

- Gates: `ruff check`, `ruff format --check`, `mypy` (97 files) and `lint-imports` are clean. `tests/model`, augmentation and evaluation pass, 159 tests. `tests/reference_flows` passes, 16 tests.
- The adversarial review found nothing critical and confirmed the rename complete. It verified both polarity tests fail under inversion (accuracy 0.0; ignoring the mask gives 0.5). Its important findings (missing `row`/`view_id`, wrong `mask` shape wording) and all minor findings are fixed (Task 5).

### File List

- `docs/component-warehouse/conventions.md` (new)
- `src/dsio/model/masking.py`
- `src/dsio/train/augmentation.py`
- `tests/eval/test_execution.py`
- `tests/train/test_augmentation.py`
- `tests/model/test_masking.py`
- `tests/model/test_components.py`
- `src/dsio/batches.py`
- `src/dsio/model/components.py`
- `_bmad-output/specs/spec-component-warehouse/conventions.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
