---
baseline_commit: 760c277bd3a053a6c6432aa06d92b8d13279710c
---

# Story 8.4: Compose token-sequence models with masked pooling

Status: done

## Story

As a consumer of token sequences,
I want a token embedding encoder with masked mean pooling that reads validity from a declared `x` channel,
so that padded tokens never influence the representation (cohort #12).

## Acceptance Criteria

1. Given time-major token input `[B,T,C]` with declared token and validity channels, when padding token values change—including to negative or out-of-vocabulary values—then the pooled representation and composed model output are unchanged.
2. Given valid token sequences, when the encoder runs, then it returns the arithmetic mean of valid embeddings only as `[B,E]`; token zero may be valid when the validity channel says so.
3. Given empty/all-invalid sequences, invalid rank or channel declarations, non-binary validity, non-integral token input, or an observed token outside the vocabulary, then the encoder raises a named contract error instead of coercing, producing NaN, or relying on backend failures.
4. Given canonical `ComponentConfig`, when the encoder is resolved and composed with the existing `Chain` and a native linear head, then construction is deterministic, importable, graph-compatible, and produces the expected sequence-level output.
5. Given a configured token-sequence Predictor, when it is exported, reloaded through MLflow, and invoked on the same input, then local and loaded predictions are exactly equal.
6. Given `essay_scoring`, when migrated, then `EssayRegressor` and `EssayObjective` are deleted, its model and objective are complete component configurations, its tokenizer remains consumer-local and registered as a candidate, and its contract tier matches the unchanged golden metrics.
7. Given warehouse admission rules, when the encoder is added, then its real use is registered, superseded candidate entries shrink, the catalog is regenerated, and admission and repository gates pass.

## Predeclared Computation Change and Tolerance

- There is no declared computation change. The encoder preserves the former construction order (`Embedding` then `Linear`), zero initialization of the safe row, and masked-mean arithmetic while allowing that row to learn when structurally valid.
- Contract metrics use the committed golden harness tolerance (`rel_tol=1e-6`, `abs_tol=1e-9`). `tests/golden_metrics.json` must remain byte-for-byte unchanged. No wider tolerance may be selected after observing results.
- The expected golden SHA-256 is `71371a90d59bb9f51c5d2d89c2acb2e6a6b81cce992ba550e2e6caa0362dc530`.

## Tasks / Subtasks

- [x] Task 1: Specify the masked token contract with failing tests (AC 1-5).
  - [x] Prove exact masked-mean arithmetic, padding-value invariance including invalid/OOV padding values, valid token zero, deterministic seeded composition, and `[B,E]`/`[B,K]` shapes.
  - [x] Prove named failures for rank, channel, dtype, validity, vocabulary, empty-time, and all-invalid inputs without tensor-to-Python synchronization.
  - [x] Prove canonical config resolution, full-graph compilation, and exact Predictor/MLflow round-trip equality.
- [x] Task 2: Implement the smallest focused token encoder (AC 1-4).
  - [x] Move and reshape the legacy `EmbeddingEncoder` into a focused token module; do not retain a compatibility shim or parallel unmasked encoder.
  - [x] Replace invalid token positions before lookup, then mean only valid embeddings; do not infer validity from token values.
  - [x] Compose with existing `Chain` and native `Linear`; add no base class, registry, task mode, result/config dataclass, or dependency.
- [x] Task 3: Migrate `essay_scoring` (AC 5-6).
  - [x] Declare complete `MODEL` and `OBJECTIVE` component configurations and resolve them for training, export, evaluation, and inference.
  - [x] Delete `EssayRegressor` and `EssayObjective` while preserving seeds, construction order, objective semantics, prediction output, validation, dynamic axes, ordering, and submission bytes.
  - [x] Record the full model and objective configurations in provenance and assert exact local-versus-MLflow configured-model output equality.
- [x] Task 4: Govern and measure the component (AC 7).
  - [x] Register the real Essay use, remove legacy encoder evidence/catalog handling and superseded local-model/objective candidates, retain the tokenizer candidate, and regenerate the catalog.
  - [x] Run admission checks, record owner standing approval, and complete an adversarial review before merge.
  - [x] Measure Essay from the Story 8.3 baseline of 539 non-ingestion lines and repository banned definitions from 22; record the resulting deltas.
- [x] Task 5: Verify the migration (AC 1-7).
  - [x] Run focused encoder and Essay contract tests, full pytest, Ruff check/format, mypy, import contracts, catalog/admission, wheel/sdist build, and distribution/consumer-flow contracts.
  - [x] Verify `tests/golden_metrics.json` remains byte-for-byte unchanged at the predeclared digest.

### Review Findings

- [x] [Review][Patch] Keep the structurally valid safe-row token learnable while preserving its zero initialization [src/dsio/experimental/model/tokens.py]
- [x] [Review][Patch] Convert every accepted integer dtype to int64 before the embedding lookup [src/dsio/experimental/model/tokens.py]
- [x] [Review][Patch] Reject numeric non-integer channel indices during construction [src/dsio/experimental/model/tokens.py]
- [x] [Review][Patch] Accumulate half-precision masked means in float32 [src/dsio/experimental/model/tokens.py]
- [x] [Review][Patch] Use one asynchronous device assertion instead of per-step host synchronizations [src/dsio/experimental/model/tokens.py]
- [x] [Review][Patch] Prove the composed Chain is padding-invariant and full-graph compilable [tests/experimental/model/test_tokens.py]
- [x] [Review][Patch] Require exact output-key parity in the configured local-versus-MLflow comparison [tests/kaggle_portfolio/test_essay_scoring.py]
- [x] [Review][Patch] Record the full-dataset browser-visible MLflow training run for every new real use [docs/component-warehouse/evidence.yaml]
- [x] [Review][Patch] Remove completed Story 8.4 references from the remaining local-model candidate [docs/component-warehouse/candidates.yaml]

## Dev Notes

- `x` remains the sole model input. Token validity is structural data in one declared channel; batch `mask` remains reserved for objective/evaluation semantics.
- Invalid positions are replaced with `safe_token_id` before embedding and vocabulary validation, so arbitrary padding values cannot reach the lookup table. Observed token IDs are validated on every call with a graph-compatible asynchronous tensor assertion.
- The encoder owns embedding plus masked pooling only and returns `[B,E]`. The task head remains a separate native `Linear` inside `Chain`; no hidden projection belongs in the encoder.
- Require an integer input tensor rather than silently truncating floats. Validate that validity is exactly zero or one and that every sample contains at least one observed token.
- Keep the tokenizer, ordinal output, and ordinal validator consumer-local. Story 8.6 owns output/validator migration; tokenizer generalization remains a candidate rather than part of this story.
- Migrate the local objective now because `SupervisedObjective` already expresses exact cross-entropy semantics and the component-warehouse candidate register assigns the Essay objective migration to the sequence stories.
- Keep the new public block experimental with one real use. Promotion belongs to Story 11.1.

### Previous Story Intelligence

- Story 8.1 established the Essay validity channel inside `x`, configured `PadCollator`, inputs-only inference data, and dynamic MLflow axes. Reuse those contracts unchanged.
- Story 8.2 established that `x` is the only model input and masks outside `x` are objective/evaluation-only.
- Story 8.3 showed that validity must remain structural, validation must remain graph-compatible, and configured local-versus-MLflow prediction equality needs a direct assertion.
- Story 7.3 established `Chain` and deterministic construction order; Story 7.4 established `SupervisedObjective`. Reuse both rather than introducing token-specific training abstractions.

### Project Structure Notes

- Put the encoder in `src/dsio/experimental/model/tokens.py`; remove it from the oversized legacy `components.py` and update exports, evidence, and generated catalog together.
- Expected consumer changes are limited to Essay components, training/downstream configuration, tests, and provenance assertions. Do not touch deployment, orchestration, task lifecycle, or Predictor input contracts.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Story 8.4]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — CAP-1 and constraints]
- [Source: `_bmad-output/specs/spec-component-warehouse/component-cohort.md` — block #12]
- [Source: `_bmad-output/implementation-artifacts/8-1-pad-variable-length-batches-with-a-dsio-collator.md`]
- [Source: `_bmad-output/implementation-artifacts/8-3-compose-dense-signal-models-with-layout-adapters-and-per-instance-standardization.md`]
- [Source: `src/dsio/experimental/model/{components.py,compositions.py,objectives.py}`]
- [Source: `reference_projects/kaggle/essay_scoring/components.py`]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- Red phase: the focused suite failed because the new token module did not exist; review regressions then exposed a frozen valid safe row, unsupported integer lookup dtypes, synchronous device validation, and half-precision reduction overflow.
- Real-data parity run: full 17,307-row Kaggle training, export, evaluation, and inference completed in MLflow experiment 55; training run `421f6f2834d64d11ba569bcbfd5e0a37`.
- Final verification: 1,278 passed, 1 deselected; 110 focused catalog/token/admission tests; 8 distribution/consumer-flow contracts.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Replaced the legacy unmasked/projection encoder with one focused time-major masked embedding mean; invalid positions are sanitized before lookup and never affect values or gradients.
- Preserved seeded Essay initialization and logits while keeping a structurally valid token zero learnable.
- Migrated Essay model and cross-entropy objective to complete `Chain`, `EmbeddingEncoder`, native `Linear`, and `SupervisedObjective` configurations; deleted both local classes.
- Added exact configured local-versus-MLflow output parity, full-graph composition, arbitrary invalid/OOV padding, integer-dtype, low-precision, gradient, and failure-contract coverage.
- Three independent review layers produced nine actionable findings; all were patched. Four compatibility or out-of-scope findings were dismissed under the explicit pre-release/no-backwards-compatibility and non-TorchScript scope.
- Essay non-ingestion code fell from 539 to 527 lines (-12); repository banned local definitions fell from 22 to 20 (-2).
- The full real Kaggle run recorded accuracy `0.44929797191887677` and quadratic-weighted kappa `0.47384293867867033`; the fixture contract golden remained unchanged.
- Verification passed: full pytest, Ruff, format, mypy over 119 source files, three import contracts, catalog/admission, wheel/sdist build, and distribution/consumer-flow contracts.
- `tests/golden_metrics.json` remained unchanged at SHA-256 `71371a90d59bb9f51c5d2d89c2acb2e6a6b81cce992ba550e2e6caa0362dc530`.

### File List

- `_bmad-output/implementation-artifacts/8-3-compose-dense-signal-models-with-layout-adapters-and-per-instance-standardization.md`
- `_bmad-output/implementation-artifacts/8-4-compose-token-sequence-models-with-masked-pooling.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/evidence.yaml`
- `reference_projects/kaggle/essay_scoring/components.py`
- `reference_projects/kaggle/essay_scoring/tasks/training.py`
- `src/dsio/experimental/model/__init__.py`
- `src/dsio/experimental/model/components.py`
- `src/dsio/experimental/model/tokens.py`
- `tests/experimental/model/test_tokens.py`
- `tests/kaggle_portfolio/test_essay_scoring.py`
- `tests/model/test_components.py`
- `tests/model/test_token_corpus.py`
- `tests/train/test_capabilities.py`
- `tools/catalog.py`

## Change Log

- 2026-10-03: Created Story 8.4 with arbitrary-padding-value invariance and no declared computation change.
- 2026-10-03: Implemented, migrated, adversarially reviewed, battle-tested on the full Essay dataset, and verified Story 8.4.
