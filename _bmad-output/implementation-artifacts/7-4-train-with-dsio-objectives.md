# Story 7.4: Train with DSio objectives

Status: review

## Story

As a consumer choosing a loss,
I want DSio objectives over any native loss, with auxiliary metrics, target adaptation, stage restriction, native class weights and mean-1 sample weights,
so that I never write an `Objective` (cohort #18, #20).

## Acceptance Criteria

1. Given a supervised objective configured with a native loss and its native parameters (e.g. `weight=` class weights), target dtype/shape adaptation, and named auxiliary metrics, when it runs on a batch, then it returns `loss` and each metric as a scalar, and a `[B,1]` target is never silently squeezed against a `[B]` prediction.
2. Given sample weights with training-role mean 1, when the same samples are split into different micro-batch partitions, then the optimized loss and logged weighted metrics are identical (micro-batch invariance test), and per-micro-batch `/sum(w)` normalization is absent.
3. Given a reconstruction objective, when the target is declared as the input, then the loss is computed against `x` without a consumer objective.
4. Given titanic, bike_sharing, store_sales and digit_recognizer (classification and reconstruction), and the supervised fixture, when they are migrated, then every local objective is deleted, logged metric names are unchanged, and contract tiers match the golden metrics. The self_supervised fixture may use the objective over legacy `NTXent`, or keep fixture code.

## Tasks / Subtasks

- [x] Task 1: `dsio.experimental.model.objectives:SupervisedObjective(loss, target="y", target_dtype=None, target_shape=None, metrics=None, metric_stages=None, sample_weighted=False)` (AC 1-3).
  - **The loss** is a component config of a native loss module. Its parameters keep native semantics: list-valued `weight` and `pos_weight` become float32 tensors, registered as the loss's buffers so they move with the module. `reduction` belongs to the objective and is refused in config.
  - **The target** is a declared batch field (`x` makes the reconstruction objective). An optional `target_dtype` cast and per-sample `target_shape` reshape are applied, then a strict shape rule:
    - a floating target must equal the prediction's shape;
    - a class-index target must equal it without the class dimension;
    - otherwise the error names both shapes and points to `target_shape`.
  - **Metrics** are named component configs of `(prediction, target)` modules, computed on the detached prediction. `metric_stages` restricts them to some stages, and `loss` is not a valid metric name.
  - **Sample weights.** With `sample_weighted`, the loss and metrics are built with `reduction="none"` and reduced as `mean(w * values)`. Weights must be finite, non-negative and one per sample, and are never renormalized per micro-batch. A metric that cannot reduce elementwise is refused by name.
- [x] Task 2: `RootMeanSquaredError`, `sqrt(mse)` as an auxiliary metric. It is Store Sales' RMSLE on log1p targets.
- [x] Task 3: Tests in `tests/experimental/model/test_objectives.py`:
  - **Parity:** results match bit for bit the MSE+MAE, MSE+RMSLE, cross-entropy, BCE-with-logits and reconstruction consumer objectives.
  - **Class weights:** native class-weight semantics, with weights stored as buffers.
  - **No silent squeeze:** a `[B,1]` target against a `[B]` prediction fails, and an explicit `target_shape=[]` adapts a class column.
  - **Micro-batch invariance:** for the loss and the weighted MAE across partitions (4/4, 2/6, 3/5, 1/7), combined as `DsioModule` logs them (batch-size-weighted epoch mean). A contrasting check shows `/sum(w)` normalization would break it.
  - **Stages:** metrics follow the configured stages.
  - **Validation:** configuration and batch errors are checked.
- [x] Task 4: Migrations (AC 4), with goldens unchanged and logged names (`loss`, `mae`, `rmsle`) unchanged:
  - **titanic:** `PassengerObjective` → BCE-with-logits.
  - **bike_sharing:** `DemandObjective` → MSE with `mae`.
  - **store_sales:** `ForecastObjective` → MSE with `rmsle`.
  - **digit_recognizer:** `ReconstructionObjective` → MSE with `target: x`, and `ClassificationObjective` → cross-entropy.
  - **supervised fixture:** `RegressionObjective` → MSE with `mae`.
  - **Provenance** records each objective's full config instead of a class reference.
  - **self_supervised** keeps its contrastive objective as fixture code (AC 4 allows this).
- [x] Task 5: Catalog.
  - Both blocks have full sections and doctested Examples, plus evidence with real uses and the owner's standing approval: `SupervisedObjective` (4 Kaggle consumers, 1 fixture) and `RootMeanSquaredError` (Store Sales).
  - "Local objectives" in the candidates register drops the four migrated consumers.
  - Legacy `LossObjective` says `SupervisedObjective` supersedes it.
  - The `dsio.experimental.model` package docstring no longer calls every block unproven.
- [x] Task 6: Review follow-ups. The adversarial review found nothing critical. It confirmed bit parity, values and parameter gradients for all six configs. It also confirmed unchanged goldens and metric names, class-weight buffers following the module, and deterministic provenance.
  - [x] 6.1 (important) The shape rule no longer lets a non-class loss broadcast. Only `CrossEntropyLoss`, `NLLLoss` and `MultiMarginLoss` drop the class axis for integer targets, so an int64 `[B]` count target against a `[B,1]` Poisson prediction fails. Same-shape integer targets (`MultiLabelMarginLoss`) are accepted.
  - [x] 6.2 (important) Under `sample_weighted`, class weights scale per-sample losses (`mean(w_i · w_c[y_i] · l_i)`, the CMI sequence semantics) rather than taking the native weighted mean. This is documented and tested. Targets equal to the loss's `ignore_index` are refused, because they would still count in the mean.
  - [x] 6.3 (important) Stateful TorchMetrics are refused as metrics. Called per batch, their state grew without reset and was shared across stages. A per-stage `update`/log path waits for a consumer that needs it.
  - [x] 6.4 (important) The invariance claim is now precise:
    - `DsioModule`'s logged epoch values (batch-size-weighted means) are partition-invariant.
    - Accumulated gradients equal the full batch for equal-size micro-batches. A new gradient test shows this. Lightning averages accumulated micro-batches equally, so uneven partitions weight them differently.
    - Epoch values are logged per process.
  - [x] 6.5 Legacy `size_average` and `reduce` are owned like `reduction`.
  - [x] 6.6 Metric names must be identifiers, not `loss`/`loss_step`/`loss_epoch`, and not a `ModuleDict` attribute.
  - [x] 6.7 Scalar `pos_weight` works.
  - [x] 6.8 Sample weights follow the loss values' dtype, and must be `[batch]` or `[batch, 1]`.
  - [x] 6.9 A non-tensor model output raises a clear error.

## Dev Notes

- **Parity reasoning.**
  - `nn.MSELoss`, `nn.L1Loss`, `nn.CrossEntropyLoss` and `nn.BCEWithLogitsLoss` call the same functional ops with the same defaults as the deleted objectives.
  - The casts are no-ops on the already-typed dataset fields.
  - Store's RMSLE was `sqrt(loss.detach())`; it is now `sqrt(mse)` over the same operands, which gives the same value.
  - No objective draws randomness, so construction order is irrelevant.
- **Why native reduction without sample weights.** `CrossEntropyLoss(weight=...)` takes a class-weighted mean (divided by the sum of target-class weights), and a plain `nll` mean can sum in a different order than `reduction="none"` followed by `.mean()`. Keeping the native reduction preserves both semantics and bits.
- **Why `mean(w * l)`.**
  - `DsioModule` logs each micro-batch value weighted by its batch size. With mean-1 weights, the batch-size-weighted mean of the per-batch `mean(w * l)` values equals the full-batch value for any partition. Per-batch `/sum(w)` does not.
  - For optimization, accumulated equal-size micro-batches reproduce the full-batch gradient.
  - AC 2's "optimized loss identical" holds for equal-size micro-batches, which is the partition DSio's loaders produce apart from a final partial batch.
- **What stays local.**
  - The masked dense objectives of FoG and ROGII (Story 8.2).
  - The objectives of Essay and both CMI consumers, which move with their sequence and weighting stories.
  - The self-supervised fixture's contrastive objective.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

- Gates: ruff, format, mypy (113 files), import contracts and the catalog check pass. The objective tests (8, plus doctests), the four Kaggle consumers' contract tests and the reference flows pass.

### File List

- `src/dsio/experimental/model/objectives.py` (new), `src/dsio/experimental/model/__init__.py`, `src/dsio/experimental/model/chain.py` (superseded note)
- `tests/experimental/model/test_objectives.py` (new)
- `reference_projects/kaggle/{titanic,bike_sharing,store_sales,digit_recognizer}/{components.py,tasks/training.py}`
- `reference_projects/supervised/{components.py,tasks/training.py}`, `tests/reference_flows/test_supervised_flow.py`
- `docs/component-warehouse/{catalog.md,evidence.yaml,candidates.yaml}`, `_bmad-output/implementation-artifacts/sprint-status.yaml`
