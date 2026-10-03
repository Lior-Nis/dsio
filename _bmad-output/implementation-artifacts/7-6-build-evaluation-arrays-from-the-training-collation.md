# Story 7.6: Build evaluation arrays from the training collation

Status: review

## Story

As a consumer evaluating an exported predictor,
I want `dsio.data` to assemble evaluation inputs, targets and masks through the same dataset and collation used for training,
so that I stop re-implementing batching in NumPy (cohort #5).

## Acceptance Criteria

1. Given a dataset configuration, collation and role, when evaluation arrays are assembled, then they equal the training collation's output for the same sample IDs, in order, with `sample_id` preserved. `dsio.eval` still imports no pipeline layer (import contract unchanged).
2. Given titanic, bike_sharing, store_sales and digit_recognizer, and the supervised/self_supervised fixtures, when they are migrated, then every `_arrays` and `evaluation_arrays` helper is deleted, and evaluation metrics match the golden metrics.

## Tasks / Subtasks

- [x] Task 1: `dsio.experimental.data.arrays:collate_arrays(dataset, store, sample_ids, *, examples=None, collate_fn=None)` (AC 1).
  - `dataset` is a dataset factory or its component config (the same one training records).
  - The samples are built by that factory and collated by `collate_fn` (PyTorch's `default_collate` when `None`, as in `DsioDataModule`). The tensors become NumPy arrays.
  - It checks that ids are non-empty and unique, that every collated value is a tensor (or the `sample_id` strings), and that `sample_id` comes back exactly as requested.
  - `examples` defaults to the store's identity without its attributes, since a dataset factory only reads name and digest. This also covers stores with ragged attributes (Store Sales' `submission_ids`).
  - It lives in `dsio.experimental.data`, not `dsio.eval`, so the eval import contract is unchanged.
- [x] Task 2: `StoredItems` gains a `uint8` dtype (no transforms), so predictor inputs that the predictor scales itself (Digit's raw pixels) come out of the same factory as training items.
- [x] Task 3: Tests in `tests/experimental/data/test_arrays.py`:
  - **Parity:** arrays equal a `DataLoader` over the same factory, in the requested order, with matching dtypes and shapes.
  - **Custom collation** and `uint8` support.
  - **Input checks:** empty, duplicate and unknown ids, non-tensor fields, and reordered collation.
- [x] Task 4: Migrations (AC 2), with goldens unchanged:
  - **titanic, bike_sharing, digit_recognizer:** `_arrays` is deleted. Each declares an `INPUTS` config (predictor inputs), and titanic and digit also declare `EVALUATION` (inputs plus targets). Bike evaluates through its training `DATASET`.
  - **store_sales:** evaluation reads the training `DATASET`, whose `y` is already `log1p` of sales, so the consumer's own `np.log1p` is gone.
  - **supervised fixture:** `evaluation_arrays` is deleted, replaced by `INPUTS` and `EVALUATION`. Export, evaluation and inference use them.
  - **self_supervised fixture:** exports through the supervised `INPUTS`.
  - The hand-written target loops over the store (`int(store.read_sample(...)["attrs"]["target"])`) are gone.
- [x] Task 5: Catalog. `collate_arrays` has full sections, a doctested Example and evidence with six real uses. The "Evaluation array builders" candidate keeps essay, FoG, ROGII and CMI.
- [x] Task 6: Review follow-ups. The review found nothing critical and confirmed dtype and shape parity with every deleted helper, including Store's `log1p` target, Digit's `uint8` inputs and Bike's `[N,1]` targets.
  - [x] 6.1 (important) Collation now runs through the training guard (`collate_items`): mapping, row-count, CPU and ordered-identity checks, with collation failures wrapped as `LoadingError`. Array-valued fields are accepted as in training.
  - [x] 6.2 The docstring says the arrays equal a training batch when given the training config, and that inference or evaluation may declare their own specs over the same factory. Limitations now cover unique ids, targetless inference, attribute-free default examples, memory and view semantics.
  - [x] 6.3 The new mypy errors in the tests are fixed.

## Dev Notes

- **Why one batch.** Evaluation arrays are one collated batch of the requested samples. For fixed-shape consumers this equals the concatenation of training batches. For a padding collation, padding follows the longest requested sample, which is recorded under Limitations and matters when sequence consumers migrate (Epic 8).
- **Parity.** Dtypes and values follow the dataset spec exactly as training does. Digit's inputs stay `uint8` (as its predictor expects), and Store's evaluation target is the same `log1p` float32 the old code computed in NumPy.
- **What stays local.** Essay, FoG, ROGII and CMI array builders, which depend on their dataset and collation stories.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

### File List

- `src/dsio/experimental/data/arrays.py` (new), `src/dsio/experimental/data/__init__.py`, `src/dsio/experimental/data/items.py` (`uint8`)
- `tests/experimental/data/test_arrays.py` (new)
- `reference_projects/kaggle/{titanic,bike_sharing,store_sales,digit_recognizer}/{components.py,tasks/downstream.py}`
- `reference_projects/supervised/{components.py,tasks/downstream.py,tasks/export.py}`, `reference_projects/self_supervised/tasks/export.py`
- `tests/reference_flows/test_{supervised,self_supervised}_flow.py`
- `docs/component-warehouse/{catalog.md,evidence.yaml,candidates.yaml}`, `_bmad-output/implementation-artifacts/sprint-status.yaml`
