# Story 7.1: Map stored samples into training items

Status: done

## Story

As a consumer with a staged store,
I want a DSio dataset that maps stored columns into `x`/`y`/`mask`/`sample_weight`, with dtype, layout and target transforms, configured and recorded like any other component,
so that I stop writing a Dataset class per project (cohort #1).

## Acceptance Criteria

1. Given a field-mapping ComponentConfig (source columns per field, dtype, time-major or channel-first layout, optional target transform: offset, log1p or scale; optional labels), when the dataset reads a sample, then it yields exactly the configured fields and verifies that sample's payload digest. A tampered payload fails, and a declared field missing from the store fails with the field named.
2. Given `DsioDataModule` and the dataset's ComponentConfig, when a training attempt records provenance, then the dataset's full configuration is recorded, and a config parameter colliding with a runtime argument raises instead of overriding it.
3. Given titanic, bike_sharing, store_sales and digit_recognizer, plus the supervised and self_supervised fixtures where the mapping fits, when they are migrated, then their local sample Dataset classes and factories are deleted, and contract tiers match the golden metrics.

## Tasks / Subtasks

- [x] Task 1: `dsio.experimental.data.items:StoredItems` (AC 1).
  - It is a dataset factory configured per field (`x` required; `y`, `mask`, `sample_weight` optional).
  - A field reads the stored data array (optional `columns`, `layout`) or one entity attribute. Each field is cast to `dtype`, then `offset`, `log1p` and `divide` apply in that order, then `shape`.
  - The factory verifies the `Examples` describe the store, and every assigned sample resolves. Whole-sample reads verify payload digests.
  - Specs are validated at construction, and errors name the field.
- [x] Task 2: `resolve_component` raises when configured parameters collide with runtime arguments, which previously overrode them silently (AC 2). Consumers record the full `StoredItems` config as `dataset_factory` in provenance, and the supervised flow test pins it.
- [x] Task 3: Migrated, with goldens unchanged (AC 3):
  - titanic, bike_sharing, store_sales and digit_recognizer (two configs, unlabelled and labelled);
  - the supervised and self_supervised fixtures;
  - and, beyond the AC, essay_scoring, rogii and child_mind. Their datasets fit exactly, and no other story covered them.

  Every migrated consumer's local Dataset class and factory is deleted. parkinsons_fog stays local until per-instance standardization lands (Story 8.3), since its dataset z-scores in NumPy. It is recorded in `candidates.yaml`.
- [x] Task 4: Catalog. `StoredItems` has all six sections plus a runnable Example, and its evidence entry lists 7 real Kaggle uses, 2 fixtures and the owner's standing approval. Conventions mark the dataset rules as enforced.

- [x] Task 5: Review follow-ups. The review found nothing critical and verified parity bit-for-bit against the old datasets: all 256 pixel values, large integer targets, every consumer, plus spawn-pickling under `num_workers=2`.
  - [x] 5.1 Transforms can no longer change the declared dtype:
    - `bool` fields take no transform;
    - `int64` fields take only an integer `offset`;
    - float fields take all three transforms;
    - the result is cast back to the declared dtype.
  - [x] 5.2 `optional` is removed. Essay and CMI only feed labelled training rows into the dataset, so the flag only weakened the fail-loud check and risked a mixed batch silently dropping `y` under default collation. A missing target now always fails.
  - [x] 5.3 Errors name the field for:
    - bad or multiple `-1` shapes;
    - unhashable `from`/`dtype`;
    - null attributes;
    - failed reshapes;
    - columns beyond the store's channel count, now checked once in the factory rather than per item.
  - [x] 5.4 The unused `FIELDS` constant is removed, and the `resolve_component` docstring states the collision rule.
  - **Recorded, not changed:** `record_provenance` strips secret and ephemeral key names at any depth, so marking a key such as `shape` ephemeral would record a dataset config different from the one used. No consumer does this; it is a provenance-layer concern beyond this story.

## Dev Notes

- **Bit-for-bit parity.** Transforms run in NumPy on the cast array, exactly as the consumers' datasets did. A test pins that NumPy float32 division matches Digit Recognizer's former `torch` `/ 255.0` bit for bit. All migrated consumers keep their golden contract metrics with no expected-value edits.
- **Bugs caught by the tests:**
  - `np.ascontiguousarray` promotes 0-d targets to shape `(1,)`, so the final conversion uses `np.asarray(order="C")`.
  - Relative store paths resolve against the current directory, so the tests and doctest use temporary directories.
- **No public dataset class.** The built dataset class is private (`_StoredItemsDataset`). Consumers reference only the factory, so it is the catalogued component.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

### File List

- `src/dsio/experimental/data/items.py` (new), `src/dsio/experimental/data/__init__.py`
- `src/dsio/config/components.py`, `tests/config/test_component_resolution.py`
- `tests/experimental/data/test_items.py` (new)
- `reference_projects/kaggle/{titanic,bike_sharing,store_sales,digit_recognizer,essay_scoring,rogii,child_mind}/{components.py,tasks/training.py}`
- `reference_projects/{supervised,self_supervised}/{components.py,tasks/training.py}`
- `tests/reference_flows/test_{supervised,self_supervised}_flow.py`
- `docs/component-warehouse/{catalog.md,evidence.yaml,candidates.yaml,conventions.md}`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
