# Warehouse component conventions

Every block must compose with every other one without adapters. These conventions are
normative for new and reshaped warehouse components.

## Batch fields

A batch is a flat mapping. The reserved names below codify fields consumers already use.
The table's type and shape column gives the target dtype and tensor shape for each field.

| Field | Type and shape | Meaning |
|---|---|---|
| `sample_id` | ordered identities | Required. Order is guarded by `DsioDataModule`. |
| `x` | tensor | The **only** model input (Predictor, export and `predict_step` accept only `x`). |
| `y` | tensor | Target. Absent for unlabelled items. |
| `mask` | `bool`, broadcastable to the target's leading axes | **True = valid** (observed/scoreable). Consumed by objectives and evaluation, never by the model. Same polarity as `dsio.eval.evaluate(mask=...)`. |
| `sample_weight` | float `[B]` | Per-sample loss weight (see weighting rule). |
| `group` | identities | Optional group membership used by group-weight fitting. |

Rules:

- A tensor where True means *hidden* (self-supervised corruption) is never called `mask`. It uses the field name `hidden`.
- Validity a model needs (masked pooling, residual validity, modality presence) travels inside `x` as a **declared channel**, or is derived from a **declared padding value**. The dataset or collator configuration names that channel. Essay and ROGII already pack validity into `x` this way.
- Multimodal inputs are declared slices of `x`, plus declared presence channels. The fusion composition (#15) takes the slice layout as parameters. There is no container class.
- Collators and datasets fail when a declared field is missing. They never invent it.

## Weighting rule

- **Class weights** keep native loss semantics: they are passed as the native loss's `weight=` parameter. This preserves `child_mind`'s current behavior exactly.
- **Sample weights** are normalized to mean 1 over the training role when fitted (#4c). The objective reduces with `mean(w · loss)`. Normalizing per micro-batch (`sum(w·l)/sum(w)` within a batch) is forbidden, because it makes the loss depend on batch composition. This follows PR #76 and `child_mind/sequence`'s invariance test.

## Tensor layouts

- Signal-sequence backbones consume channel-first `[B, C, T]`.
- Datasets declare their layout. A consumer converts time-major data with the layout adapter (#13), never with an ad hoc transpose inside a model.
- Dense per-timestep outputs are `[B, T, K]`. Sequence-level outputs are `[B, K]`. Scalar regression is `[B]` or `[B, 1]`, as declared, and is never silently squeezed.

## Targets

- Multiclass and ordinal targets are zero-based int64 class indices `[B]`. An ordinal label offset (e.g. scores 1–6) is applied only by the output (#25).
- Binary targets are float in {0, 1} with the output's shape.
- Transformed regression targets (log1p, scale) are declared once on the dataset (#1). The matching inverse is declared on the output (#26). Both are recorded in provenance.

## Objectives

- Signature `(model, batch, stage) -> Mapping[str, Tensor]`. `loss` is a mandatory scalar tensor. Every other entry is a named scalar metric or TorchMetrics value logged by `LightningModule.log()`.
- Auxiliary metrics are configured by name. Objectives do not compute evaluation metrics already owned by `dsio.eval`.

## Construction, configuration, provenance

- Every public component is a module-level class or function, selectable as `{"reference": "module:qualname", "parameters": {...}}`. Parameters are canonical JSON.
- Datasets and collators are supplied as ComponentConfigs, not bare callables. Their full parameters enter provenance.
- Fitted values (#4) are logged as evidence artifacts. Provenance references those artifacts, and they are passed as `resolve_component` runtime arguments. A configured parameter that collides with a runtime argument raises an error. It never overrides silently.
- Public imports are re-exported from the owning package's `__init__.py`. A file that grows is promoted to a same-named package without changing imports.
- Generic names only. No consumer, competition or dataset names appear anywhere in DSio source, including docstrings, which the admission audit enforces.

## Catalog

**Public component.** Every name in `__all__` of these packages:

- `dsio.data.loading`
- `dsio.model`
- `dsio.train`
- `dsio.inference`
- the `dsio.experimental.<domain>` packages

Closed-dispatcher entries (split algorithms, `METRICS`) are catalogued too.

**Docstring sections**, parsed from source:

- **Consumes:** batch fields, dtypes, shapes/layouts.
- **Produces:** fields or tensor shapes returned.
- **Parameters:** each configuration parameter and its default.
- **Devices:** CPU/accelerator placement and dtype support.
- **Limitations:** known unsupported cases.
- **Example:** a minimal, copyable configuration or call, executed as a doctest.

**Evidence register:** `docs/component-warehouse/evidence.yaml`. It sits outside `src/dsio`, because consumer names in source fail the audit. Each entry records, per public callable:

- consumer path and test path (both verified to exist by CI);
- whether the use is real or a fixture;
- the unrelated-use group (the CMI pair shares one);
- MLflow parity run URIs (recorded but not dereferenced in CI);
- for promotions, the human approval reference.

**Candidates register:** `docs/component-warehouse/candidates.yaml`. It records each consumer-local admission candidate: path, reason and use count.

**Generator and output:**

- The generator is repository tooling (under `tools/`), not part of the wheel.
- It writes `docs/component-warehouse/catalog.md`, grouped by pipeline package.
- Maturity is the component's location, which the evidence register must justify.

## Tests beyond the admission checklist

A component that runs inside a Predictor needs an export → MLflow load → `predict`
round-trip test that proves identical outputs. This covers models, deterministic
preprocessing, outputs and validators.
