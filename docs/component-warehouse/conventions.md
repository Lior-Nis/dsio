# Warehouse component conventions

These rules let DSio datasets, collators, models, objectives, outputs and evaluation compose
without adapters. They bind every component in the warehouse and every consumer that
feeds one. Normative source: `_bmad-output/specs/spec-component-warehouse/conventions.md`.
The static form of the batch fields is `dsio.batches`.

Rules marked *(from Story N)* are the target contract. They become enforced as that
Component Warehouse v1 story lands its block. Everything unmarked holds today.

## Batch fields

A batch is a flat mapping. These names are reserved:

| Field | Type and shape | Meaning |
|---|---|---|
| `sample_id` | ordered identities | Required. `DsioDataModule` guards the order. |
| `x` | tensor | The **only** model input. `predict_step`, the Predictor and exported models accept only `x`. |
| `y` | tensor | Target. Absent for unlabelled items. |
| `mask` | `bool` tensor | **`True` = valid** (observed/scoreable), never read by the model. `dsio.eval.evaluate(mask=...)` requires the target's exact shape, or every target axis except a final axis named by `target_names`; implicit broadcasting is rejected. DSio objectives read it *(from Story 8.2)*. |
| `hidden` | `bool` `[B, T]` | **`True` = hidden**: a self-supervised corruption tensor. Only masking strategies produce it. It is never called `mask`. |
| `sample_weight` | float `[B]` | Per-sample loss weight (see the weighting rule). |
| `group` | identities | Optional group membership, used for group-weight fitting. |
| `row` | integer `[B]` | Optional window/entity row index. `WindowDataset` items always carry it, and `DsioModule.predict_step` copies it into predictions when present. |
| `view_id` | identities | Optional view identity, set by two-view augmentation. |

Rules:

- **`hidden` stays out of batches today.** The strategies in `dsio.experimental.model.masking` return `hidden` tensors. `MaskedReconstruction` keeps its tensor internal: hidden positions are zeroed in `x`, and visible positions are NaN in `y`. It emits no batch `mask` and passes a consumer's `mask` through untouched.
- **Validity a model needs travels inside `x`** *(from Stories 8.4, 8.5 and 9.4)*. This covers masked pooling, residual validity and modality presence. Declare it as a channel of `x`, or derive it from a declared padding value.
- **Multimodal inputs are declared slices of `x`** *(from Story 9.4)*, plus declared presence channels. There is no container class.
- **Datasets and collators fail on a missing declared field.** They never invent one. `StoredItems` enforces this for datasets; collators follow *(from Story 8.1)*.

## Weighting rule *(from Stories 7.4 and 9.3)*

- **Class weights** keep native loss semantics. Pass them as the native loss's `weight=`.
- **Sample weights** are normalized to mean 1 over the training role, and objectives reduce with `mean(w · loss)`. Normalizing per micro-batch (`sum(w·l) / sum(w)` within a batch) is forbidden, because it makes the loss depend on batch composition.

## Tensor layouts and targets

- **Signal-sequence backbones** consume channel-first `[B, C, T]`.
- **Datasets declare their layout** (`StoredItems` `layout`). Convert time-major data with the layout adapter *(from Story 8.3)*, not an ad hoc transpose inside a model.
- **Output shapes.** Dense per-timestep outputs are `[B, T, K]`, and sequence-level outputs are `[B, K]`.
- **Scalar regression** is `[B]` or `[B, 1]` as declared, and is never silently squeezed *(from Story 7.4; the legacy `bce_loss`/`mse_loss` factories still squeeze until then)*.
- **Multiclass and ordinal targets** are zero-based int64 indices. An ordinal label offset is applied only by the output.
- **Binary targets** are float in {0, 1}, with the output's shape.
- **Transformed regression targets** (log1p, scale) are declared once on the dataset. The dataset side is recorded in provenance with the dataset configuration; the inverse is declared on the output *(from Story 7.5)*.

## Objectives

An objective has the signature `(model, batch, stage) -> Mapping[str, Tensor]`. `loss` is a
mandatory scalar tensor. Every other entry is a named scalar metric or TorchMetrics value
logged through `LightningModule.log()`. Auxiliary metrics are configured by name, and
objectives do not recompute evaluation metrics that `dsio.eval` owns.

## Construction

- Every public component is a module-level class or function, selected as `{"reference": "module:qualname", "parameters": {...}}` with canonical-JSON parameters.
- Public imports are re-exported from the owning package's `__init__.py`. A file that grows becomes a same-named package without changing imports.
- Datasets (and, from Story 8.1, collators) are supplied as component configurations, and the full configuration enters provenance.
- Fitted values, such as statistics and weights, are logged as evidence and passed as runtime arguments. A configured parameter that collides with a runtime argument is an error (`resolve_component` raises).
- DSio source never names a consumer, competition or dataset.
