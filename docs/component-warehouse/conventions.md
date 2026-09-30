# Warehouse component conventions

These rules let DSio datasets, collators, models, objectives, outputs and evaluation compose
without adapters. They bind every component in the warehouse and every consumer that
feeds one. Normative source: `_bmad-output/specs/spec-component-warehouse/conventions.md`.

Enforcement lands with each warehouse block (Component Warehouse v1, Epics 7–9). The mask and
`hidden` polarity is enforced now. The dataset/collator provenance and config-collision rules
arrive with the first warehouse dataset (Story 7.1). The weighting rule arrives with the
warehouse objectives (Stories 7.4 and 9.3).

## Batch fields

A batch is a flat mapping. These names are reserved:

| Field | Type and shape | Meaning |
|---|---|---|
| `sample_id` | ordered identities | Required. `DsioDataModule` guards the order. |
| `x` | tensor | The **only** model input. `predict_step`, the Predictor and exported models accept only `x`. |
| `y` | tensor | Target. Absent for unlabelled items. |
| `mask` | `bool`, broadcastable to the target's leading axes | **`True` = valid** (observed/scoreable). Read by objectives and by `dsio.eval.evaluate(mask=...)`, never by the model. |
| `sample_weight` | float `[B]` | Per-sample loss weight. See the weighting rule. |
| `group` | identities | Optional group membership, used for group-weight fitting. |

Rules:

- **Never call a True-means-hidden tensor `mask`.** Self-supervised corruption tensors are called `hidden`. The strategies in `dsio.model.masking` return `hidden` tensors, and `MaskedReconstruction` keeps its tensor internal: hidden positions are zeroed in `x`, and visible positions are NaN in `y`.
- **Validity a model needs travels inside `x`.** This covers masked pooling, residual validity and modality presence. Declare it as a channel of `x`, or derive it from a declared padding value.
- **Multimodal inputs are declared slices of `x`**, plus declared presence channels. There is no container class.
- **Datasets and collators fail on a missing declared field.** They never invent one.

## Weighting rule

- **Class weights** keep native loss semantics. Pass them as the native loss's `weight=`.
- **Sample weights** are normalized to mean 1 over the training role, and objectives reduce with `mean(w · loss)`. Normalizing per micro-batch (`sum(w·l) / sum(w)` within a batch) is forbidden, because it makes the loss depend on batch composition.

## Tensor layouts and targets

- **Signal-sequence backbones** consume channel-first `[B, C, T]`. Convert time-major data with a layout adapter, not an ad hoc transpose inside a model.
- **Output shapes.** Dense per-timestep outputs are `[B, T, K]`, and sequence-level outputs are `[B, K]`. Scalar regression is `[B]` or `[B, 1]` as declared, and is never silently squeezed.
- **Class targets.** Multiclass and ordinal targets are zero-based int64 indices. An ordinal label offset is applied only by the output.
- **Transformed regression targets** (log1p, scale) are declared once on the dataset. Their inverse is declared on the output.

## Objectives

An objective has the signature `(model, batch, stage) -> Mapping[str, Tensor]`. `loss` is a
mandatory scalar tensor. Every other entry is a named scalar metric or TorchMetrics value
logged through `LightningModule.log()`.

## Construction

- Every public component is a module-level class or function, selected as `{"reference": "module:qualname", "parameters": {...}}` with canonical-JSON parameters.
- Datasets and collators are supplied as component configurations, and their parameters enter provenance.
- Fitted values, such as statistics and weights, are logged as evidence and passed as runtime arguments. A configured parameter that collides with a runtime argument is an error.
- DSio source never names a consumer, competition or dataset.
