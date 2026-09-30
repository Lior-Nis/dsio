# Warehouse v1 component cohort

The v1 cohort, derived from the 2026-09-30 consumer inventory ([brownfield.md](brownfield.md))
and corrected by adversarial review.

**Counting rules:**

- Uses are counted **per public callable**.
- "Real uses" counts Kaggle reference consumers only. The synthetic `supervised` and `self_supervised` references are fixtures, not uses.
- `child_mind` and `child_mind/sequence` count as **one** unrelated use.

**Targets:**

- **Stable-candidate:** two or more unrelated real uses, so it may be promoted after compatibility tests and human approval.
- **Experimental:** exactly one real use.
- **Legacy experimental:** a pre-existing component with no real use, kept under the admission doc's pre-1.0 legacy clause.
- **Spine:** a CAP-7 function. It is not a Component and follows spine review.

Block numbers are stable IDs. Dropped blocks keep their number. Final public names are
decided in the implementing story, following [conventions.md](conventions.md).

## `dsio.data` — items, collation, fitting, arrays

| # | Block | Replaces (pattern, consumers) | Existing DSio start | Real uses | Target |
|---|---|---|---|---|---|
| 1 | Stored-sample field mapping dataset. Selects columns into `x`/`y`/`mask`/`sample_weight`, with dtype, layout, target transform (offset, log1p, scale), and optional labels. Keeps the digest check and is configured as a recorded ComponentConfig. | D1: titanic, bike, digit, essay, fog, rogii, store, cmi | `StoredSamples` (no target, dtype, layout or transform) | 8 | Stable-candidate |
| 2 | Window dataset with the same mapping, plus per-window `sample_weight` and `group` | cmi-seq `CmiSequenceWindows` | `WindowDataset` | 1 | Experimental |
| 3 | Pad collator. Pads the declared variable-length fields, stacks fixed fields, and optionally emits a validity `mask`. Its interface must stay smaller than the three implementations it replaces. | D4: essay, fog, rogii (only fog emits a mask today) | none | 3 | Stable-candidate |
| 4a | Training-role mean/std fitter, NaN-aware with an observed-only option; fitted values recorded as evidence | D7: bike, cmi (both consumers) | none | 2 | Stable-candidate |
| 4b | Training-role class-frequency weight fitter | D8: cmi, cmi-seq | none | 1 | Experimental |
| 4c | Training-role group-frequency weight fitter (split-wide mean-1 normalization) | D8: cmi-seq | none | 1 | Experimental |
| 5 | Evaluation/inference array assembly from the training dataset and collation (in `dsio.data`, honoring the `dsio.eval` import ban) | D15: `_arrays` in 8 consumers; `_targets_and_mask` in fog, rogii | none | 8 | Stable-candidate |
| 6 | ~~Labelled-example selection by attribute~~ | — | — | — | **Dropped:** a one-line `source == "train"` filter fails the depth test; consumers keep it |

A weighted **sampler** hook in `DsioDataModule` is not in v1. The only proven balancing is
per-sample loss weights (cmi-seq).

## `dsio.model` — backbones, heads, adapters, compositions

| # | Block | Replaces | Existing DSio start | Real uses | Target |
|---|---|---|---|---|---|
| 7 | MLP backbone (flatten → hidden layers) | D12: store, digit | `MLP1d` | 2 | Stable-candidate |
| 8a | Linear head | titanic, bike | `linear_head` | 2 | Stable-candidate |
| 8b | Activation-bounded head (softplus for non-negative targets) | bike, store | none | 2 | Stable-candidate |
| 9a | Fixed standardization/scaling slot, fed by #4a or constants, with an observed-only option | bike, cmi, digit `ScalePixels` (÷255) | `FixedStandardize` | 3 | Stable-candidate |
| 9b | Per-instance standardization, with an observed-only option | fog `normalize_signal`, cmi-seq `_normalize_window` | `InstanceStandardize` | 2 | Stable-candidate |
| 10 | Pooled temporal Conv1d encoder | cmi-seq temporal encoder | `Conv1dEncoder` | 1 | Experimental |
| 11 | Dense per-timestep Conv1d encoder (no pooling) | fog `FogDetector` | none | 1 | Experimental |
| 12 | Token embedding encoder with masked mean pooling. Validity comes from a declared `x` channel. | essay `EssayRegressor` | `EmbeddingEncoder` (unmasked) | 1 | Experimental |
| 13 | Layout adapter (time-major ↔ channel-first) with extent checks | D14: fog, cmi-seq | none | 2 | Stable-candidate |
| 14 | Preprocessor → backbone → head composition, with `encode()` | bike, store, digit | `ComponentChain` | 3 | Stable-candidate |
| 15 | Multimodal fusion composition with modality-presence gating. Modalities are declared slices of `x`, and presence is declared channels. | D10: cmi, cmi-seq | none | 1 | Experimental |
| 16 | Zero-initialized bounded residual around an explicit baseline channel | rogii `TvtRegressor` | none | 1 | Experimental |
| 17 | Autoencoder composition (encoder + decoder, `encode()`) | digit `DigitAutoencoder` | `mae_decoder_head` | 1 | Experimental |

Compositions are separate factories that each return an `nn.Module`. The catalog documents
which objective and output fit each composition. There is no dispatcher.

## `dsio.train` — objectives and fit

| # | Block | Replaces | Existing DSio start | Real uses | Target |
|---|---|---|---|---|---|
| 18 | Supervised objective over any native loss. Covers native loss parameters (class weights keep native semantics), target dtype/shape adaptation (no silent `squeeze`), auxiliary metrics, stage restriction and sample weights (mean-1 rule). | D2: titanic, bike, essay, digit, store, cmi | `LossObjective` | 6 | Stable-candidate |
| 19 | Boolean-masked dense objective over any elementwise loss | D3: fog (BCE), rogii (MSE) | none | 2 | Stable-candidate |
| 20 | Reconstruction objective (target is the input) | digit `ReconstructionObjective` | none | 1 | Experimental |
| 22 | Fit function (CAP-7): native `Trainer.fit` once, MLflow logger, capabilities, optional execution calibration, provenance, digest-verified checkpoint | D16 training bodies in all 11 | `build_trainer`, `save_artifact`, calibration | 8 | Spine |

## `dsio.inference` — outputs, validators, export

| # | Block | Replaces | Existing DSio start | Real uses | Target |
|---|---|---|---|---|---|
| 23 | Export function (CAP-7): checkpoint lineage → Predictor → Logged Model | D16 export bodies in all 11 | `require_checkpoint_lineage`, `build_predictor`, `log_predictor` | 8 | Spine |
| 24 | Binary output + validator: sigmoid, threshold, probability; `[B]` and dense `[B,T,K]` | titanic, fog | none | 2 | Stable-candidate |
| 25 | Multiclass/ordinal output + validator: softmax, argmax, label offset; checks simplex and argmax consistency | essay, cmi, digit | none | 3 | Stable-candidate |
| 26 | Regression output + validator: inverse target transform (scale, expm1), non-negative check; `[B]`/`[B,H]`/`[B,T]` | bike, rogii, store | `TensorOutput`, `validate_tensor_prediction` | 3 | Stable-candidate |

The synthetic self-supervised reference keeps its embedding-norm output and validator as
fixture code, since they have no real use.

## `dsio.eval` — evaluation

| # | Block | Replaces | Existing DSio start | Real uses | Target |
|---|---|---|---|---|---|
| 27 | ~~Evaluate/infer task function~~ | — | — | — | **Dropped:** downstream tasks compose #5 arrays with the existing `evaluate`/`predict` |
| 28 | ~~Window → entity aggregation~~ | — | — | — | **Deferred to roadmap:** its only use is inside cmi-seq's streaming evaluation |
| 29 | Subgroup metrics and modality-ablation deltas over consumer-supplied arrays | D11: cmi | none | 1 | Experimental |

## Existing components with no real use (legacy experimental)

Under the pre-1.0 legacy clause, these move to `dsio.experimental.<domain>`, carry no
compatibility promise, and are deleted at 1.0 if still unproven:

- **Heads:** `mae_decoder_head`, `simclr_projector_head`, `vicreg_projector_head`, `mlp_head`, `identity_head`.
- **Losses:** `CrossEntropy`, `MaskedMSE` (kept alongside `MaskedReconstruction`, which depends on it), `NTXent`, `VICReg`.
- **Augmentation wrappers:** `TwoView`, `MaskedReconstruction`.
- **Masks:** `RandomMask`, `SpanMask`, `PatchMask`, `CausalMask`, `apply_mask`, with the output renamed `hidden` per conventions.
- **Augmentations:** `Jitter`, `RandomScale`, `IdentityAugmentation`, `no_augmentation`.
- **Callbacks:** `OnlineProbe`, `RankMeMonitor`, `rankme`.
- **Other:** `export_encoder`, `entity_attribute_labels`, `effective_sample_size`.

Deleted outright once no consumer or fixture uses them:

- `bce_loss` and `mse_loss`: `<locals>` classes with a forced `squeeze(-1)`, superseded by #18.
- `TensorOutput` and `validate_tensor_prediction`: superseded by #26.

## Remaining consumer-local code after v1

These are admission candidates or domain code, listed in the catalog candidates register:

| Code | Consumer | Reason it stays |
|---|---|---|
| Hash-bucket tokenizer `tokenize` | essay | 1 use, text preprocessing candidate |
| `well_arrays` feature engineering | rogii | Domain |
| Verified encoder handoff (save, verify, load, freeze) | digit | Pretrained path deferred to v2 |
| Streaming evaluation with window → participant aggregation, subgroup and ablation | cmi-seq | Bounded-memory evaluation deferred |
| `labelled_examples` attribute filter | 8 consumers | Fails the depth test; consumer-owned split meaning |
| Kaggle submission formatting and ingestion | all | Consumer-owned |

## Execution calibration and telemetry

`dsio.experimental.training:calibrate_training_execution` and
`dsio.experimental.telemetry:measure_phase`/`log_phase_evidence` have real uses in Parkinson
FoG and CMI sequence (two unrelated uses). These are stable-candidates pending the open
question in SPEC.md, and Story 11.1 decides. Only
`dsio.experimental.execution:calibrate_execution` is admission-audited today.
