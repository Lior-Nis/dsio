# Story 7.3: Compose MLP models from backbones and heads

Status: review

## Story

As a consumer building a tabular or flat model,
I want to compose a preprocessor, an MLP backbone and a head (including bounded activations) into one model exposing `encode()`,
so that I never define an `nn.Module` for flat inputs (cohort #7, #8a, #8b, #14, #17).

## Acceptance Criteria

1. Given preprocessor, backbone and head component configurations, when the composition factory builds the model, then it returns a native `nn.Module` whose `forward` and `encode` match the configured chain, with shape errors naming the offending stage.
2. Given a non-negative regression target, when a softplus-bounded head is configured, then outputs are non-negative without a consumer activation.
3. Given titanic (linear head), bike_sharing (standardization, linear and bounded head), store_sales (MLP and bounded head), and digit_recognizer (classifier, plus the experimental autoencoder composition #17), when they are migrated, then their local model classes are deleted, and contract tiers match the golden metrics or a pre-declared tolerance where initialization order changes. digit_recognizer keeps only its verified encoder handoff, listed in the candidates register.

## Tasks / Subtasks

- [x] Task 1: `dsio.experimental.model.compositions` (AC 1, 2).
  - **`MLP(input_shape, output, hidden=(), activation="relu", output_activation=None)`** is an `nn.Sequential`.
    - Layers: `Flatten`, then `Linear` + activation per hidden width, then the output `Linear` and an optional output activation.
    - It checks the per-sample shape on every call and casts integer inputs to float32.
    - Activation names are validated at construction, even when no hidden layer uses them.
    - `output_activation="softplus"` bounds a regression output.
  - **`Stages(stages)`** runs configured components in order. Stages may be native modules (`torch.nn:ReLU`, `torch.nn:Unflatten`) or DSio blocks.
  - **`Chain(backbone, head, preprocessor=None, frozen_backbone=False)`** computes `forward = head(encode(x))`; `encode` stops after the backbone. A frozen backbone has `requires_grad` off and runs under `no_grad`.
  - **Error notes.** An error raised inside a `Chain` or `Stages` stage carries a PEP 678 note naming the stage, its class and the input shape. Nested compositions add one note per level. The exception type and message are unchanged.
- [x] Task 2: Tests in `tests/experimental/model/test_compositions.py`:
  - **Seeded bit parity:** compositions match hand-written `Sequential` models for the softplus MLP, the frozen classifier and the autoencoder.
  - **Validation:** shape checks, integer casts and configuration validation.
  - **Freezing:** the frozen backbone takes no gradient.
  - **Composition behavior:** the preprocessor runs first, and stage-naming notes are added.
- [x] Task 3: Migrations (AC 3), with goldens unchanged and no tolerance needed:
  - **titanic:** `PassengerClassifier` → `MLP`, a linear head.
  - **bike_sharing:** `DemandRegressor` → `Stages` of the fitted `Standardize` and a softplus `MLP`.
  - **store_sales:** `ForecastRegressor` → `MLP` with hidden 32 and softplus.
  - **digit_recognizer:**
    - `DigitAutoencoder` → `Chain(MLP encoder, Stages(ReLU, sigmoid MLP, Unflatten))`.
    - `FrozenDigitClassifier` → `Chain(MLP encoder, Linear, frozen_backbone=True)`.
    - Both share one `ENCODER` config.
    - The handoff now saves and loads `model.backbone`, and the consumer's `freeze_encoder()` calls are gone.
    - Provenance records the full autoencoder config instead of a class reference.
  - **Fixtures:** `supervised` (`TinyRegressor`) and `self_supervised` (`TinyEmbedding`) → `MLP`.
- [x] Task 4: Catalog.
  - `MLP`, `Chain` and `Stages` have full sections and doctested Examples, plus evidence with real uses and the owner's standing approval:
    - `MLP`: 4 Kaggle consumers and 2 fixtures;
    - `Chain`: Digit Recognizer;
    - `Stages`: Bike Sharing and Digit Recognizer.
  - Candidates register:
    - "Local nn.Module models" drops the four migrated consumers.
    - A new "Verified encoder handoff" entry records Digit's lineage-checked load, which waits for the v2 pretrained-weights path.
  - Legacy `ComponentChain`, `MLP1d` and `FixedStandardize` say in their catalog summaries which block supersedes them.
- [x] Task 5: Review follow-ups. The adversarial review found nothing critical. It verified bit-identical parameters, outputs and RNG streams for all six migrations. It also verified byte-identical Digit encoder state, three identical Adam steps on the frozen classifier, and predictor export and pyfunc reload for Digit and Bike. Notes appear once per level under `torch.compile` (eager, aot_eager and inductor).
  - [x] 5.1 (important) A frozen backbone now stays in eval mode, set at construction and kept through `train()`. BatchNorm statistics and dropout stay pretrained; a test covers BatchNorm.
  - [x] 5.2 (important) The stage note is best effort. A non-tensor input (e.g. an LSTM's tuple) is described by its type, and a failing `add_note` no longer replaces the real error.
  - [x] 5.3 `MLP` and `Standardize` cast inputs of another dtype (integers, float64) to their parameters' dtype. Before, float64 batches crashed `MLP`, and `Standardize` returned float64. Casting to the module's own dtype keeps `.double()` or bf16-true modules working.
  - [x] 5.4 Slicing an `MLP` or `Stages` returns a plain `nn.Sequential`. Before, it re-entered the config constructor and failed.
  - [x] 5.5 `frozen_backbone` is validated before any stage is built.
  - [x] 5.6 Digit's pretraining provenance records the exported encoder's architecture. The classifier requires it to match its own backbone before loading. A strict `load_state_dict` alone missed activation changes.
  - [x] 5.7 The handoff evidence test now fails each configuration state (foreign, label-tainted, architecture) for its own reason. Before, a wrong identity masked all of them.
  - [x] 5.8 The `Chain` Limitations section states the export path (`backbone.state_dict()`). The legacy `export_encoder` serves only `ComponentChain`.

## Dev Notes

- **Parity reasoning.**
  - An `MLP` is a flat `nn.Sequential` that builds its layers in the hand-written order, so seeded initialization and the encoder's `state_dict` keys (`1.weight`, `3.weight`, ...) are unchanged.
  - `Chain` builds the backbone before the head, which matches each digit model's encoder-then-decoder/classifier order.
  - Every migrated consumer resolves its model at the same point relative to `seed_everything` and the data module. All golden contract metrics are unchanged, so no pre-declared tolerance was needed.
- **Superseded legacy blocks.**
  - `ComponentChain` takes built modules and has a `transform` slot. `export_encoder` depends on it, and the legacy DsioModule tests use it as a fixture.
  - It stays legacy (deleted at 1.0 if unproven) rather than being widened, so the warehouse has one config-driven chain.
  - `MLP1d` (fixed 3-D input, one hidden layer) and `FixedStandardize` (channel-first with eps) are likewise superseded by `MLP` and `Standardize`.
- **What stays local.**
  - Objectives (Story 7.4) and output normalizers (Story 7.5).
  - Digit Recognizer's encoder handoff, which verifies run status, identity, digest and label-free lineage before loading.
  - The supervised fixture's `TimeMajorToChannelFirst`, a fixture-only layout adapter.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

- Gates: ruff, format, mypy (112 files) and import contracts pass. The composition tests (6, plus doctests), the six migrated consumers' contract tests and the catalog check pass.

### File List

- `src/dsio/experimental/model/compositions.py` (new), `src/dsio/experimental/model/__init__.py`
- `src/dsio/experimental/model/{chain.py,components.py}` (superseded notes)
- `tests/experimental/model/test_compositions.py` (new)
- `reference_projects/kaggle/{titanic,bike_sharing,store_sales,digit_recognizer}/{components.py,tasks/training.py}`, `reference_projects/kaggle/digit_recognizer/tasks/downstream.py`
- `reference_projects/{supervised,self_supervised}/{components.py,tasks/training.py}`
- `tests/kaggle_portfolio/test_digit_recognizer.py`, `tests/reference_flows/test_{supervised,self_supervised}_flow.py`
- `docs/component-warehouse/{catalog.md,evidence.yaml,candidates.yaml}`, `_bmad-output/implementation-artifacts/sprint-status.yaml`
