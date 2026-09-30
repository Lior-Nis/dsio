---
id: SPEC-component-warehouse
companions:
  - component-cohort.md
  - conventions.md
  - brownfield.md
  - roadmap.md
  - ../../../docs/component-admission.md
  - ../../../docs/superpowers/specs/2026-09-18-generic-experiment-spine.md
  - ../../../CONTEXT.md
sources: []
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability only — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# DSio Component Warehouse v1

## Why

A vision to realize, backed by a measured pain. DSio should work like `timm` or Hugging Face
`transformers`: consumers compose battle-tested lego blocks instead of developing models. The
generic spine is complete. Epics 1–5 are done, and it is proven at scale: the full Parkinson
corpus (4.2B rows scanned), full CMI raw sequences (315M rows), and replayable CUDA training.
The warehouse above the spine is missing. Across the 9 Kaggle reference consumers, local code
re-implements models, objectives, datasets, collators, normalizers and validators. Another
4.4k lines of train/export/evaluate wiring are 60–70% copy-paste. DSio's own backbones,
heads, losses and compositions have **zero Kaggle-consumer users**; only the synthetic
references touch a few. They do not fit real batches. Algua and Pulse are the next
consumers, and each would pay this cost again. Warehouse v1 makes composition the default
path before those integrations begin.

## Capabilities

- id: CAP-1
  intent: A consumer builds models for flat/tabular, token-sequence, signal-sequence (whole and dense per-timestep) and multimodal missing-modality inputs by composing DSio backbones, heads, layout adapters, standardization slots and named compositions, without defining an `nn.Module`.
  success: Every Kaggle consumer whose model pattern is in the v1 cohort expresses its model as DSio component configuration, and its local model class is deleted.

- id: CAP-2
  intent: A consumer selects DSio objectives for binary, multiclass and ordinal classification (class- and sample-weighted), regression including transformed-target space, boolean-masked dense losses and reconstruction. Each objective honors the `(model, batch, stage) -> {loss, ...metrics}` contract with auxiliary metrics.
  success: No Kaggle consumer defines an `Objective`. Metric names logged to MLflow are preserved or renamed with migration notes.

- id: CAP-3
  intent: A consumer maps stored samples and windows into training items, pads variable-length batches, and records every dataset and collator parameter in provenance, using only DSio datasets and collators. Mapping covers field selection, dtype, layout, target transforms and weights.
  success: No Kaggle consumer defines a `Dataset` class or collate function. The store digest check is no longer bypassed. Provenance contains each dataset's and collator's full configuration.

- id: CAP-4
  intent: A consumer fits standardization statistics and class/group weights on training-role samples only. The fitted values are recorded as evidence and injected into models and objectives.
  success: Bike Sharing and both CMI consumers use DSio fitters. A test fails if any non-training sample influences a fitted value. A configured parameter that conflicts with a fitted runtime value raises an error instead of overriding it silently.

- id: CAP-5
  intent: A consumer declares binary, multiclass, ordinal, regression (inverse target transform, non-negative) and dense per-timestep prediction outputs with parameterized validation and MLflow signatures, without writing a normalizer or validator.
  success: The 9 Kaggle consumers' local normalizers and validators are deleted. DSio validators have tests that reject each violation class: shape, finiteness, sign, range, simplex, and threshold/argmax consistency.

- id: CAP-6
  intent: A consumer builds evaluation and inference arrays with the same DSio dataset and collation used in training, and evaluates through `dsio.eval`, including subgroup metrics and modality-ablation deltas.
  success: No Kaggle consumer keeps an `_arrays` or `_targets_and_mask` builder. `child_mind` evaluates through `dsio.eval.evaluate`. `child_mind/sequence`'s bounded-memory streaming evaluation remains a catalogued candidate.

- id: CAP-7
  intent: A consumer's own Prefect task trains or exports by calling a DSio spine function. The task passes a project-constructed `DsioModule`/`DsioDataModule` (or their factories, for calibration) and a `TrainerConfig`. The function calls native `Trainer.fit` once, or builds and logs the Predictor, and adds only DSio invariants: capability evidence, provenance, a digest-verified checkpoint and lineage-checked export. It takes no task, mode or model-family argument, chooses no hyperparameter, and owns no loop, callback policy, flow, task or retry.
  success: Train/export wiring across Kaggle consumers falls by at least 50% from its baseline. Every `@flow` and `@task` stays in consumer code. An ADR amends the spine's "universal trainer" non-goal: a universal trainer is defined as a DSio-owned loop or a task-dispatching entry point, and neither exists.

- id: CAP-8
  intent: A consumer discovers every warehouse component through a catalog generated from docstring sections and a separate evidence register. Each entry gives the import path, maturity, parameters, consumed and produced batch fields, tensor layouts, output fields, supported devices, limitations, a doctested example and its evidence.
  success: CI fails when a public component lacks a required section or evidence entry, when a recorded consumer path or test does not exist, or when the committed catalog differs from the generated one. The catalog lists every admission candidate still living in a consumer.

- id: CAP-9
  intent: Every warehouse component's maturity follows recorded evidence under `docs/component-admission.md`, amended with a pre-1.0 legacy clause, and the static admission audit covers every experimental component. Uses are counted per public callable. Warehouse components are the injectable Components in the catalog. The spine's composition roots, store, closed dispatchers, tracking and the CAP-7 functions are not warehouse components.
  success: The admission doc's legacy clause states that pre-existing components with no real use move to `dsio.experimental`, carry no compatibility promise, and are deleted at 1.0 if still unproven. After v1, every catalogued component in a stable package has two or more unrelated real uses in the evidence register, and every promotion made in v1 records explicit human approval. Tests run `require_admissible_component` over every experimental public component.

- id: CAP-10
  intent: Migrated consumers reproduce their pre-migration results, so the warehouse is proven to preserve behavior rather than merely to run.
  success: Parity compares metrics, not execution identities, because component identities change by design. Contract-tier results match committed golden metrics recorded at the baseline commit. The exception is a migration whose story declares in advance which computation changes (initialization order, random draws, normalization placement, loss reduction) and a tolerance; its deltas are logged in MLflow. Representative-tier parity runs record their hardware. The replay-identity flake is resolved first, either by a root-cause fix with a regression test or by 20 consecutive green replays with provenance-diff capture in place.

- id: CAP-11
  intent: A consumer pins DSio 0.3.0, a release containing the warehouse with a single-sourced version and per-component maturity in its release notes.
  success: `dsio.__version__` equals the distribution metadata version, enforced by a test. Tag `v0.3.0` exists. The distribution and consumer-flow contract CI is green. The release notes list every component with its maturity and evidence.

## Constraints

- **Supersession.** This spec supersedes the 2026-09-25 portfolio guardrails that left model choice, loss and collators with consumers. The collator caution in those guardrails still binds: a shared block's interface must stay materially smaller than the implementations it replaces.
- **Native objects.** Components are, or return, native PyTorch, Lightning, TorchMetrics or MLflow objects. There is no DSio component base class, and no result, evaluation or batch dataclass.
- **No dispatch.** Compositions are separate named factories. There is no task-mode flag, no dispatch on task or project name, and no universal collator or model with mode switches.
- **Selection.** Components are selected as `{"reference": "module:qualname", "parameters": {...}}`. There is no runtime registry, entry point or project-side registration. Factories return module-level, importable classes, never `<locals>` classes, lambdas or closures.
- **Training classes.** `DsioModule` and `DsioDataModule` stay the only training classes and are never subclassed. Projects construct them. CAP-7 functions never define Prefect flows or tasks, and never decide topology, retries or scheduling.
- **Single model input.** The model input stays a single `x`, and the Predictor and export API are not widened. Validity a model needs travels as a declared channel of `x` or a declared padding value. The batch `mask` serves objectives and evaluation.
- **Import contracts hold.** `dsio.eval` stays free of `data`, `model`, `train` and `tracking` imports. Array assembly lives in `dsio.data`.
- **Evidence-based maturity.**
  - Experimental requires one real downstream use.
  - Stable requires an unrelated second use plus human approval.
  - Kaggle reference consumers count as real uses. Synthetic references are fixtures.
  - The two CMI consumers count as one use.
  - Nothing is promoted automatically.
- **Depth test.** A block is admitted only if it removes meaningful implementation from at least one consumer. A thin rename, or a one-line filter, is rejected.
- **Placement.** Code is placed by pipeline responsibility: `data`, `model`, `train`, `eval`, `inference`, `tracking`. There is no `dsio.torch`. A concern starts as a file and becomes a same-named package only when justified, with public imports re-exported.
- **Conventions.** Components follow [conventions.md](conventions.md): batch fields, mask polarity, weighting rule, layouts, docstring sections and the evidence register.
- **Determinism.** Components are deterministic under seed. Stochastic accelerator augmentation runs only in `training_step`. Deterministic preprocessing lives in the Predictor. There is no silent CPU/accelerator fallback.
- **Execution calibration.** When a CAP-7 function enables it, calibration changes only execution knobs. These are micro-batch size, gradient accumulation at a fixed effective batch, workers, pinned memory and prefetch. It never changes architecture, objective, optimizer, learning rate, epochs, augmentation, precision or effective batch. Every candidate and the selection are logged to MLflow.
- **Leakage guards.** Fitted statistics and weights use training-role samples only. The ordered `sample_id` guard and the store digest check stay intact.
- **Consumer ownership.** Consumers keep ingestion, domain semantics, feature/label mapping, flow topology, configuration, submission formatting, promotion and deployment. DSio source never names a consumer or competition.
- **Dependencies.** Components depend only on the admission allowlist: dsio, lightning, mlflow, numpy, prefect, pydantic, sklearn, torch, torchmetrics, yaml. Adding a dependency is a separate decision. DSio package metadata stays accelerator-neutral.

## Non-goals

- Pretrained weights: a weight catalog, weight hub, or verified encoder load-and-freeze handoff (v2, see [roadmap.md](roadmap.md)).
- Runtime discovery APIs (`list_models()`, `describe()`), plugin registries and entry points.
- Admitting architectures on benchmark evidence alone, without a consumer use.
- DSio-owned Prefect flows, tasks, DAG templates, CLI, scheduling or deployment.
- Multi-field model inputs, bounded-memory streaming evaluation, and window-to-entity aggregation (roadmap).
- Forecasting or recommender frameworks, TensorFlow/TFLite, Kaggle notebook packaging.
- The Algua/Pulse integration API. It is the milestone after v1.
- Multi-GPU or multi-node support claims, and new competitions added only for breadth.
- Re-hosting architectures that `timm`, Hugging Face or torchvision already maintain.

## Success signal

- A syntax-tree check over the 9 Kaggle consumers finds no `nn.Module` or `Dataset` subclass, and no locally defined value passed as an objective, collate function, normalizer or validator. Catalogued admission candidates are the only exceptions.
- Non-ingestion consumer lines, measured as recorded at the baseline commit, fall by at least 50%. Migration parity is recorded for every migrated consumer.
- A new reference consumer on a dataset not yet in the portfolio is written outside the repository, given only the installed DSio 0.3.0 wheel, README and catalog. It contains only ingestion, feature/label mapping, configuration, flow topology and submission formatting.

## Assumptions

- The release is 0.3.0. The acceptance consumer uses a public dataset that needs no competition-rule acceptance.

## Open Questions

- Should `calibrate_training_execution` and the telemetry functions (Parkinson FoG and CMI sequence uses) be promoted to stable in v1? Doing so needs human approval under the admission policy, and Story 11.1 decides it.
