---
stepsCompleted: [1, 2, 3, 4]
inputDocuments: []
session_topic: 'Placement of training augmentations and package ownership in DSio'
session_goals: 'Choose the GPU, CPU-collate, or hybrid augmentation boundary while preserving determinism and validation isolation; replace the vague dsio.torch namespace with cohesive pipeline-domain packages.'
selected_approach: 'AI-Recommended Techniques'
techniques_used: ['First Principles Thinking', 'Morphological Analysis', 'Failure Analysis']
ideas_generated: []
context_file: ''
technique_execution_complete: true
facilitation_notes: 'The user consistently preferred one explicit owner per responsibility, deterministic replay, and failure over silent fallback.'
session_active: false
workflow_completed: true
---

# Brainstorming Session Results

**Facilitator:** Codex
**Date:** 2026-09-18

## Session Overview

**Topic:** Placement of training augmentations and package ownership in DSio

**Goals:** Choose the GPU, CPU-collate, or hybrid augmentation boundary while preserving determinism and validation isolation; replace the vague `dsio.torch` namespace with cohesive pipeline-domain packages.

### Context Guidance

DSio uses mandatory reusable `DsioModule` and `DsioDataModule` composition roots, Lightning for training, Prefect for DAG orchestration, and MLflow as the evidence system of record. The design must stay minimal and avoid shadow abstractions.

### Session Setup

Explore augmentation placement as an explicit performance and correctness boundary, then assign each reusable primitive to the package that owns its pipeline responsibility.

## Technique Selection

**Approach:** AI-Recommended Techniques

**Analysis Context:** The design must choose an augmentation execution boundary and cohesive package ownership while keeping the mandatory DSio training path minimal.

**Recommended Techniques:**

- **First Principles Thinking:** Derive placement from invariants rather than convention.
- **Morphological Analysis:** Compare viable execution locations across performance and correctness dimensions.
- **Failure Analysis:** Stress-test the preferred boundary against leakage, nondeterminism, and hardware limitations.

**AI Rationale:** This is a concrete technical decision with several valid implementations. A short analytical sequence is more useful than broad unconstrained ideation.

## Technique Execution Results

### First Principles Thinking

**[Pipeline Boundary #1]: Two-Lane Augmentation**

_Concept_: Sample selection, decoding, window extraction, padding, and collation stay on the CPU under `dsio.data`. Batched tensor augmentation runs on the target device from `DsioModule.training_step()` immediately before the injected objective. Validation, test, prediction, and exported inference paths do not invoke it.

_Novelty_: Placement follows the operation's execution requirements—before batching/device transfer versus batched tensor execution afterward—rather than treating every operation called "augmentation" identically.

**Accepted implications:**

- GPU augmentation is structurally train-only because only `training_step()` invokes it.
- Augmentation remains outside `forward()`, preserving clean inference semantics.
- CPU preprocessing can overlap accelerator work and may reduce transfer volume.
- `dsio.torch` is unnecessary; ownership follows pipeline responsibility.

### Morphological Analysis

**[Ownership #2]: Domain Placement, Revised**

_Concept_: PyTorch-dependent primitives live under the pipeline domain that owns their responsibility rather than under a technology-layer `dsio.torch` package. Split generation is part of the data domain because it consumes dataset identity and produces immutable membership views.

_Novelty_: Top-level packages represent cohesive domains, not implementation technologies or concepts promoted merely because they are important.

**Accepted correction:** `dsio.splits` becomes `dsio.data.splits`.

**Packaging rule:** Begin each cohesive concern as a module, such as `data/loading.py`. When distinct responsibilities make the module difficult to navigate or review, promote it to a same-named package such as `data/loading/{samplers,batching,collation}.py`. Re-export the established public names from `loading/__init__.py` so growth does not force consumer import changes. Do not create package scaffolding speculatively.

### Failure Analysis

**[Reproducibility Risk #3]: GPU Randomness After Resume**

_Concept_: Training augmentation must derive randomness from stable execution seed, epoch, sample or batch identities, view identity, and augmentation identity rather than consuming an uncontrolled global RNG stream.

_Novelty_: Retry and checkpoint resume reproduce the same augmented evidence for an unchanged execution identity. Batch size, sampling configuration, and hardware topology belong to execution identity when they affect batch membership.

**Accepted requirement:** Exact augmentation replay is required for an unchanged execution identity; statistical equivalence is insufficient.

**[Execution Risk #4]: Silent Device Fallback**

_Concept_: CPU and accelerator placement are explicit execution inputs. An augmentation unsupported by the declared device or dtype fails during preflight rather than silently moving to another device.

_Novelty_: Placement changes require a new execution identity, making performance and numerical behavior observable rather than an implicit runtime optimization.

**Accepted requirement:** No automatic fallback between CPU and accelerator augmentation paths.

**[Artifact Risk #5]: Training Checkpoint Is Not the Predictor**

_Concept_: A resumable Lightning checkpoint contains the complete training system, while an MLflow Logged Model contains only deterministic preprocessing, the predictor, prediction behavior, signature, and example. Training augmentation and objective logic are excluded from inference export.

_Novelty_: The mandatory universal training class does not become the inference product by accident; native MLflow concepts remain the system of record without a parallel DSio artifact hierarchy.

**Accepted requirement:** Inference export is required only for DAG nodes that declare a predictor output. Evidence-only and representation-analysis nodes may complete without one.

**[Ownership Risk #6]: Two GPU Transformation Owners**

_Concept_: `DsioDataModule` owns CPU decoding, shape preparation, batching, and collation. `DsioModule.training_step()` exclusively owns stochastic train-only accelerator augmentation. The exported predictor owns deterministic preprocessing required during both training and inference. DataModule post-transfer hooks remain uncustomized.

_Novelty_: Placement encodes lifecycle semantics as well as device location, preventing double augmentation and training/inference drift.

**Accepted requirement:** The strict three-lane ownership model is mandatory.

### Creative Facilitation Narrative

The session began with a proposed CPU/GPU choice and converged on an execution-trait boundary. The user identified that technology-layer packaging was unnecessary and corrected split generation into the data domain. Failure analysis then hardened the design around exact replay, explicit placement, distinct training and inference artifacts, and a single GPU augmentation owner.

## Idea Organization and Prioritization

### Data preparation and training execution

- CPU decoding, windowing, sampling, batching, and collation belong to `dsio.data`.
- Stochastic batched accelerator augmentation belongs exclusively to `DsioModule.training_step()`.
- Deterministic preprocessing required at inference belongs to the exported predictor.

### Package cohesion

- Remove the proposed technology-layer `dsio.torch` package.
- Keep split generation under `dsio.data.splits`.
- Start cohesive concerns as modules and promote them to same-named packages only when distinct internal responsibilities emerge.

### Reproducibility and failure semantics

- Augmentation randomness must replay exactly for unchanged execution identity.
- CPU/accelerator placement is explicit and never falls back silently.
- Hardware topology and batching configuration join execution identity when they affect sample composition.

### Artifact boundaries

- Lightning checkpoints preserve the full training system for resume.
- MLflow Logged Models contain inference-only behavior and deterministic preprocessing.
- Predictor export is required only when declared as a DAG-node output.

### Prioritization and action plan

1. Record the three-lane ownership and package-promotion rules in the architecture specification.
2. Define deterministic augmentation keys before implementing accelerator transforms.
3. Keep DataModule post-transfer hooks uncustomized and make `training_step()` the only GPU augmentation entry point.
4. Test exact replay across retry and checkpoint resume.
5. Test that exported predictors exclude stochastic augmentation and training objectives.

## Session Summary and Insights

The accepted design optimizes for one owner per behavior. Performance choices remain benchmark-driven, but correctness boundaries are structural: validation cannot enter the training augmentation path, inference cannot inherit training-only behavior, and unsupported placement fails visibly.
