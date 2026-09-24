---
stepsCompleted:
  - 1
  - 2
  - 3
  - 4
inputDocuments:
  - docs/superpowers/specs/2026-09-18-generic-experiment-spine.md
  - CONTEXT.md
  - docs/adr/0019-versioned-library-with-project-owned-prefect-flows.md
---

# DSio - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for DSio, decomposing the requirements from the agreed generic experiment-spine architecture into implementable stories.

## Requirements Inventory

### Functional Requirements

FR1: A consumer project can install and pin DSio as a versioned Python dependency without cloning, vendoring, or modifying DSio source.

FR2: A consumer project can define its experiment DAG directly as a Prefect flow using ordinary Prefect tasks and DSio functions.

FR3: A flow can explicitly resolve a native MLflow Experiment without creating a Run; importing DSio must not create runs or other external state.

FR4: Prefect remains authoritative for flow execution identity, topology, and status; DSio does not duplicate them in an empty MLflow Run.

FR5: Each tracked DAG node records its work in a visible top-level Run, with every retry represented as a separate attempt rather than overwriting prior evidence.

FR6: A node can reuse prior work only by resolving and validating immutable references to existing MLflow evidence.

FR7: Each Attempt Run records its actual successful, failed, or cancelled status, while the corresponding flow status remains visible and authoritative in Prefect.

FR8: Pure deterministic tasks can use Prefect caching, while tasks whose result is MLflow evidence resolve and validate that evidence through MLflow rather than a parallel DSio cache.

FR9: A flow can log its normalized project configuration to MLflow while excluding credentials and other declared secrets.

FR10: DSio can derive a deterministic execution identity from the relevant configuration, code, data, split, component, and environment provenance.

FR11: A project can rerun downstream evaluation or inference from referenced immutable model and dataset evidence without retraining.

FR12: All supported training uses the exact reusable `DsioModule` and `DsioDataModule` classes; projects do not subclass or replace them with alternate DSio training paths.

FR13: Projects can vary training behavior by injecting named, importable components with explicit configuration; anonymous lambdas and closures are rejected.

FR14: `DsioModule` accepts a task objective whose stable contract consumes the batch and model output and returns a loss plus named metrics.

FR15: `DsioModule` accepts native PyTorch optimizer and scheduler factories and works with native Lightning callbacks and TorchMetrics objects.

FR16: Training and inference batches carry a stable `sample_id`, and produced predictions preserve the corresponding identity and declared output schema.

FR17: `DsioDataModule` and `dsio.data` own CPU-side decoding, sampling, windowing, batching, and collation.

FR18: Stochastic accelerator-side augmentation runs only inside `DsioModule.training_step()` and is reproducible from stable sample identity plus declared seed material.

FR19: Deterministic preprocessing required to interpret a model is packaged with and executed by the exported predictor.

FR20: DSio preflights required devices, dtypes, and augmentation capabilities and fails before training when the declared execution cannot be honored; it does not silently fall back between CPU and accelerator implementations.

FR21: Datasets are exposed through one stable DSio store interface backed by one benchmark-selected primary storage format.

FR22: The primary storage format is selected through a reproducible benchmark covering representative access patterns, throughput, memory use, build cost, and operational simplicity.

FR23: `dsio.data.splits` exposes a closed dispatcher of DSio-owned split algorithms and does not permit project-side runtime registration.

FR24: A split implementation may delegate to a trusted third-party algorithm while DSio owns parameter validation, provenance capture, output normalization, and invariant checks.

FR25: Every split produces a common manifest containing dataset identity, algorithm and version, parameters, seed, named roles, validations, digest, and the sample assignments needed for replay.

FR26: Split datasets and manifests are logged as native MLflow dataset inputs and artifacts rather than represented by a duplicate DSio experiment-result model.

FR27: Named split roles map explicitly to Lightning phases, and `DsioDataModule` consumes the split manifest plus dataset/component configuration to construct its DataLoaders.

FR28: Reusable samplers, batching policies, collation functions, transforms, and storage adapters live under the relevant `dsio.data` package rather than a generic `dsio.torch` namespace.

FR29: Training metrics are implemented with native TorchMetrics where appropriate and are emitted from `LightningModule.log()` so Lightning remains the source of training metric events.

FR30: Evaluation runs as an ordinary Prefect task, loads immutable MLflow evidence, computes reusable DSio metrics, and logs results back to MLflow without making promotion decisions.

FR31: A logged predictor carries an MLflow Model Signature, and DSio validates semantic output invariants that the signature alone cannot express.

FR32: A consumer can call `dsio.inference.predict(model_uri, inputs)` to load a logged predictor, apply its packaged deterministic preprocessing, and receive validated predictions.

FR33: DSio can export native PyTorch and MLflow PyFunc representations while keeping training checkpoints distinct from the deployable predictor artifact.

FR34: Optional export forms are produced only when explicitly declared by the project and supported by the selected components.

FR35: Reusable models, objectives, transforms, metrics, splitters, and related components enter DSio through a documented experimental-to-stable admission process.

FR36: Experimental components are isolated from the stable namespace until their contracts, tests, provenance behavior, and reuse value pass strict agentic review and human approval.

FR37: A synthetic supervised-learning reference flow demonstrates data preparation, splitting, training, evaluation, inference, MLflow lineage, and selective downstream reruns through the public API.

FR38: A synthetic self-supervised-learning reference flow demonstrates the same spine with an SSL objective and accelerator-side stochastic augmentation through the public API.

### NonFunctional Requirements

NFR1: Given identical declared inputs, seeds, component versions, and supported environment, DSio must reproduce split assignments, augmentation decisions, execution identities, and validated outputs independently of worker scheduling.

NFR2: Experiment evidence must be append-only in practice: retries, failures, cancellations, and reused evidence remain auditable and are never silently rewritten.

NFR3: DSio must fail fast and fail closed when required MLflow evidence, provenance, device capabilities, schema constraints, or output invariants are absent or invalid.

NFR4: DSio must prefer native Prefect, Lightning, PyTorch, TorchMetrics, and MLflow concepts over parallel wrappers, registries, result classes, or configuration models.

NFR5: Public APIs must remain generic and must not branch on consumer-project names, task names, deployment topology, or the current behavior of Pulse, Algua, or any other project.

NFR6: Every reusable result must carry enough code, data, split, component, configuration, seed, and environment provenance to explain and reproduce it.

NFR7: Input and output validation must identify contract violations at the narrowest public boundary and report actionable errors.

NFR8: Stable public APIs follow semantic versioning; experimental APIs may change only through the documented admission and review process.

NFR9: DSio guarantees behavior only on its tested compatibility matrix and must reject unsupported combinations when correctness cannot be assured.

NFR10: Secrets and credentials must never be included in normalized configuration, cache identities, MLflow parameters, tags, or artifacts.

NFR11: Storage and performance claims must be supported by repeatable benchmarks on declared representative workloads, not assumed to generalize to every consumer.

NFR12: Source organization must favor cohesive, deep modules and a small public surface: start with a single module and promote it to a same-named package only when distinct subconcerns make that clearer.

NFR13: Core training, data, evaluation, and inference behavior must remain independent of deployment, serving, remote scheduling, and infrastructure choices.

NFR14: Stable reusable components require focused unit and integration tests, architecture checks, agentic review, and explicit human approval before promotion.

NFR15: DSio must not import consumer-project code; all project variation crosses documented component contracts through importable references and serializable configuration.

### Additional Requirements

- Prefect, PyTorch, Lightning, MLflow, and TorchMetrics are required first-class dependencies of the single DSio distribution; projects pin a compatible DSio version rather than selecting optional backend variants.
- The initial public package is organized by pipeline responsibility: `dsio.data`, `dsio.model`, `dsio.train`, `dsio.metrics`, `dsio.inference`, and `dsio.tracking`.
- Splitting is part of `dsio.data` and lives under `dsio.data.splits`; there is no top-level splits package.
- There is no `dsio.torch` namespace. PyTorch-specific primitives live with the pipeline concern they serve.
- The architecture contains no DSio DAG abstraction, orchestration protocol, runner registry, backend registry, or duplicate orchestration state model. Prefect flows and tasks are the DAG.
- The architecture contains no required DSio CLI. Consumer projects execute their Prefect flows using normal Python and Prefect entry points.
- The architecture contains no duplicate `Experiment`, `EvaluationResult`, `TrainingResult`, or similar persistence model when an MLflow Run, dataset, artifact, logged model, signature, or metric is the canonical record.
- MLflow Experiments group a project or workstream; visible top-level Runs capture node attempts and carry native Prefect flow-run, task-run, dynamic-key, and retry identities.
- Component selection uses explicit import paths or a small closed DSio dispatcher where DSio governance is required; no general runtime plugin registry is introduced.
- Split algorithms are admitted into DSio itself. A project with a novel split strategy contributes it to the experimental namespace with invariants and tests before using it through the supported path.
- The store interface and split manifest are stable boundaries; the selected physical storage implementation and third-party split libraries remain replaceable internals.
- The predictor artifact is the inference boundary. It owns inference-required deterministic preprocessing and output validation but makes no assumptions about batch, streaming, windowed, online, or deployed serving.
- Promotion gates, model approval, deployment, serving, project lifecycle management, remote scheduling, and infrastructure provisioning are explicit non-goals for the current DSio version.
- Existing clone-based templates, DSio-owned fold runners, CLI orchestration, and shadow tracking abstractions are migration targets to supersede or remove while preserving proven correctness checks.
- Both reference flows must run entirely on synthetic data so they are deterministic, fast enough for CI, and independent of consumer repositories or private infrastructure.
- The first implementation should add abstractions only when required by a reference flow or a stated contract; speculative framework layers are out of scope.

### UX Design Requirements

Not applicable. DSio is a Python library with no user-interface deliverable in this scope. Its developer-facing usability requirements are captured by the functional and nonfunctional requirements above.

### FR Coverage Map

FR1: Epic 1 - Install and pin DSio as a versioned dependency.
FR2: Epic 1 - Define experiment DAGs directly as Prefect flows.
FR3: Epic 1 - Resolve explicit native MLflow Experiments without creating Runs.
FR4: Epic 1 - Keep flow identity, topology, and status solely in Prefect.
FR5: Epic 1 - Record node attempts as visible top-level Runs without overwriting evidence.
FR6: Epic 1 - Reuse only validated immutable MLflow references.
FR7: Epic 1 - Reflect actual flow completion, failure, and cancellation status.
FR8: Epic 1 - Use Prefect caching for pure work and MLflow resolution for evidence.
FR9: Epic 1 - Log normalized configuration without secrets.
FR10: Epic 1 - Derive deterministic execution identities from relevant provenance.
FR11: Epic 1 - Rerun downstream work without retraining.
FR12: Epic 3 - Use the exact reusable DSio Lightning classes for all training.
FR13: Epic 3 - Inject named, importable training components.
FR14: Epic 3 - Execute objectives through one stable loss-and-metrics contract.
FR15: Epic 3 - Accept native optimizer, scheduler, callback, and metric components.
FR16: Epic 2 - Preserve stable sample identity through batches and predictions.
FR17: Epic 2 - Own CPU-side data preparation in `DsioDataModule` and `dsio.data`.
FR18: Epic 3 - Run deterministic stochastic accelerator augmentation in `training_step()`.
FR19: Epic 4 - Package inference-required deterministic preprocessing with predictors.
FR20: Epic 3 - Preflight execution capabilities and reject silent fallbacks.
FR21: Epic 2 - Expose one stable store interface backed by one primary format.
FR22: Epic 2 - Select the primary storage format through a reproducible benchmark.
FR23: Epic 2 - Provide a closed dispatcher of DSio-owned split algorithms.
FR24: Epic 2 - Wrap trusted split algorithms with DSio validation and provenance.
FR25: Epic 2 - Produce replayable common split manifests.
FR26: Epic 2 - Log splits through native MLflow datasets and artifacts.
FR27: Epic 2 - Map split roles to Lightning phases and construct DataLoaders.
FR28: Epic 2 - Organize reusable data primitives under `dsio.data`.
FR29: Epic 3 - Emit training metrics through TorchMetrics and `LightningModule.log()`.
FR30: Epic 4 - Evaluate immutable evidence in ordinary Prefect tasks.
FR31: Epic 4 - Combine MLflow Model Signatures with semantic output validation.
FR32: Epic 4 - Predict through `dsio.inference.predict(model_uri, inputs)`.
FR33: Epic 4 - Export distinct native PyTorch and MLflow PyFunc predictor artifacts.
FR34: Epic 4 - Produce optional export forms only when explicitly declared.
FR35: Epic 5 - Admit reusable components through a documented governance process.
FR36: Epic 5 - Isolate experimental components until review and promotion.
FR37: Epic 5 - Demonstrate the public API with a synthetic supervised flow.
FR38: Epic 5 - Demonstrate the public API with a synthetic SSL flow.

## Epic List

### Epic 1: Run and Reuse Auditable Experiments

A project can install DSio, define its DAG directly in Prefect, execute it with explicit MLflow tracking, inspect every attempt, and safely reuse immutable prior evidence.

**FRs covered:** FR1, FR2, FR3, FR4, FR5, FR6, FR7, FR8, FR9, FR10, FR11

### Epic 2: Build Replayable Training Data

A project can store data, generate governed splits, reproduce sample assignments, and construct identity-preserving Lightning DataLoaders through the standard DSio data path.

**FRs covered:** FR16, FR17, FR21, FR22, FR23, FR24, FR25, FR26, FR27, FR28

### Epic 3: Train Through One Reusable Lightning Spine

A project can train different tasks through the exact `DsioModule` and `DsioDataModule`, injecting named components while retaining deterministic augmentation, native metrics, and fail-fast capability checks.

**FRs covered:** FR12, FR13, FR14, FR15, FR18, FR20, FR29

### Epic 4: Evaluate and Run Portable Inference

A project can evaluate immutable evidence, export a self-contained predictor, and obtain schema-validated predictions without DSio imposing a deployment or serving model.

**FRs covered:** FR19, FR30, FR31, FR32, FR33, FR34

### Epic 5: Adopt Battle-Tested Components and Reference Workflows

Projects can reuse reviewed DSio components and learn the complete supported workflow from executable supervised and self-supervised examples.

**FRs covered:** FR35, FR36, FR37, FR38

## Epic 1: Run and Reuse Auditable Experiments

A project can install DSio, define its DAG directly in Prefect, execute it with explicit MLflow tracking, inspect every attempt, and safely reuse immutable prior evidence.

### Story 1.1: Install DSio and Run a Project-Owned Flow

As a consumer-project developer,
I want to install DSio and use its functions from my own Prefect flow,
So that my project owns its workflow without cloning DSio or adopting a parallel orchestration layer.

**Requirements:** FR1, FR2

**Acceptance Criteria:**

**Given** a clean supported Python environment
**When** the project installs the built DSio distribution
**Then** `import dsio` succeeds with the documented required dependencies
**And** importing the package creates no MLflow Run, Prefect flow, files, or other external state.

**Given** a consumer project with an ordinary `@flow` and `@task`
**When** those functions call a minimal public DSio function
**Then** the flow executes through normal Python and Prefect entry points
**And** no DSio DAG class, runner registry, backend registry, or required CLI is involved.

**Given** a built wheel or source distribution
**When** its contents are inspected and installed outside the repository
**Then** the public package and required metadata are present
**And** consumer-project source, repository-relative imports, and clone-based templates are not required.

**Given** the supported public package layout
**When** a consumer imports DSio functionality
**Then** functionality is organized by pipeline responsibility
**And** no `dsio.torch` namespace or project-specific branch is introduced.

### Story 1.2: Resolve the Native MLflow Experiment

As an experiment author,
I want to resolve the native MLflow Experiment for my workstream without creating a Run,
So that evidence is grouped without duplicating Prefect's flow lifecycle.

**Requirements:** FR3, FR4, FR7

**Acceptance Criteria:**

**Given** valid MLflow tracking and experiment configuration
**When** the flow calls `dsio.tracking.resolve_experiment(...)`
**Then** the native MLflow Experiment is resolved or created
**And** its Experiment ID is available to tasks without creating or activating a Run.

**Given** the same workstream executes more than once
**When** each flow resolves its MLflow Experiment
**Then** each receives the same native Experiment identity
**And** no empty flow-level Run is created.

**Given** another MLflow Run is active in the process
**When** a flow resolves its MLflow Experiment
**Then** that Run remains active and unchanged.

**Given** a flow succeeds, fails, or is cancelled
**When** its lifecycle changes
**Then** Prefect remains the sole source of flow status
**And** DSio does not project or duplicate that state into MLflow.

**Given** MLflow cannot resolve or create the required Experiment
**When** the flow requests it
**Then** DSio fails closed with an actionable error.

### Story 1.3: Track Task Attempts as Visible Immutable Runs

As an experiment author,
I want every tracked Prefect task attempt recorded as a directly visible MLflow Run,
So that retries and failures remain auditable without overwriting earlier evidence.

**Requirements:** FR5

**Acceptance Criteria:**

**Given** a task receives a valid MLflow Experiment ID
**When** a tracked task attempt begins
**Then** DSio creates one new top-level native MLflow Run in that Experiment
**And** records its Prefect flow-run, task-run, dynamic-key, and attempt identities using native MLflow metadata.

**Given** Prefect retries a failed task
**When** the next attempt begins
**Then** a new Run is created for that attempt
**And** the failed prior Run and all of its evidence remain unchanged.

**Given** a task attempt succeeds, fails, or is cancelled
**When** the attempt ends
**Then** its Run records the corresponding terminal status
**And** failures and cancellations continue to propagate through Prefect.

**Given** multiple tasks execute concurrently
**When** they create and update Runs
**Then** every Run remains linked to the correct Prefect flow and logical node
**And** no task relies on another task's process-global active Run.

**Given** a tracked task has no valid Experiment reference or MLflow cannot persist its required evidence
**When** it attempts to start or finish
**Then** it fails with an actionable tracking error
**And** it does not create a falsely successful record.

**Given** task code logs parameters, metrics, artifacts, or models
**When** the task completes
**Then** those values remain native MLflow evidence
**And** DSio does not create a parallel task-result persistence model.

### Story 1.4: Record Deterministic Identity and Safe Configuration

As an experiment author,
I want each execution identified from its meaningful inputs and accompanied by a safe normalized configuration,
So that I can compare, reproduce, and reuse work without exposing credentials.

**Requirements:** FR9, FR10

**Acceptance Criteria:**

**Given** semantically equivalent identity inputs with different mapping order or supported equivalent representations
**When** DSio normalizes and hashes them
**Then** it produces the same execution identity
**And** repeated calculation is independent of process and worker scheduling.

**Given** a relevant configuration, code, data, split, component, seed, or supported environment input changes
**When** the identity is recalculated
**Then** the execution identity changes.

**Given** ephemeral runtime metadata or a declared secret changes
**When** the identity is recalculated
**Then** the execution identity does not change
**And** the secret value is not present in the normalized representation.

**Given** an execution starts successfully
**When** provenance is recorded
**Then** the identity and normalized configuration are stored as native MLflow metadata or artifacts
**And** the recorded provenance identifies the DSio version and relevant component references.

**Given** configuration contains credentials or fields declared secret
**When** DSio normalizes, hashes, logs, or reports that configuration
**Then** secret values are excluded rather than merely masked after logging
**And** tests confirm they do not appear in MLflow parameters, tags, artifacts, or error messages.

**Given** a required identity input cannot be normalized deterministically
**When** execution identity is requested
**Then** DSio rejects it with an error identifying the unsupported input
**And** it does not substitute an unstable string or memory-address representation.

### Story 1.5: Resolve and Reuse Verified MLflow Evidence

As an experiment author,
I want a task to reuse previously produced evidence only when it is complete and provenance-compatible,
So that selective reruns remain fast without silently accepting stale or ambiguous results.

**Requirements:** FR6, FR8

**Acceptance Criteria:**

**Given** a task requests prior evidence by deterministic execution identity and expected contract
**When** DSio resolves a matching MLflow Run
**Then** it verifies the Run succeeded, required artifacts exist, and recorded provenance satisfies the requested contract
**And** it returns native immutable MLflow identifiers or URIs rather than a duplicate DSio result object.

**Given** matching evidence is missing, incomplete, failed, incompatible, or no longer accessible
**When** reuse is evaluated
**Then** DSio treats it as unusable and reports the failed validation
**And** the Prefect flow may execute a fresh task attempt according to project-owned flow logic.

**Given** a reference uses a mutable alias, stage, or other location that can later resolve to different evidence
**When** immutable reuse is required
**Then** DSio rejects the reference
**And** identifies the immutable Run, artifact, dataset, or model reference form that is required.

**Given** evidence from an earlier execution is reused in a new flow execution
**When** the consuming task records its provenance
**Then** the new Attempt Run records the source Run ID and immutable evidence URI
**And** the source evidence is neither copied unnecessarily nor modified.

**Given** a deterministic task produces only a pure serializable value
**When** caching is enabled by the project
**Then** the task can use Prefect's native cache mechanism with an identity-derived key.

**Given** a task's meaningful output is MLflow evidence
**When** reuse is enabled
**Then** availability and validity are resolved through MLflow
**And** DSio does not add a parallel cache store or serialize the evidence into Prefect's cache.

**Given** MLflow is unavailable or evidence validity cannot be established
**When** reuse is requested
**Then** DSio fails closed with the unavailable or unverifiable evidence identified
**And** it neither claims a cache hit nor consumes the unverified evidence.

### Story 1.6: Rerun Downstream Work Without Recomputing Upstream Evidence

As an experiment author,
I want a new flow execution to consume immutable evidence from an earlier execution,
So that I can change and rerun downstream analysis without repeating valid upstream work.

**Requirements:** FR11

**Acceptance Criteria:**

**Given** immutable model and dataset evidence from a successful earlier Run
**When** a project-owned Prefect flow supplies those references to a downstream task
**Then** DSio validates the references before the task executes
**And** no upstream training or data-production task is invoked by DSio.

**Given** the downstream-only flow executes
**When** its downstream task starts
**Then** it creates a new visible Attempt Run
**And** records lineage to every consumed source Run and immutable artifact.

**Given** downstream configuration changes while source evidence remains unchanged
**When** execution identity is calculated
**Then** the downstream execution receives a new identity
**And** the source artifacts remain unmodified.

**Given** a project wants to choose which downstream tasks run
**When** it defines its Prefect flow
**Then** selection and ordering remain ordinary project-owned Python and Prefect logic
**And** DSio does not introduce a rerun planner, DAG model, or hidden scheduler.

**Given** a source reference is mutable, missing, failed, or incompatible with the downstream contract
**When** the flow validates its inputs
**Then** execution fails before downstream computation
**And** reports which evidence contract was not satisfied.

**Given** the downstream task succeeds
**When** its outputs and required validations are complete
**Then** the new evidence is logged to its visible Attempt Run
**And** the original source execution remains unchanged and auditable.

## Epic 2: Build Replayable Training Data

A project can store data, generate governed splits, reproduce sample assignments, and construct identity-preserving Lightning DataLoaders through the standard DSio data path.

### Story 2.1: Select the Primary Store Through a Reproducible Benchmark

As a DSio data practitioner,
I want the primary storage format selected from measured representative workloads,
So that the library makes one evidence-based storage choice instead of exposing speculative backend complexity.

**Requirements:** FR22

**Acceptance Criteria:**

**Given** the focused set of candidate formats and representative synthetic datasets
**When** the benchmark runs through equivalent read and write operations
**Then** it measures declared sequential, random, and multi-worker access patterns
**And** records throughput, memory use, build cost, and operational complexity.

**Given** a benchmark execution
**When** its results are saved
**Then** the inputs, seed, environment, candidate versions, and raw measurements are recorded
**And** another supported environment can rerun the benchmark from one documented command.

**Given** the benchmark results
**When** the primary format is selected
**Then** the decision and its workload-specific tradeoffs are documented
**And** DSio exposes one primary implementation rather than a public storage-backend registry.

**Given** benchmark results from one workload or machine
**When** the decision is communicated
**Then** claims are limited to the measured conditions
**And** unsupported universal performance claims are not made.

### Story 2.2: Read and Write Samples Through One Store Interface

As a dataset author,
I want to build and reopen a DSio store through one stable interface,
So that training code is independent of the selected physical format.

**Requirements:** FR21

**Acceptance Criteria:**

**Given** supported sample fields and stable sample identifiers
**When** a store is built and reopened
**Then** every sample can be retrieved by its stable identity or deterministic position
**And** field shape, dtype, and schema metadata are preserved.

**Given** the same completed store is opened by multiple DataLoader workers
**When** they read samples concurrently
**Then** reads are deterministic and do not require loading the entire dataset into each worker's memory.

**Given** a truncated, corrupt, or schema-incompatible store
**When** DSio opens or reads it
**Then** validation fails with the affected store and invariant identified
**And** partial data is not silently returned.

**Given** a consumer uses the public store API
**When** the physical implementation changes compatibly in a later DSio version
**Then** consumer training code does not need a backend-specific branch
**And** no runtime backend registration mechanism is required.

### Story 2.3: Generate a Governed Split Manifest

As a dataset author,
I want to generate train, validation, test, or task-specific roles through a DSio-owned splitter,
So that sample assignments are reproducible, validated, and portable across projects.

**Requirements:** FR23, FR24, FR25

**Acceptance Criteria:**

**Given** a dataset identity, known splitter name, validated parameters, and seed
**When** the closed split dispatcher executes
**Then** it returns a common manifest containing dataset identity, algorithm and version, parameters, seed, named roles, validations, assignments, and digest.

**Given** identical inputs and supported environment
**When** split generation is repeated
**Then** the manifest digest and sample assignments are identical
**And** worker ordering does not influence the result.

**Given** a splitter delegates to a trusted third-party library
**When** the split is generated
**Then** DSio validates its parameters, normalizes its output, records dependency provenance, and checks declared invariants.

**Given** generated role assignments
**When** manifest validation runs
**Then** unknown sample identifiers, forbidden overlap, missing required roles, and violated coverage constraints are rejected with actionable errors.

**Given** an unknown or project-runtime-registered splitter name
**When** the dispatcher resolves it
**Then** resolution fails
**And** the error directs novel algorithms to DSio's governed experimental admission path.

### Story 2.4: Record and Reload Split Evidence in MLflow

As an experiment author,
I want split data and its manifest recorded with native MLflow lineage,
So that later training and evaluation Runs can reuse the exact dataset evidence.

**Requirements:** FR26

**Acceptance Criteria:**

**Given** a validated store and split manifest inside a tracked task
**When** split evidence is logged
**Then** the dataset is recorded as a native MLflow dataset input
**And** the full manifest is stored as an artifact with its digest and source dataset identity.

**Given** an immutable Run and manifest artifact reference
**When** a later task reloads the split evidence
**Then** DSio verifies the Run status, artifact digest, dataset identity, and manifest invariants before returning it.

**Given** missing, mutable, corrupt, or mismatched split evidence
**When** reuse is requested
**Then** DSio rejects it before constructing DataLoaders
**And** identifies the failed evidence check.

**Given** valid split evidence
**When** it is consumed by another Run
**Then** the consuming Run records native MLflow lineage to the source dataset and artifact
**And** no duplicate DSio dataset-result entity is persisted.

### Story 2.5: Construct Identity-Preserving DataLoaders

As a training-task author,
I want `DsioDataModule` to construct DataLoaders from a store and split manifest,
So that every training paradigm shares one replayable CPU data path.

**Requirements:** FR16, FR17, FR27, FR28

**Acceptance Criteria:**

**Given** a validated store, split manifest, and explicit mapping from roles to Lightning phases
**When** `DsioDataModule` is set up
**Then** it constructs the requested train, validation, test, and predict DataLoaders
**And** each loader contains exactly the sample assignments for its mapped role.

**Given** configured decoding, sampling, windowing, batching, and collation components
**When** a batch is produced
**Then** those CPU-side operations execute through reusable primitives under `dsio.data`
**And** accelerator-side stochastic augmentation is not performed in the DataLoader or collate function.

**Given** a produced batch
**When** it enters the training or prediction pipeline
**Then** every example retains its stable `sample_id`
**And** identity remains aligned with its data and target fields after batching and collation.

**Given** the same store, manifest, component configuration, and seed
**When** loaders run with supported worker counts
**Then** sample membership and declared deterministic transformations are reproducible.

**Given** a missing role, invalid component reference, incompatible batch, or worker-unsafe data primitive
**When** setup or iteration occurs
**Then** DSio fails at the narrowest boundary with an actionable error
**And** does not silently substitute a default data path.

## Epic 3: Train Through One Reusable Lightning Spine

A project can train different tasks through the exact `DsioModule` and `DsioDataModule`, injecting named components while retaining deterministic augmentation, native metrics, and fail-fast capability checks.

### Story 3.1: Train a Task Through the Exact DSio Lightning Classes

As a training-task author,
I want to train through the concrete `DsioModule` and `DsioDataModule` classes,
So that every project exercises one battle-tested training path.

**Requirements:** FR12, FR14

**Acceptance Criteria:**

**Given** a PyTorch model, a valid objective, and a configured `DsioDataModule`
**When** a native Lightning `Trainer` runs `fit`
**Then** training and validation execute through exact instances of `DsioModule` and `DsioDataModule`
**And** no DSio runner, alternate training framework, or project-specific subclass is required.

**Given** a task objective receives the batch and model output
**When** it executes
**Then** it returns a scalar loss plus a mapping of named metric values through one documented contract
**And** malformed loss or metric output is rejected with an actionable error.

**Given** a project attempts to subclass either DSio training class
**When** the class is defined or validated for execution
**Then** DSio rejects the unsupported extension path
**And** directs variation through injected components.

**Given** a successful training run
**When** Lightning checkpoints its state
**Then** the checkpoint can resume the same training configuration
**And** remains a training artifact rather than being treated as the deployable predictor.

### Story 3.2: Configure Training with Native Importable Components

As a training-task author,
I want to inject named PyTorch and Lightning components into the reusable module,
So that tasks vary without forks, subclasses, or a custom plugin framework.

**Requirements:** FR13, FR15, FR29

**Acceptance Criteria:**

**Given** an import path and serializable configuration for a model, objective, optimizer, scheduler, transform, or metric
**When** DSio resolves the component
**Then** it imports the named object, validates the expected contract, and includes the reference and configuration in execution provenance.

**Given** an anonymous lambda, closure, local-only object, or unresolved import path
**When** it is supplied as a reusable component
**Then** DSio rejects it before training
**And** explains the named importable form that is required.

**Given** native PyTorch optimizer and scheduler factories
**When** `DsioModule.configure_optimizers()` executes
**Then** it returns Lightning-supported native optimizer and scheduler structures
**And** DSio does not wrap them in a parallel optimizer abstraction.

**Given** native Lightning callbacks and TorchMetrics objects
**When** training runs
**Then** callbacks remain owned by the Lightning `Trainer`
**And** training metrics are emitted from `LightningModule.log()` rather than directly through a DSio-to-MLflow metrics bridge.

**Given** two configurations with different component references or meaningful component parameters
**When** execution identity is calculated
**Then** their identities differ
**And** their MLflow provenance records the exact selected components.

### Story 3.3: Apply Reproducible Stochastic Augmentation on the Accelerator

As a training-task author,
I want stochastic training augmentation applied after device transfer,
So that augmentation is fast while remaining exactly replayable.

**Requirements:** FR18

**Acceptance Criteria:**

**Given** a configured accelerator-side training augmentation
**When** `DsioModule.training_step()` receives a transferred batch
**Then** the augmentation executes inside `training_step()` before the model forward pass
**And** it is not duplicated in DataLoader workers or collation.

**Given** stable sample identity, declared seed material, epoch or step identity, and view identity
**When** augmentation parameters are generated
**Then** the same inputs reproduce the same augmented view independently of worker scheduling
**And** a declared seed or view change produces a distinct deterministic result.

**Given** validation, test, prediction, or deterministic preprocessing
**When** those paths execute
**Then** stochastic training augmentation is not applied
**And** inference-required deterministic transforms remain outside the training augmentation lane.

**Given** multi-view training such as contrastive learning
**When** the objective requests declared view identities
**Then** each view is generated reproducibly from the same source sample
**And** each view remains associated with the source `sample_id`.

### Story 3.4: Reject Unsupported Training Configurations Before Execution

As a training-task author,
I want execution capabilities checked before expensive work begins,
So that an unsupported device, dtype, or operation cannot silently alter experiment semantics.

**Requirements:** FR20

**Acceptance Criteria:**

**Given** a declared device, dtype, model, objective, and accelerator augmentation configuration
**When** training preflight runs
**Then** DSio verifies required devices and operations are available and compatible
**And** reports every detectable blocking incompatibility before `Trainer.fit()` starts.

**Given** a required accelerator operation is unavailable
**When** preflight evaluates the configuration
**Then** execution fails with the unsupported component and capability identified
**And** DSio does not move that operation to CPU, change dtype, or choose another implementation silently.

**Given** a supported configuration from the tested compatibility matrix
**When** preflight and training run
**Then** the declared execution path is used
**And** the relevant environment and capability provenance is logged to MLflow.

**Given** a configuration outside the tested compatibility matrix whose correctness cannot be assured
**When** it is validated
**Then** DSio rejects it before training
**And** directs the component to the explicit experimental admission path rather than presenting it as stable behavior.

## Epic 4: Evaluate and Run Portable Inference

A project can evaluate immutable evidence, export a self-contained predictor, and obtain schema-validated predictions without DSio imposing a deployment or serving model.

### Story 4.1: Build a Self-Contained Predictor from Training Evidence

As a model author,
I want to turn successful training evidence into a predictor with its required preprocessing,
So that inference does not depend on reconstructing project training code by convention.

**Requirements:** FR16, FR19, FR33

**Acceptance Criteria:**

**Given** an immutable successful checkpoint and importable model components
**When** a predictor is built
**Then** it packages the model state, deterministic inference preprocessing, output normalization, and semantic output validator
**And** excludes stochastic training-only augmentation.

**Given** an input sample with a stable `sample_id`
**When** the predictor executes locally
**Then** it applies deterministic preprocessing, model forward, normalization, and semantic validation in order
**And** the validated prediction retains the source identity and declared output fields.

**Given** preprocessing or validation behavior cannot be imported or serialized reproducibly
**When** predictor construction is attempted
**Then** construction fails before logging
**And** identifies the unsupported component.

**Given** a Lightning training checkpoint and a completed predictor
**When** both are recorded
**Then** the checkpoint remains the resumption artifact and the predictor remains the inference artifact
**And** neither is represented by a duplicate DSio model-registry entity.

### Story 4.2: Log a Signed MLflow Model with Explicit Export Forms

As a model author,
I want the predictor logged with MLflow's native model contract,
So that its accepted inputs, produced outputs, and supported representations are portable and inspectable.

**Requirements:** FR31, FR33, FR34

**Acceptance Criteria:**

**Given** a valid predictor and representative input example
**When** it is logged to MLflow
**Then** the logged model includes an MLflow Model Signature and input example
**And** its immutable Run and artifact references are recorded for reuse.

**Given** semantic output constraints that an MLflow Signature cannot express
**When** the model is packaged
**Then** the DSio semantic validator is included with the predictor
**And** invalid outputs fail inference rather than being logged as successful predictions.

**Given** native PyTorch and MLflow PyFunc export forms are declared
**When** export completes
**Then** each supported form is logged through native MLflow model facilities
**And** loading either form yields contract-equivalent predictions for the validation fixture.

**Given** an optional export form is not declared or is unsupported by a selected component
**When** export runs
**Then** DSio does not produce that form
**And** an explicitly requested unsupported form fails with its incompatibility identified.

### Story 4.3: Predict from an Immutable Model URI

As an inference consumer,
I want to call `dsio.inference.predict(model_uri, inputs)`,
So that I can obtain validated predictions without knowing the model's training implementation.

**Requirements:** FR32

**Acceptance Criteria:**

**Given** an immutable URI for a valid logged predictor and signature-compatible inputs
**When** `dsio.inference.predict(model_uri, inputs)` is called
**Then** DSio loads the predictor, applies packaged deterministic preprocessing, runs inference, and validates outputs
**And** returns native prediction data rather than a DSio result wrapper.

**Given** batched inputs with stable sample identifiers
**When** prediction succeeds
**Then** output ordering and identity align with the corresponding inputs
**And** declared prediction fields satisfy both the MLflow Signature and semantic validator.

**Given** an incompatible input, mutable or missing model reference, unsupported device request, or invalid model output
**When** prediction is attempted
**Then** DSio fails at the applicable boundary with an actionable error
**And** does not silently coerce schema, device, or dtype semantics.

**Given** a caller uses the prediction function
**When** choosing batch, streaming, windowed, online, or deployed execution around it
**Then** that lifecycle remains caller-owned
**And** the prediction API introduces no scheduler, server, promotion gate, or deployment assumption.

### Story 4.4: Evaluate Immutable Evidence in a Prefect Task

As an experiment author,
I want evaluation to consume immutable model and dataset evidence in an ordinary Prefect task,
So that I can recompute metrics independently of training and deployment decisions.

**Requirements:** FR30

**Acceptance Criteria:**

**Given** immutable compatible model and dataset references plus configured metrics
**When** a project-owned Prefect evaluation task executes
**Then** it validates and loads the evidence, runs the predictor, and computes reusable native metric implementations
**And** does not invoke training.

**Given** evaluation executes within a tracked flow
**When** metrics and supporting artifacts are complete
**Then** they are logged to the visible evaluation Attempt Run using native MLflow metrics, dataset inputs, and artifacts
**And** lineage to the source model and dataset Runs is recorded.

**Given** only metric configuration or evaluation data changes
**When** the evaluation flow reruns
**Then** it creates a new visible evaluation Attempt Run
**And** reuses the unchanged validated predictor without retraining.

**Given** incompatible prediction schema, target schema, metric inputs, or invalid source evidence
**When** evaluation validates its contract
**Then** it fails before logging successful metrics
**And** identifies the incompatible boundary.

**Given** evaluation results are available
**When** the task completes
**Then** DSio makes no promotion, approval, registry-alias, or deployment decision
**And** those policies remain outside DSio.

## Epic 5: Adopt Battle-Tested Components and Reference Workflows

Projects can reuse reviewed DSio components and learn the complete supported workflow from executable supervised and self-supervised examples.

### Story 5.1: Govern Components from Experimental to Stable

As a DSio maintainer,
I want reusable components admitted through one strict experimental-to-stable path,
So that projects can share battle-tested code without turning DSio into an unrestricted plugin system.

**Requirements:** FR35, FR36

**Acceptance Criteria:**

**Given** a proposed reusable model, objective, transform, metric, splitter, sampler, collator, or related component
**When** it enters DSio
**Then** it first lives in the documented experimental namespace
**And** is selected by a named import path or an explicitly governed closed dispatcher.

**Given** a component is proposed for stable promotion
**When** its admission evidence is reviewed
**Then** focused unit tests, integration tests, deterministic and provenance behavior, generic contracts, reuse evidence, agentic review, and explicit human approval are required
**And** failed criteria keep the component experimental.

**Given** a component branches on a consumer-project name, relies on private project code, or bypasses a stable DSio contract
**When** admission checks run
**Then** the component is rejected
**And** the failed genericity or dependency rule is reported.

**Given** a stable component changes incompatibly
**When** the change is proposed
**Then** semantic-versioning policy is applied
**And** projects are not silently migrated to new behavior.

**Given** a project needs a novel split strategy or training component
**When** it seeks supported DSio execution
**Then** the component is contributed and tested through this admission path
**And** project-side runtime registration is not introduced.

### Story 5.2: Run the Supervised Reference Flow

As a new DSio consumer,
I want an executable supervised example using only public APIs,
So that I can understand and verify the complete supported experiment spine.

**Requirements:** FR37

**Acceptance Criteria:**

**Given** a clean supported environment with local MLflow tracking
**When** the synthetic supervised Prefect flow runs
**Then** it builds data, generates and logs a split, trains through the exact DSio Lightning classes, evaluates, exports a predictor, and performs inference
**And** every node is represented by correctly linked native MLflow evidence.

**Given** the same declared inputs and seed
**When** the flow is repeated in CI
**Then** split assignments, execution identities, deterministic outputs, and validated lineage are reproducible
**And** the flow requires no private dataset, consumer repository, remote scheduler, or deployment infrastructure.

**Given** successful model and dataset evidence from the first execution
**When** the example's downstream-only flow is run with changed evaluation configuration
**Then** it creates new evaluation evidence without retraining
**And** records immutable lineage to the reused sources.

**Given** a reader inspects the example
**When** following its workflow definition
**Then** orchestration is ordinary project-owned Prefect code
**And** no hidden DSio DAG, CLI, runner, result model, or promotion gate is required.

### Story 5.3: Run the Self-Supervised Reference Flow

As a new DSio consumer,
I want an executable self-supervised example using the same public training spine,
So that I can verify DSio supports a distinct training paradigm without a second framework.

**Requirements:** FR38

**Acceptance Criteria:**

**Given** a clean supported environment with local MLflow tracking
**When** the synthetic self-supervised Prefect flow runs
**Then** it uses the exact `DsioModule` and `DsioDataModule` classes with a named SSL objective and components
**And** introduces no SSL-specific runner or subclass.

**Given** a source batch with stable sample identities
**When** multi-view stochastic augmentation runs on the accelerator
**Then** declared views are created inside `training_step()` and remain tied to their source samples
**And** the same declared seed material reproduces the same views and training outcome within the supported environment.

**Given** the SSL flow completes
**When** its evidence is inspected
**Then** data, split, component, seed, training, evaluation, model, and environment provenance are linked through native MLflow records
**And** its predictor produces signature-valid and semantically valid outputs.

**Given** the supervised and self-supervised examples are compared
**When** their implementation paths are inspected
**Then** both use the same tracking, data, Lightning, evaluation, and inference public contracts
**And** variation is confined to named components and serializable configuration.
