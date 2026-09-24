# DSio — Generic Experiment Spine

Status: accepted (2026-09-18), implemented; tracking hierarchy amended by ADR-0020

Supersedes: `2026-08-20-dsio-lean-design.md`

## Goal

DSio is an installable, versioned Python package for building reproducible machine-learning experiment workflows across independent projects. Projects consume released package versions through their dependency lockfiles; they do not clone, vendor, or merge DSio source. DSio provides one dependable training path and reusable data, evaluation, and inference components without owning project lifecycle, promotion, deployment, or infrastructure.

The governing rule is:

> Use Prefect, Lightning, PyTorch, and MLflow natively. DSio adds only reusable components and invariants those products do not provide.

## Boundaries

DSio owns:

- execution identity, provenance, required-output validation, and MLflow coordination;
- one canonical data-store API and the format selected by a focused benchmark;
- split generation, immutable split manifests, and leakage validation;
- the only supported `DsioModule` and `DsioDataModule` training composition roots;
- reusable datasets, samplers, batching, collation, transforms, networks, objectives, optimizer factories, callbacks, metrics, prediction schemas, and inference;
- the admission and compatibility policy for battle-tested Components.

Consuming projects own:

- Prefect flow topology and typed flow configuration;
- domain-specific data production, labels, task meaning, and acceptance policy;
- local Components before they have earned admission to DSio;
- promotion decisions, deployment, project lifecycle, and remote scheduling;
- whether inference is windowed, streaming, batch, or otherwise deployed.

DSio makes no assumption about where a project trains or serves a Predictor.

## Native systems

**Prefect owns orchestration.** A project-owned Prefect flow is the only DAG definition. DSio does not mirror it in a graph model, provide an orchestration protocol, or wrap it in a DSio CLI. Projects call flows directly as Python or through Prefect's native tooling.

**Lightning owns training lifecycle.** Training loops, device strategies, optimization hooks, callbacks, checkpointing, and metric emission use native Lightning behavior. Training, validation, and test metrics originate from `LightningModule.log()`.

**MLflow owns evidence.** Runs, datasets, metrics, artifacts, Logged Models, signatures, evaluation, and registry entries remain native MLflow entities. DSio does not define parallel Run, Dataset, Artifact, Evaluation Result, or Model Registry abstractions.

## Workflow and tracking

A project flow calls `dsio.tracking.resolve_experiment(...)` to resolve a native MLflow Experiment without creating a Run. Every evidence-producing Experiment Node is a Prefect task and logs its Evidence to a visible top-level MLflow Run created by `dsio.tracking.attempt(experiment_id)`.

Each retry creates a new Run tagged with the same Execution Identity and its native Prefect flow-run, task-run, dynamic-key, and attempt identities. Failed Attempts are never rewritten. A new Attempt may explicitly consume a prior checkpoint. Only a verified successful Attempt is reusable.

Prefect owns flow execution identity, topology, and terminal status; DSio does not duplicate them in an empty MLflow flow Run. Attempts belonging to one flow execution share the Prefect flow-run tag, while immutable MLflow inputs and source-Run references express evidence lineage. Nodes may reuse verified Evidence from earlier Experiments by immutable MLflow Run, Logged Model, or artifact reference; reused Evidence is referenced as an input rather than copied or reopened.

Each Attempt records its own successful, failed, or cancelled terminal status in MLflow, while the corresponding flow status remains authoritative in Prefect. No task or flow failure is converted into a partial success.

Pure Prefect tasks may use Prefect result caching with DSio's deterministic cache-key utility. MLflow-producing tasks instead query MLflow by Execution Identity and verify required native outputs before reusing them. A cached identifier never overrules missing or invalid Evidence.

Project configuration remains project-owned. The normalized resolved configuration is logged as an artifact, while DSio functions accept only the values they need. Credentials are never logged; non-secret resource identity must still be explicit.

## Execution identity and provenance

Execution Identity includes every relevant input capable of changing output:

- normalized node inputs and project configuration;
- Dataset Identity and Split Manifest digest;
- seed and deterministic randomness inputs;
- named Component identities and versions;
- consumer repository commit and dirty patch;
- dependency lockfile digest, DSio version and package digest;
- relevant code identity, command, and environment;
- batching or hardware topology when it changes sample composition or numerical behavior.

Changing only a downstream evaluation node preserves reusable upstream Evidence because identity belongs to each DAG node rather than to a monolithic Experiment result.

## Training system

`DsioModule` and `DsioDataModule` are the only supported training classes. Projects instantiate them directly and do not subclass them. Variation enters through named, importable Components; lambdas and closures are rejected because their identity and reproduction are ambiguous.

`DsioModule` is a composition root, not a task-mode dispatcher. It accepts:

- a model/network;
- an objective step callable receiving `(model, batch, stage)` and returning a mapping with mandatory `loss` and optional metrics;
- an optimizer and scheduler factory returning Lightning-native configuration;
- metrics and native Lightning callbacks;
- a prediction callable returning schema-bound outputs.

There are no branches on project or task names. If a new paradigm cannot fit, DSio evolves the generic composition contract after a concrete use proves the need; it does not create a project escape hatch or add a mode flag.

Every batch is a mapping containing `sample_id`. Remaining fields belong to the injected objective. Prediction output contains the same `sample_id` plus the declared Prediction Schema fields.

## Augmentation ownership

Augmentation has three exclusive lanes:

1. `DsioDataModule` and `dsio.data` own CPU decoding, sampling, windowing, shape preparation, batching, and collation.
2. `DsioModule.training_step()` exclusively owns stochastic, train-only accelerator augmentation after device transfer and before the objective.
3. The Predictor owns deterministic preprocessing required identically during training and inference.

`DsioDataModule.on_after_batch_transfer()` remains uncustomized. `forward()`, validation, test, prediction, and exported inference never invoke training augmentation.

Augmentation randomness is derived from stable execution seed, epoch, Sample Identity or ordered batch membership, view identity, and augmentation identity. Retry and checkpoint resume reproduce augmentation exactly for unchanged Execution Identity. CPU/accelerator placement is explicit; unsupported device or dtype combinations fail preflight rather than silently falling back.

## Data and splits

DSio exposes one stable storage API backed by one selected format. There is no public backend registry. Selection follows a focused benchmark covering ingestion, contiguous and random windows, concurrent readers, memory use, disk size, and interrupted-write recovery across declared synthetic workloads and at least one representative real corpus.

Split generation lives under `dsio.data.splits`. The public entry point is one closed DSio-owned dispatcher. Projects cannot register splitters at runtime. DSio delegates established algorithms to trusted libraries where appropriate, then adds identity, validation, and manifest serialization.

A Split Manifest is one versioned metadata document plus content-addressed membership arrays logged together as an MLflow artifact directory. It records Dataset Identity, algorithm identity and parameters, seed, named partitions and roles, validation results, and its digest. Source data is also recorded through native MLflow Dataset inputs and contexts.

Roles are named rather than fixed to train/validation/test. `DsioDataModule` maps declared roles to Lightning phases. A novel splitter must enter DSio's experimental namespace through review, determinism and property tests, leakage checks, and one real use; it becomes stable after an unrelated second use.

`DsioDataModule` consumes a Split Manifest and named dataset factory, constructs stage datasets in `setup(stage)`, and exclusively owns deterministic `DataLoader` construction. Reusable datasets, samplers, batch samplers, transforms, and collators live under `dsio.data`.

## Metrics, evaluation, and inference

DSio does not define a universal metric interface. `DsioModule` consumes native TorchMetrics components and logs through Lightning. Experiment-level evaluation is an ordinary Prefect node using native MLflow evaluation and DSio-owned native metric implementations where the ecosystem lacks them. Validation conditions report scientific acceptance; they do not trigger promotion or deployment.

The MLflow Model Signature is the canonical Predictor interface. A named DSio validation function enforces task-family output constraints that MLflow does not validate. DSio does not create row-level prediction result classes.

`dsio.inference.predict(model_uri, inputs)` loads the model through MLflow PyFunc, invokes it, preserves `sample_id`, and validates the declared output. Export uses the native MLflow PyTorch flavor when its tensor interface fits and a native MLflow PyFunc when deterministic preprocessing or structured prediction requires it.

A resumable Lightning checkpoint contains the full Training System. An MLflow Logged Model contains only deterministic preprocessing, prediction behavior, signature, example, and predictor state. Predictor export is required only for DAG nodes that declare a Predictor output.

## Package shape

Top-level packages represent cohesive pipeline domains, not implementation technologies:

```text
dsio/
├── data/       # storage, datasets, loading, split generation and manifests
├── model/      # reusable model components
├── train/      # DSio Lightning classes, objectives, optimization, GPU augmentation
├── metrics/    # native reusable metric implementations
├── inference/  # prediction schemas, export, loading and validation
└── tracking/   # thin MLflow coordination and provenance
```

There is no `dsio.torch` package. Start a cohesive concern as one module. Promote it to a same-named package only when distinct internal responsibilities create navigation, testing, or review friction; preserve the public import path through `__init__.py` re-exports.

## Component admission and compatibility

DSio ships as one distribution with Prefect, Lightning, PyTorch, and MLflow as required dependencies. Projects pin released DSio versions in their lockfiles.

An Experimental Component requires one real downstream confirmation Experiment, generic naming and configuration, no project imports, deterministic behavior, provenance, documentation, relevant integration or property tests, and adversarial multi-agent review. A Stable Component additionally requires an unrelated second use, compatibility tests, migration discipline, and human approval.

Experimental APIs may change between minor releases. Stable APIs follow semantic versioning and require migration notes for breaking changes. Support claims cover only environments exercised by the published compatibility test matrix.

## First generic acceptance slice

The target architecture is proven by synthetic reference flows, not by prematurely migrating Pulse or Algua. One supervised and one self-supervised DAG must demonstrate:

- canonical storage and immutable Dataset Identity;
- DSio-owned split generation and Split Manifest validation;
- the exact `DsioModule` and `DsioDataModule` path;
- deterministic CPU preparation and accelerator augmentation boundaries;
- MLflow provenance, Evidence, evaluation, and optional Predictor export;
- failed Attempt visibility, retry, verified reuse, and downstream-only reevaluation;
- generic schema-validated inference.

## Explicit non-goals

DSio does not own promotion policy, deployment, project lifecycle, remote scheduling, project-specific label meaning, streaming/windowing serving strategy, or infrastructure provisioning. It does not provide an orchestration abstraction, backend plugin system, universal trainer, parallel tracking model, or project-specific CLI.
