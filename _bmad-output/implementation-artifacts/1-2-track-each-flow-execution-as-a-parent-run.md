---
baseline_commit: c88cc280316ab612bbfc6bc7eb297d5da8d4fd63
---

# Story 1.2: Track Each Flow Execution as a Parent Run

Status: done

## Story

As an experiment author,
I want an explicit MLflow tracking context around my Prefect flow execution,
so that every execution has one auditable record with an accurate terminal state.

## Acceptance Criteria

1. **Given** valid MLflow tracking and experiment configuration, **when** the flow enters the DSio tracking context, **then** exactly one new native MLflow parent Run is created, **and** its Run ID is available to tasks without introducing a duplicate DSio experiment model.
2. **Given** the same logical experiment is executed or retried more than once, **when** each execution enters its tracking context, **then** each execution receives a new parent Run, **and** previous Runs and their evidence remain unchanged.
3. **Given** the Prefect flow and its required output validation complete successfully, **when** the tracking context exits, **then** the parent Run receives the successful MLflow terminal status.
4. **Given** the flow raises, required output validation fails, or execution is cancelled, **when** the tracking context exits, **then** the parent Run records the corresponding non-successful terminal status, **and** the original exception or cancellation remains visible to Prefect.
5. **Given** MLflow cannot create or update the required Run, **when** execution enters or exits the tracking boundary, **then** DSio fails closed with an actionable error, **and** it does not continue while pretending the execution is tracked.

## Tasks / Subtasks

- [x] Define the public parent-run boundary test-first (AC: 1, 2)
  - [x] Add focused tests for `dsio.tracking.experiment(...)` using an isolated native MLflow file store.
  - [x] Prove the yielded object is MLflow's native `ActiveRun`, exposes `info.run_id`, and is `RUNNING` while the context body executes.
  - [x] Prove two entries for the same experiment create distinct Run IDs without changing the first Run or its evidence.
- [x] Implement the minimal MLflow context (AC: 1-5)
  - [x] Create the `dsio.tracking` package with one small public `experiment(...)` context manager and an actionable `TrackingError`.
  - [x] Resolve or create the named native MLflow Experiment, handling only the concurrent create race before creating exactly one fresh parent Run.
  - [x] Activate and yield the newly created native MLflow `ActiveRun`; do not create a DSio Experiment, Run, result, context, status, repository, or configuration model.
  - [x] Mark clean exits `FINISHED`, ordinary and validation exceptions `FAILED`, and cancellation `KILLED` using native MLflow status values.
  - [x] Re-raise the original body exception or cancellation unchanged. If finalization also fails, retain an actionable tracking failure on that original error rather than hiding either fact.
  - [x] Fail before executing the body when experiment resolution or Run creation fails, and fail the successful path when terminal-state persistence fails.
- [x] Prove the consumer-owned Prefect boundary (AC: 1, 3, 4)
  - [x] Extend the project-owned flow test to open the context inside the flow, pass `parent.info.run_id` explicitly to task code, and validate required output before context exit.
  - [x] Assert the parent status from MLflow after the flow, including failure and cancellation propagation where the focused context tests provide the narrower proof.
  - [x] Keep Prefect flow topology, invocation, retries, and scheduling entirely in consumer code; add no DSio flow wrapper, decorator, hook, runner, or CLI.
- [x] Preserve the installed-library and import contracts (AC: 1, 5)
  - [x] Exercise the new public context from the installed wheel outside the checkout and inspect its native MLflow Run.
  - [x] Keep root `import dsio` inert and free of eager MLflow or Prefect imports; do not re-export `experiment` from the package root.
  - [x] Update the active README example to show explicit parent-ID propagation and output validation inside the context.
- [x] Run the complete verification gate (AC: 1-5)
  - [x] Focused tracking, consumer-flow, distribution, and import-safety tests.
  - [x] Full `uv run pytest -q` regression suite.
  - [x] `uv run ruff check .`, `uv run mypy`, and `uv run lint-imports`.
  - [x] `uv lock --check`, `uv build`, and `git diff --check`.

### Review Findings

- [x] [Review][Patch] Finalize only the captured parent Run and reject missing or mismatched fluent active state. [`src/dsio/tracking/experiment.py`:93]
- [x] [Review][Patch] Finalize the created parent when cancellation interrupts activation. [`src/dsio/tracking/experiment.py`:46]
- [x] [Review][Patch] Preserve activation context when cleanup raises any ordinary exception. [`src/dsio/tracking/experiment.py`:53]
- [x] [Review][Patch] Classify only explicit cancellation and interruption types as `KILLED`; record other `BaseException` failures as `FAILED`. [`src/dsio/tracking/experiment.py`:62]
- [x] [Review][Patch] Close the cancellation window between successful activation and body protection. [`src/dsio/tracking/experiment.py`:47]
- [x] [Review][Patch] Preserve cancellation raised during experiment resolution or parent creation. [`src/dsio/tracking/experiment.py`:43]
- [x] [Review][Patch] Recover exact parent status and preserve the body error when finalization is interrupted. [`src/dsio/tracking/experiment.py`:117]
- [x] [Review][Patch] Recognize cancellation-only `BaseExceptionGroup` values without misclassifying mixed failures. [`src/dsio/tracking/experiment.py`:105]
- [x] [Review][Patch] Preserve `Exception`-derived cancellation raised during terminal persistence. [`src/dsio/tracking/experiment.py`:161]
- [x] [Review][Patch] Reconcile a parent Run when `create_run` commits before raising. [`src/dsio/tracking/experiment.py`:41]

## Dev Notes

### Current State and Required Delta

- Story 1.1 established an installable library, inert root import, required MLflow/Prefect dependencies, and consumer-owned Prefect flows. It intentionally added no generic tracking coordination because this story owns that boundary.
- `src/dsio/train/tracking.py` is legacy training coordination tied to `RunConfig`, the local scratch `Run`, Lightning's `MLFlowLogger`, and the old fold runner. It is not the generic flow-parent API. Preserve it unchanged until the training stories replace that vertical; do not import it from the new package.
- `src/dsio/runs/record.py` is legacy scratch/provenance state. It must not become the flow parent or leak into this API.
- `tests/conftest.py` already gives each test an isolated real MLflow `file:` store. Use the real `MlflowClient` for lifecycle behavior and mocks only for forced persistence failures.

### Technical Requirements

- Public call: `dsio.tracking.experiment(experiment_name, *, run_name=None)`. MLflow's own tracking configuration remains authoritative; project configuration remains project-owned.
- Create the Run explicitly with `MlflowClient.create_run`, then activate that exact fresh ID with `mlflow.start_run(run_id=...)` and yield the native `mlflow.ActiveRun`. Explicit creation prevents inherited `MLFLOW_RUN_ID` from resuming prior evidence. Refetch the Run to observe its terminal status after exit.
- Reject an ambient active Run before creating anything; this boundary creates a top-level parent and must not silently nest, resume, or terminate a Run it does not own.
- Consumers pass `parent.info.run_id` explicitly. Tasks must not discover the parent through MLflow's process-local active-run state.
- Resolve an existing active experiment by name or create it. A get-then-create race may retry only after MLflow reports `RESOURCE_ALREADY_EXISTS`, by refetching the experiment. Do not retry arbitrary failures or create a second parent Run.
- Status mapping uses native strings: clean exit to `FINISHED`; ordinary exception and output-validation exception to `FAILED`; `asyncio.CancelledError`, `concurrent.futures.CancelledError`, Prefect `CancelledRun`, Prefect `TerminationSignal`, and `KeyboardInterrupt` to `KILLED`.
- Catch `BaseException` around the body so cancellations and interrupts are finalized, then re-raise the same object. Never turn a Prefect-visible failure into a successful return.
- On terminal update failure after a successful body, raise `TrackingError` naming the operation, experiment, and Run ID. When the body already failed, add an actionable note to the original exception and re-raise it so both the original failure and tracking failure remain observable.
- Required-output validation is ordinary project code inside the context. Do not add a callback, validation protocol, result dataclass, or evaluation model.

### Architecture Compliance

- Prefect owns the DAG; MLflow owns Experiments, Runs, evidence, and terminal states. DSio adds only the explicit lifecycle invariant.
- Every context entry creates a new parent Run, even for identical inputs or a retried flow. Never query by name or identity to resume, reopen, overwrite, or reuse a parent.
- Child task Runs, attempt numbers, Prefect task-run tags, and retry handling belong to Story 1.3. Identity/configuration/provenance belong to Story 1.4. Evidence resolution and caching belong to Story 1.5.
- Do not add a DSio `Experiment`, `Run`, `TrackingContext`, status enum, repository/service layer, adapter protocol, event system, global current-parent accessor, DAG wrapper, or orchestration configuration.
- Do not address the three deferred Story 1.1 provenance/replay findings in this story.
- Keep `src/dsio/__init__.py` declarative. Importing `dsio` must not import MLflow or Prefect, resolve an experiment, create a Run, or touch external state.

### File Structure Requirements

Expected additions:

- `src/dsio/tracking/__init__.py`
- `src/dsio/tracking/experiment.py`
- `tests/tracking/test_experiment.py`

Expected updates:

- `tests/test_project_flow.py`
- `tests/test_built_distribution.py`
- `README.md`
- this story file and `sprint-status.yaml`

`dsio.tracking` is intentionally a package, matching the accepted public package layout. Keep the implementation cohesive and small; add no placeholder modules for future stories.

### Testing Requirements

- Real local MLflow tests must assert the exact native type, distinct IDs, `RUNNING` state during the body, and refetched `FINISHED`, `FAILED`, or `KILLED` state after exit.
- Preserve evidence immutability by writing a native tag/parameter to the first Run, executing the second context, and showing the first Run remains terminal and unchanged.
- Test an ordinary exception, validation exception, `asyncio.CancelledError`, `KeyboardInterrupt`, Run-creation failure, success-path finalization failure, and body-failure-plus-finalization-failure.
- For exception propagation, assert identity where practical (`caught.value is original`) rather than merely matching text.
- The consumer-flow test must pass the parent ID as a normal task argument. Do not rely on MLflow's process-global active Run inside a task.
- The built-wheel probe must run outside the repository and use its isolated MLflow store so it cannot accidentally import source or mutate host tracking state.
- No Docker or live MLflow server belongs in the default gate.

### Previous Story Intelligence

- Start from merged main at `c88cc280316ab612bbfc6bc7eb297d5da8d4fd63` or newer. Preserve the untracked user-owned review reports and keep them outside build artifacts.
- Python remains `>=3.12,<3.15`; full `mlflow>=3,<4` and `prefect>=3.8,<4` are already required. The lock currently resolves MLflow 3.16.1. Do not change dependency policy.
- Configure test environment variables before importing Prefect or MLflow when those settings govern state isolation.
- Story 1.1's strongest tests inspected installed artifacts and exact native state. Continue that pattern instead of asserting against a DSio shadow representation.
- Historical ADRs/plans are evidence. Update active guidance only; do not rewrite frozen history.

### Git Intelligence

- PR #34 established the current package boundary and consumer-owned flow test in merge commit `c88cc28`.
- The previous change deliberately left `dsio.tracking` absent until it had real behavior. This story supplies that behavior without reorganizing unrelated data, split, model, or training modules.

### Latest Technical Information

- Locked MLflow 3.16.1 exposes `MlflowClient.create_run(...) -> mlflow.entities.Run`, `mlflow.start_run(run_id=...) -> mlflow.ActiveRun`, and native terminal-state operations. Explicitly passing the newly created ID bypasses environment-driven resumption.
- MLflow's fluent `ActiveRun.__exit__` maps every exceptional exit to `FAILED`; DSio needs an explicit client lifecycle to preserve the accepted `KILLED` cancellation state.
- Prefect cancellation can surface through `TerminationSignal`, `CancelledRun`, or an async cancellation. These are not all ordinary `Exception` subclasses.

### Project Structure Notes

- The package name follows pipeline responsibility (`tracking`), not implementation technology or project identity.
- The public import is `from dsio.tracking import experiment`; do not add a `dsio.mlflow` or `dsio.torch` namespace.
- Passing an immutable Run ID is the stable process boundary. The yielded native Run itself is not a DSio transfer object.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-12-Track-Each-Flow-Execution-as-a-Parent-Run]
- [Source: docs/superpowers/specs/2026-09-18-generic-experiment-spine.md#Workflow-and-tracking]
- [Source: docs/superpowers/specs/2026-09-18-generic-experiment-spine.md#Native-systems]
- [Source: docs/superpowers/specs/2026-09-18-generic-experiment-spine.md#Package-shape]
- [Source: docs/adr/0019-versioned-library-with-project-owned-prefect-flows.md]
- [Source: CONTEXT.md#Experimentation]
- [Source: _bmad-output/implementation-artifacts/1-1-install-dsio-and-run-a-project-owned-flow.md]
- [MLflow: `MlflowClient`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
- [MLflow: Run status](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunStatus)
- [Prefect: Exceptions](https://reference.prefect.io/prefect/exceptions/)

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-21: Started from merged main at `c88cc28`; confirmed public test seams for the tracking context, consumer-owned Prefect flow, and installed wheel.
- 2026-09-21: Implemented the parent lifecycle through native MLflow entities and client/fluent APIs; focused gate passed with 21 tests.
- 2026-09-21: Complete gate passed: 581 tests, Ruff, mypy, four import contracts, lock validation, build, and diff checks.
- 2026-09-21: Three adversarial review rounds found and closed ten lifecycle edge cases; the final blind, edge-case, and acceptance passes approved the complete diff with no remaining findings.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Added one thin `dsio.tracking.experiment(...)` context yielding native `ActiveRun` objects, with fresh parent identity and fail-closed terminal-state persistence.
- Preserved exact exception/cancellation propagation while recording `FAILED` or `KILLED`, including actionable notes when MLflow finalization also fails.
- Hardened activation, finalization, grouped-cancellation, and uncertain-creation paths while keeping the public API to one context manager and one error type.
- Proved explicit parent-ID propagation through a consumer-owned Prefect flow and through the installed wheel outside the checkout.
- Final adversarial review: blind, edge-case, and acceptance layers all gave substantive approval with no remaining actionable findings.
- Final post-review gate: 596 tests passed and 3 live tests were intentionally deselected; Ruff, mypy, four import contracts, lock validation, build, and diff checks passed.

### File List

- `README.md`
- `pyproject.toml`
- `src/dsio/tracking/__init__.py`
- `src/dsio/tracking/experiment.py`
- `tests/tracking/test_experiment.py`
- `tests/conftest.py`
- `tests/test_project_flow.py`
- `tests/test_built_distribution.py`
- `_bmad-output/implementation-artifacts/1-2-track-each-flow-execution-as-a-parent-run.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

### Change Log

- 2026-09-21: Created Story 1.2 and marked it ready for development.
- 2026-09-21: Implemented Story 1.2 and moved it to review after the complete verification gate passed.
- 2026-09-21: Closed all adversarial review findings and marked Story 1.2 done after unanimous final approval.
