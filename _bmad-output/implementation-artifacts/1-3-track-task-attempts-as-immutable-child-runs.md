---
baseline_commit: 033f9fad40b5919956f7a03c1d348fd8c7d09c80
---

# Story 1.3: Track Task Attempts as Immutable Child Runs

Status: done

## Story

As an experiment author,
I want every tracked Prefect task attempt recorded beneath its flow Run,
so that retries and failures remain auditable without overwriting earlier evidence.

## Acceptance Criteria

1. **Given** a task receives a valid parent Run reference, **when** a tracked task attempt begins, **then** DSio creates one new native MLflow child Run linked to that parent, **and** records the Prefect task key, task-run ID, dynamic key, and attempt number as MLflow tags.
2. **Given** Prefect retries a failed task, **when** the next attempt begins, **then** a new child Run is created for that attempt, **and** the failed child Run and all of its evidence remain unchanged.
3. **Given** a task attempt succeeds, fails, or is cancelled, **when** the attempt ends, **then** its child Run records `FINISHED`, `FAILED`, or `KILLED` respectively, **and** the original failure or cancellation continues through Prefect unchanged.
4. **Given** multiple sync or async tasks execute concurrently, **when** they create, log to, and update child Runs, **then** every child remains linked to the correct parent and Prefect task attempt, **and** no task discovers its parent or logging destination through MLflow's active-Run stack.
5. **Given** a tracked task is outside a Prefect task context, has no valid active parent reference, or MLflow cannot persist required lifecycle evidence, **when** it attempts to start or finish, **then** it fails closed with an actionable `TrackingError`, **and** it does not create an unlinked or falsely successful record.
6. **Given** task code logs parameters, metrics, artifacts, or models to the yielded child Run ID, **when** the task completes, **then** those values remain native MLflow evidence, **and** DSio creates no parallel task-result or evidence model.

## Tasks / Subtasks

- [x] Define the child-attempt boundary test-first (AC: 1, 4, 5, 6)
  - [x] Add focused tests for `dsio.tracking.attempt(parent_run_id)` against the isolated real MLflow store.
  - [x] Prove the context requires a Prefect `TaskRunContext`, yields MLflow's native `Run`, and never changes `mlflow.active_run()`.
  - [x] Prove native evidence logged with the yielded `run.info.run_id` remains attached to that child.
- [x] Create and identify one immutable Run per Prefect attempt (AC: 1, 2)
  - [x] Validate the explicit parent Run before child creation and use its experiment ID rather than ambient MLflow state.
  - [x] Link the child with MLflow's native `mlflow.parentRunId` tag and add the Prefect task key, task-run ID, dynamic key, and `run_count` as tags.
  - [x] Reconcile an uncertain create operation by a one-use creation token so a commit-then-raise failure cannot leave an untracked `RUNNING` child.
  - [x] Prove a Prefect retry produces a distinct child and preserves the failed attempt and its evidence unchanged.
- [x] Preserve exact lifecycle semantics without fluent active state (AC: 3, 5)
  - [x] Mark clean exit `FINISHED`, ordinary failure `FAILED`, and explicit cancellation/interruption `KILLED` through `MlflowClient.set_terminated` on the captured child ID.
  - [x] Re-raise the original task exception or cancellation object. Attach terminal-persistence context without masking an existing task failure.
  - [x] Fail a successful task closed when terminal persistence fails; never report success while its child remains nonterminal.
- [x] Prove concurrency through consumer-owned Prefect code (AC: 2, 4)
  - [x] Exercise concurrent tracked tasks beneath one explicit parent and assert exact parent/task metadata for every child.
  - [x] Exercise a real Prefect retry and assert one immutable child per `run_count`.
  - [x] Keep task topology, retry policy, submission, scheduling, and output validation in the consumer flow.
- [x] Preserve the package and architecture boundaries (AC: 4-6)
  - [x] Export only `attempt` beside the existing `experiment` and `TrackingError`; add no decorator, task wrapper, context/result dataclass, repository, or Prefect hook.
  - [x] Keep root `import dsio` inert and update the installed-wheel proof and README example.
  - [x] Split tracking implementation files by lifecycle responsibility rather than enlarging `experiment.py` or duplicating its error semantics.
- [x] Run the complete verification gate (AC: 1-6)
  - [x] Focused child-attempt, retry, concurrency, distribution, and import-safety tests.
  - [x] Full pytest, Ruff, mypy, and import-linter suite.
  - [x] Lock validation, package build, and diff checks.

### Review Findings

- [x] [Review][Patch] Reject a soft-deleted parent even when MLflow retains its `RUNNING` status. [`src/dsio/tracking/attempt.py`]
- [x] [Review][Patch] Reconcile and terminate a committed child when cancellation lands after creation but before body entry. [`src/dsio/tracking/attempt.py`]
- [x] [Review][Patch] Verify every concurrent child's Prefect metadata tags against its own native task context. [`tests/tracking/test_attempt.py`]
- [x] [Review][Patch] Preserve cancellation during recovery from a failed successful-path terminal write and recover the exact child as `KILLED`. [`src/dsio/tracking/attempt.py`]

## Dev Notes

### Current State and Required Delta

- Story 1.2 added `dsio.tracking.experiment(...)`: one explicit fresh native parent Run per flow execution, exact lifecycle propagation, and fail-closed uncertain-create reconciliation.
- `src/dsio/tracking/experiment.py` is already a substantial lifecycle module. Do not append child behavior to it. Extract only genuinely shared lifecycle helpers and keep parent- and child-specific policy in separate modules.
- `tests/conftest.py` now resets MLflow's tracking URI for every test, matching the documented isolation guarantee.
- The active README passes `parent.info.run_id` as ordinary task input. This story adds the child boundary inside that task without adding a DSio flow or task decorator.

### Minimal Public Contract

```python
from mlflow import MlflowClient
from prefect import task

from dsio.tracking import attempt


@task(retries=1)
def identify(parent_run_id: str) -> str:
    with attempt(parent_run_id) as child:
        MlflowClient().log_param(child.info.run_id, "source", "algae")
        return child.info.run_id
```

- `attempt(parent_run_id)` reads only the current Prefect `TaskRunContext` for authoritative task metadata. The parent ID remains explicit ordinary task input.
- Yield MLflow's native `Run`, not a DSio wrapper and not an `ActiveRun`.
- Do not activate the child through `mlflow.start_run`. MLflow's active stack is thread-local rather than async-task-local; explicit run-ID logging is the minimal boundary that remains correct for concurrent sync and async Prefect tasks.
- Native evidence APIs or framework integrations must target `child.info.run_id` explicitly. DSio does not proxy parameters, metrics, artifacts, models, or Lightning logging.

### Lifecycle and Metadata Requirements

- Obtain `TaskRunContext` before creating anything. A flow context or no context is an actionable tracking error.
- Validate the parent exists, is `RUNNING`, and is a top-level Run before creating the child. Create the child in the parent's experiment.
- Use MLflow's reserved `mlflow.parentRunId` tag for the relationship. Use a small stable set of DSio-owned Prefect tags for `task_key`, `task_run_id`, `dynamic_key`, and one-based `run_count`; do not serialize a context model.
- Every context entry creates a new child. Never search by task identity to resume, overwrite, or deduplicate a prior attempt.
- Reuse the explicit cancellation classification and uncertain-create discipline proven by Story 1.2. Refactoring must preserve every parent lifecycle regression test.
- If child creation commits and the response is lost, search only by the unique creation token and terminate the exact reconciled child with the failure/cancellation status.
- If finalization fails after the task body failed, retain the original exception and add actionable tracking context. If a clean body cannot be finalized, raise `TrackingError`.

### Architecture Compliance

- Prefect remains the only DAG, task, retry, and concurrency model. DSio reads native task context only at the tracking boundary.
- MLflow remains the only Run, relationship, status, and evidence model. DSio holds only local IDs long enough to enforce lifecycle invariants.
- No global current-parent accessor, decorator, task wrapper, retry handler, event subscriber, result class, status enum, logging facade, or orchestration configuration.
- Execution identity and normalized safe configuration belong to Story 1.4. Evidence resolution and caching belong to Story 1.5. Do not pull either concern into attempt tags.
- Keep `src/dsio/__init__.py` declarative and root import free of eager MLflow or Prefect imports.

### Testing Requirements

- Use real MLflow storage for lifecycle, linkage, evidence, retry, and concurrency behavior. Mock only forced commit-then-raise and terminal-persistence failures.
- Run real Prefect tasks for authoritative task metadata and retry numbering. Do not manufacture a parallel task context DTO.
- Assert concurrent tasks have distinct child IDs, exact parent tags, correct Prefect tags, and no active MLflow Run inside task code.
- Inspect the built wheel and run the representative flow outside the checkout.
- Keep live external services out of the default gate.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-13-Track-Task-Attempts-as-Immutable-Child-Runs]
- [Source: docs/superpowers/specs/2026-09-18-generic-experiment-spine.md#Workflow-and-tracking]
- [Source: docs/adr/0019-versioned-library-with-project-owned-prefect-flows.md]
- [Source: _bmad-output/implementation-artifacts/1-2-track-each-flow-execution-as-a-parent-run.md]
- [MLflow: `MlflowClient`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
- [MLflow: Parent and child Runs](https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/part1-child-runs/)
- [Prefect: Runtime context](https://docs.prefect.io/v3/concepts/runtime-context/)

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-21: Created from merged Story 1.2 at `033f9fa`; selected an explicit-run-ID child boundary to remain correct under both threaded and async Prefect concurrency.
- 2026-09-21: Implemented the child lifecycle test-first; focused parent/child, Prefect, and distribution gate passed with 40 tests.
- 2026-09-21: Complete pre-review gate passed with 605 tests, Ruff, mypy, four import contracts, lock validation, build, and diff checks.
- 2026-09-21: Adversarial review found four lifecycle/acceptance gaps; all were patched test-first and the updated tracking gate passed with 40 tests.
- 2026-09-21: Final blind, edge-case, and acceptance passes approved the complete patched diff with no remaining actionable findings.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Story is implementation-ready with a native MLflow/Prefect contract and no new DSio persistence or orchestration model.
- Added `attempt(parent_run_id)` yielding a native MLflow `Run` and using only explicit Run IDs for concurrent-safe evidence logging.
- Recorded native parent linkage plus authoritative Prefect task and retry metadata without adding a task wrapper or context model.
- Shared creation reconciliation and cancellation classification between parent and child lifecycles in one private module.
- Proved immutable retries, same-thread async overlap, exact cancellation propagation, fail-closed terminal persistence, consumer flow use, and installed-wheel execution.
- Pre-review gate: 605 tests passed and 3 live tests were intentionally deselected; all static, architecture, lock, build, and diff checks passed.
- Review patches reject deleted parents, close the committed-child/pre-yield cancellation window by exact creation token, and pin concurrent child metadata to authoritative Prefect contexts.
- Nested terminal-write recovery preserves a cancellation signal and makes a final exact-ID `KILLED` persistence attempt instead of wrapping it as `TrackingError`.
- Final post-review gate: 609 tests passed and 3 live tests were intentionally deselected; Ruff, mypy, four import contracts, lock validation, build, and diff checks passed.

### File List

- `_bmad-output/implementation-artifacts/1-3-track-task-attempts-as-immutable-child-runs.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `src/dsio/tracking/__init__.py`
- `src/dsio/tracking/_lifecycle.py`
- `src/dsio/tracking/attempt.py`
- `src/dsio/tracking/experiment.py`
- `tests/tracking/attempt/test_lifecycle.py`
- `tests/tracking/attempt/test_prefect.py`
- `tests/test_project_flow.py`
- `tests/test_built_distribution.py`

### Change Log

- 2026-09-21: Created Story 1.3 and marked it ready for development.
- 2026-09-21: Implemented Story 1.3 and moved it to review after the complete verification gate passed.
- 2026-09-21: Applied all first-round adversarial review findings and requested final panel approval.
- 2026-09-21: Received unanimous final approval and marked Story 1.3 done after the post-review gate passed.
