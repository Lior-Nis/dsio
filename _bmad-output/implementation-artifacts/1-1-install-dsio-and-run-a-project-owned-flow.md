---
baseline_commit: e31274f29c4be4267fba770818f8a91b7de98134
---

# Story 1.1: Install DSio and Run a Project-Owned Flow

Status: done

## Story

As a consumer-project developer,
I want to install DSio and use its functions from my own Prefect flow,
so that my project owns its workflow without cloning DSio or adopting a parallel orchestration layer.

## Acceptance Criteria

1. **Given** a clean supported Python environment, **when** the project installs the built DSio distribution, **then** `import dsio` succeeds with the documented required dependencies, **and** importing the package creates no MLflow Run, Prefect flow, files, or other external state.
2. **Given** a consumer project with an ordinary `@flow` and `@task`, **when** those functions call a minimal public DSio function, **then** the flow executes through normal Python and Prefect entry points, **and** no DSio DAG class, runner registry, backend registry, or required CLI is involved.
3. **Given** a built wheel or source distribution, **when** its contents are inspected and installed outside the repository, **then** the public package and required metadata are present, **and** consumer-project source, repository-relative imports, and clone-based templates are not required.
4. **Given** the public surface supported by this story, **when** a consumer imports DSio functionality, **then** new supported functionality is exposed through a responsibility-named package rather than a technology or project namespace, **and** no `dsio.torch` namespace or project-specific branch is introduced. Legacy top-level modules awaiting later vertical migration are not documented as the target public layout.

## Tasks / Subtasks

- [x] Start from the current mainline without losing planning artifacts (AC: 1-4)
  - [x] Base the implementation branch on the current local/remote `main` merge base; do not implement on the stale `artifacts/drop-model-registry` tip.
  - [x] Preserve all uncommitted planning documents and the user-owned `architecture_review.md`, `claude_review.md`, `codex_review.md`, and `opencode_review.md`; do not edit or package the review files.
  - [x] Re-run the focused baseline after the branch is current before changing behavior.
- [x] Publish one installable dependency contract (AC: 1, 3)
  - [x] Keep the existing Hatchling `src/dsio` wheel layout; do not introduce another build backend or workspace.
  - [x] Declare `prefect>=3.8,<4`, `torch>=2.7,<3`, `lightning>=2.5,<3`, `torchmetrics>=1.7,<2`, and full `mlflow>=3,<4` as ordinary required runtime dependencies in `[project].dependencies`; regenerate `uv.lock` and record the exact resolved versions there.
  - [x] Remove Typer, the `dsio` console-script entry point, and the `cpu`/`gpu` DSio extras. Published metadata must be accelerator-source-neutral; consumer projects choose their PyTorch wheel/index in their own lockfile.
  - [x] It is acceptable for DSio's development lock to resolve the CPU PyTorch index for CI, but that choice must remain tool configuration rather than wheel metadata.
  - [x] Add a narrow Hatch sdist inclusion policy so source releases contain the package and required build/readme metadata, not `_bmad-output`, internal reviews, repository automation, or unrelated project files.
- [x] Remove the obsolete supported orchestration surface (AC: 2, 4)
  - [x] Delete `src/dsio/cli/`, `src/dsio/application.py`, their tests, and their import-linter references; do not leave a dead console entry point.
  - [x] Delete `src/dsio/presets.py`, `src/dsio/config/presets.py`, and `src/dsio/config/overrides.py`; remove the `preset` export from `dsio.config`.
  - [x] Delete `tests/config/test_preset_discovery.py` and `tests/train/test_runner_bootstrap.py`, and remove only the preset/override cases from `tests/config/test_config.py`; preserve its still-valid schema, hashing, and registry invariants.
  - [x] Keep the still-used legacy training runner internals until Stories 3.1-3.2 replace them, but do not route the new flow through them or document them as supported orchestration APIs.
  - [x] Delete the CLI-entrypoint Docker image rather than converting it into a DSio deployment artifact; deployment and infrastructure remain consumer concerns.
  - [x] Preserve `src/dsio/contracts/hashing.py`; use its existing public `sha256_of` function in the consumer-flow proof instead of inventing a demonstration API.
  - [x] Keep `src/dsio/__init__.py` declarative and minimal. Remove the stray historical comment, but do not add eager imports of Prefect, MLflow, Lightning, or PyTorch.
  - [x] Do not reorganize data, splits, training, tracking, metrics, or inference in this story. Do not create empty `tracking`, `metrics`, or `inference` placeholder packages; later vertical stories create them when they have real behavior.
- [x] Replace clone-and-run documentation with library consumption (AC: 2-4)
  - [x] Rewrite the README quick start around a pinned DSio dependency and a project-owned Python module containing ordinary Prefect `@flow` and `@task` functions.
  - [x] Make the example use `from dsio.contracts import sha256_of`, invoke the flow as a normal Python function, and avoid `.submit()`, deployments, workers, scheduling, or a DSio wrapper.
  - [x] Update `CLAUDE.md`, `.gitignore`, and active documentation comments that still prescribe cloning, merging upstream, `dsio run`, accelerator extras, or the built-in starter corpus.
  - [x] Do not rewrite frozen historical plans or accepted/superseded ADR history merely to erase old terminology; current guidance must clearly point to ADR 0019 and the accepted generic-spine spec.
- [x] Prove the distribution boundary and import contract (AC: 1, 3, 4)
  - [x] Add an automated build check for both wheel and sdist and inspect their metadata and member lists.
  - [x] Assert the wheel has the `dsio` package and distribution metadata, declares the required runtime stack, exposes no `dsio` console script, contains no tests/project code, and contains no `dsio/torch` package.
  - [x] Assert wheel metadata names full `mlflow` rather than `mlflow-skinny`, and the wheel contains no `dsio/cli`, `dsio/application.py`, `dsio/presets.py`, `dsio/config/presets.py`, or `dsio/config/overrides.py`.
  - [x] Assert the sdist omits `_bmad-output` and all four review reports.
  - [x] Install the wheel and all `Requires-Dist` dependencies into an isolated environment outside the checkout, using an explicit CPU PyTorch index only in CI/test tooling; do not use `--no-deps`. Run dependency validation and import with `python -I -B`; assert `dsio.__file__` resolves from the installed environment rather than the repository.
  - [x] In a subprocess with isolated working, home, Prefect, and MLflow paths, assert `import dsio` creates no DSio-authored or external-system state, starts no flow or Run, makes no service call, and leaves `prefect`, `mlflow`, `torch`, `lightning`, and `torchmetrics` absent from `sys.modules`. Interpreter-managed bytecode is excluded from this contract and disabled with `-B` in the probe.
- [x] Prove consumer-owned orchestration and keep CI honest (AC: 2)
  - [x] Add a consumer-style integration test that defines its own `@task` and `@flow`, calls `sha256_of`, invokes the flow directly, and asserts the returned digest.
  - [x] Isolate Prefect's test home/settings so Prefect's own runtime state cannot be mistaken for a DSio import side effect.
  - [x] Update CI and developer commands to install the locked required stack without DSio accelerator extras, build both distributions, run the isolated-install smoke test, and then run the full quality suite.
  - [x] Replace the obsolete accelerator-extra contract test with a package-metadata contract test for the required dependencies and absence of a console script.
- [x] Run the complete verification gate (AC: 1-4)
  - [x] `uv lock --check`
  - [x] `uv build`
  - [x] focused packaging, import-safety, and project-flow tests
  - [x] `uv run pytest -q`
  - [x] `uv run ruff check .`
  - [x] `uv run mypy`
  - [x] `uv run lint-imports`
  - [x] `git diff --check`

### Review Findings

- [x] [Review][Patch] Enforce the intended sdist allowlist, allowing only Hatch's mandatory `.gitignore` addition. [`tests/test_built_distribution.py`:72]
- [x] [Review][Patch] Remove active source and test commentary that still prescribes the deleted `dsio run` CLI. [`src/dsio/eval/pool.py`:3]
- [x] [Review][Patch] Monitor the isolated working, cache, configuration, data, and temporary paths in the inert-import probe. [`tests/test_built_distribution.py`:96]
- [x] [Review][Patch] Execute the documented consumer-owned Prefect flow from the isolated wheel installation. [`tests/test_built_distribution.py`:167]
- [x] [Review][Patch] Document accelerator-index selection before the command that resolves DSio's required Torch dependency. [`README.md`:13]
- [x] [Review][Patch] Resolve the isolated environment's interpreter portably instead of assuming POSIX `bin/python`. [`tests/test_built_distribution.py`:112]
- [x] [Review][Defer] Resolve `uv.lock` relative to `repo_root` when capturing run provenance. [`src/dsio/runs/record.py`:195] — deferred, pre-existing
- [x] [Review][Defer] Define replay behavior for consumer projects that do not use `uv.lock`. [`src/dsio/runs/record.py`:261] — deferred, pre-existing
- [x] [Review][Defer] Shell-quote recorded command arguments with a standard quoting primitive. [`src/dsio/runs/record.py`:266] — deferred, pre-existing

## Dev Notes

### Current State and Required Delta

- The existing Hatch configuration already builds a valid wheel from `src/dsio`; preserve it. The missing contract is consumer installation: Prefect is absent, the Lightning/PyTorch/MLflow stack is hidden behind DSio extras, and CI tests the checkout rather than an installed artifact.
- `import dsio` is currently safe because the root package defines only its version. Preserve this deep-module behavior: installing heavy dependencies does not justify importing them at the root.
- Mainline still exposes a console command that resolves presets, stages data, creates runs, and dispatches registered runners through `dsio.application`. That is the orchestration layer this story removes. Prefect functions live in the consumer project; DSio supplies called functions only.
- The mainline wheel contains legacy domain packages that later stories will migrate. Incremental migration is intentional. This story forbids adding another technology namespace and removes the supported CLI/application path; it does not perform the data or training redesign early.
- The default sdist currently includes internal review and planning material. Treat the sdist member list as a disclosure boundary, not just a build detail.

### Technical Requirements

- Keep Python `>=3.12` and the current Hatchling build backend.
- Use `prefect>=3.8,<4`; as of 2026-09-21 the locked/current stable release is 3.8.6 and supports Python `>=3.10,<3.15`.
- Publish `torch>=2.7,<3`, `lightning>=2.5,<3`, `torchmetrics>=1.7,<2`, and full `mlflow>=3,<4` as unconditional dependencies. Exact versions belong in `uv.lock`.
- Do not encode CPU versus CUDA as a DSio feature or extra. The DSio wheel declares `torch`; the consuming project selects a compatible wheel and index.
- Keep the README flow synchronous and local. Calling a decorated Prefect flow/task directly proves ownership without introducing task-runner or deployment behavior.
- Scope the zero-side-effect guarantee to `import dsio`. Running a Prefect flow may legitimately initialize Prefect-owned state, which tests must isolate.
- Use native distribution metadata (`Requires-Dist`, `Requires-Python`, files, entry points) as the package contract; do not add a DSio manifest model.

### Architecture Compliance

- Prefect is the only DAG definition. No `DsioFlow`, graph model, runner facade, CLI adapter, or orchestration protocol may be added.
- This story adds no MLflow coordination; Story 1.2 owns the explicit parent-Run context.
- Keep module boundaries aligned to pipeline responsibilities. Do not create `dsio.torch` or consumer-project-specific code.
- Preserve useful internal functionality not yet replaced, but do not retain dead CLI/application adapters for hypothetical compatibility. The accepted architecture is a deliberate breaking change.
- Prefer deletion and native library APIs over deprecation shims. There is no released compatibility promise for the superseded clone-and-run surface.

### File Structure Requirements

Expected updates:

- `pyproject.toml`, `uv.lock`
- `README.md`, `CLAUDE.md`, `.gitignore`
- `.github/workflows/ci.yml`
- `src/dsio/__init__.py`
- `src/dsio/config/__init__.py` and directly affected legacy bootstrap files only as required to remove the CLI-owned preset path
- `tests/test_accelerator_extras.py` (replace or rename to a distribution-metadata contract)
- `tests/test_import_contracts.py` if contract module lists change

Expected deletions:

- `src/dsio/cli/`
- `src/dsio/application.py`
- `src/dsio/presets.py`
- `src/dsio/config/presets.py`
- `src/dsio/config/overrides.py`
- `tests/cli/`
- `tests/test_application.py`
- `tests/config/test_preset_discovery.py`
- `tests/train/test_runner_bootstrap.py`
- `Dockerfile`

Expected additions:

- A focused distribution/import-safety test module
- A focused consumer-owned Prefect-flow integration test module

Keep each concern as one test module initially. Split a module into a same-named directory only if distinct subconcerns make the single file hard to navigate.

### Testing Requirements

- Test the built artifacts, not only editable checkout imports.
- The isolated import probe must run outside the repository with bytecode disabled. Assert an installed `site-packages` origin and no DSio-authored, Prefect, or MLflow state; interpreter-managed cache files are not DSio side effects.
- Keep import-safety and flow-execution assertions separate: DSio import must be inert; Prefect execution owns any orchestration state.
- Inspect wheel `METADATA` and entry points directly so a future packaging regression cannot pass because the development environment already has dependencies installed.
- Run the existing suite after deletion. Any test tied only to the removed CLI/application contract should be deleted or rewritten around its still-valid lower-level invariant; do not preserve obsolete behavior solely to keep an old test green.
- Keep Docker and live external services out of the default test gate.

### Git Intelligence

- Implement from `main` at `e31274f` or newer. The current branch is `1` commit ahead and `17` behind; its unique patch is equivalent to an already merged mainline patch.
- Mainline commits `b05579c` and `50590f3` established the current Hatch distribution and centralized CLI/application composition. Reuse the packaging foundation; remove the now-superseded orchestration boundary.
- Recent fixes consistently favor fail-closed behavior, atomic publication, and explicit stable identity. Preserve that style when replacing tests and metadata contracts.
- The current dirty tree contains planning material only. Treat it as user-owned and carry it forward without folding the four review reports into source distributions or implementation edits.

### Latest Technical Information

- PyPA maps `[project].dependencies` directly to unconditional `Requires-Dist`; optional dependencies become extras. The agreed first-class stack therefore belongs in the main dependency list.
- Prefect documents flows and tasks as decorated Python functions and direct flow invocation as normal usage. `.submit()` is unnecessary for this acceptance slice.
- Hatch documents `packages = ["src/dsio"]` as the intended src-layout wheel configuration; no build-system migration is needed.
- Python's `importlib.metadata` API can validate installed version, files, requirements, and entry points without a DSio metadata wrapper.

### Project Structure Notes

- The accepted target introduces real `tracking`, `metrics`, and `inference` domains in later stories. Empty packages now would be speculative structure and should not be added.
- `dsio.contracts.sha256_of` is an existing pure callable suitable for the first consumer-flow proof. Do not re-export it from the package root just to make the example shorter.
- Historical ADRs and plans remain historical evidence. Active README, contributor guidance, packaging metadata, and CI must reflect ADR 0019.

### References

- [Source: docs/superpowers/specs/2026-09-18-generic-experiment-spine.md#Goal]
- [Source: docs/superpowers/specs/2026-09-18-generic-experiment-spine.md#Native-systems]
- [Source: docs/superpowers/specs/2026-09-18-generic-experiment-spine.md#Package-shape]
- [Source: docs/adr/0019-versioned-library-with-project-owned-prefect-flows.md]
- [Source: _bmad-output/planning-artifacts/epics.md#Story-11-Install-DSio-and-Run-a-Project-Owned-Flow]
- [Source: pyproject.toml#project]
- [Source: README.md#Start-a-project]
- [PyPA: Writing `pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [PyPA: `pyproject.toml` specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
- [PyPA: Packaging flow](https://packaging.python.org/en/latest/flow/)
- [Hatch: Build configuration](https://hatch.pypa.io/latest/config/build/#packages)
- [Prefect: How to write and run a workflow](https://docs.prefect.io/latest/tutorial/flows)
- [Prefect 3.8.6 package metadata](https://pypi.org/project/prefect/)
- [Python 3.12: `importlib.metadata`](https://docs.python.org/3.12/library/importlib.metadata.html)

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-21: Created `architecture/generic-experiment-spine` from current `main` without stashing or altering user-owned files. Focused pre-change suite: 55 passed, 1 deselected.
- 2026-09-21: Replaced accelerator extras and the CLI entry point with one required, source-neutral runtime dependency contract; regenerated the lock and verified a clean locked sync plus focused metadata tests.
- 2026-09-21: Removed the CLI/application/preset orchestration layer and its behavior-specific tests while preserving lower-level config and legacy runner internals for later stories.
- 2026-09-21: Reframed active guidance around a pinned library and project-owned Prefect flow; added artifact, isolated-install, inert-import, and consumer-flow contract tests. Focused gate: 40 passed.
- 2026-09-21: BMad adversarial review completed across blind, edge-case, and acceptance layers. Six acceptance gaps were resolved; three pre-existing provenance issues were recorded for later work.
- 2026-09-21: Independent Codex and OpenCode review hardened exact wheel metadata, isolated package resolution and Prefect settings from the host, extended inert-import mutation checks to the installed environment, and clarified active README claims.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Published one required, accelerator-neutral runtime contract and a narrowly scoped sdist.
- Replaced clone-and-run orchestration with an inert installable library called from project-owned Prefect flows.
- Removed the obsolete CLI, application, preset, override, and CLI-container surfaces without prematurely reorganizing later pipeline domains.
- Added built-artifact, isolated-install, inert-import, metadata, and consumer-flow regression coverage; updated CI to exercise the distribution boundary.
- Final gate: 8 focused tests and 569 repository tests passed; Ruff, mypy, four import contracts, lock validation, build, and diff checks all passed.
- Post-review gate passed unchanged at 569 tests; the installed wheel now executes the representative Prefect flow and the import probe monitors all isolated state roots.
- Final independent-review gate: 8 focused tests and 569 repository tests passed; Ruff, mypy, all four import contracts, lock validation, build, and diff checks passed.

### File List

- `.github/workflows/ci.yml`
- `.gitignore`
- `CLAUDE.md`
- `README.md`
- `pyproject.toml`
- `uv.lock`
- `src/dsio/__init__.py`
- `src/dsio/config/__init__.py`
- `src/dsio/config/registry.py`
- `src/dsio/eval/pool.py`
- `src/dsio/runs/provenance.py`
- `src/dsio/runs/record.py`
- `src/dsio/train/__init__.py`
- `src/dsio/train/runner.py`
- `src/dsio/train/torch_task.py`
- `src/dsio/train/tracking.py`
- `tests/config/test_config.py`
- `tests/conftest.py`
- `tests/model/test_registry_bootstrap.py`
- `tests/runs/test_runs.py`
- `tests/train/test_artifacts.py`
- `tests/train/test_execute_seeding.py`
- `tests/train/test_runner_registry.py`
- `tests/train/test_ssl_runner.py`
- `tests/train/test_token_run.py`
- `tests/train/test_torch_runner.py`
- `tests/train/test_tracking.py`
- `tests/test_built_distribution.py`
- `tests/test_distribution.py`
- `tests/test_project_flow.py`
- Deleted: `Dockerfile`, `src/dsio/application.py`, `src/dsio/presets.py`, `src/dsio/cli/`, `src/dsio/config/overrides.py`, `src/dsio/config/presets.py`
- Deleted: `tests/cli/`, `tests/test_accelerator_extras.py`, `tests/test_application.py`, `tests/config/test_preset_discovery.py`, `tests/train/test_runner_bootstrap.py`

### Change Log

- 2026-09-21: Implemented Story 1.1 and moved it to review after the complete quality gate passed.
- 2026-09-21: Resolved all review patches and moved Story 1.1 to done.
- 2026-09-21: Resolved independent review findings and retained Story 1.1 as done.
