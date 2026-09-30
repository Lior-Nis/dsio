# Story 6.4: Generate a CI-checked component catalog

Status: review

## Story

As a consumer looking for a lego block,
I want a catalog generated from source and the evidence register that lists every component with its contract, maturity and evidence,
so that I can compose from what exists instead of re-implementing it.

## Acceptance Criteria

1. Given every public name in the spine modules and the experimental domain packages, plus closed-dispatcher entries (split algorithms, `METRICS`), when the generator (`tools/`, not shipped in the wheel) runs, then it writes `docs/component-warehouse/catalog.md`, grouped by pipeline package. Each entry has its import path, its maturity (from location), the Consumes, Produces, Parameters, Devices, Limitations and Example docstring sections, and evidence joined from `docs/component-warehouse/evidence.yaml`.
2. Given the evidence register, when CI validates it, then each entry names consumer and test paths that exist, whether the use is real or a fixture, and its unrelated-use group, plus optional MLflow run URIs (format-checked, not dereferenced). DSio source contains no consumer names.
3. Given a public component missing a docstring section or evidence entry, or a committed catalog that differs from the regenerated one, when CI runs, then it fails and names the component or the drifted entry.
4. Given each Example section, when the test suite runs, then every example executes as a doctest.
5. Given `docs/component-warehouse/candidates.yaml`, when the catalog is generated, then candidates render with consumer path, reason and use count, and the stale docstrings found while writing sections are corrected.
6. *(Moved from Story 6.5.)* Given the catalog and evidence register, when CI runs, then it fails if any importable component in a stable package lists fewer than two unrelated real uses in the evidence register. Closed-dispatcher entries and spine functions are exempt.

## Tasks / Subtasks

- [x] Task 1: `tools/catalog.py` (AC 1, 3, 6).
  - [x] 1.1 `collect()`:
    - **Spine:** the `__all__` of the spine modules.
    - **Components:** every `dsio.experimental` module's `__all__`, excluding the admission auditor.
    - **Dispatchers:** split algorithms and `METRICS`.
    - **Exclusions:** exceptions and type aliases.
  - [x] 1.2 `validate()` rules:
    - every component has an evidence entry, and evidence entries name real components;
    - legacy components live in `dsio.experimental`;
    - non-legacy components carry all six sections and have at least one real use when experimental;
    - stable components need at least two real uses in distinct groups plus a recorded `approval`;
    - fixtures never count;
    - consumer and test paths must exist;
    - run URIs must be immutable (`runs:/<id>/...` or an MLflow UI run URL);
    - candidates need existing paths and a reason.
  - [x] 1.3 `render()` sections: proven components (full sections plus evidence), legacy experimental, spine by package, closed dispatchers, admission candidates.
  - [x] 1.4 `--check` exits non-zero on any rule violation or on a stale `catalog.md`.
- [x] Task 2: `docs/component-warehouse/evidence.yaml` (AC 2). It covers 47 components:
  - 5 proven: calibration and telemetry, with real uses in Parkinson FoG and CMI sequence, groups `parkinsons-fog` and `child-mind`, and MLflow run URIs from experiments 57 and 59. `calibrate_execution` is recorded `via` `calibrate_training_execution`.
  - 42 legacy: `NTXent`, `Jitter`, `TwoView`, `TensorOutput` and `validate_tensor_prediction` record their synthetic fixture uses.
- [x] Task 3: Six catalog sections on the five proven components. Each Example executes as a doctest (AC 4). `calibrate_training_execution` needs real project factories, so its call is `+SKIP`, but the test requires every Example to execute at least one statement. The doctests caught two wrong Consumes descriptions: the calibration benchmark measurement keys, and `log_calibration`'s training-calibration measurement keys. Both are corrected.
- [x] Task 4: `docs/component-warehouse/candidates.yaml` (AC 5): 15 candidates. Nine are consumer-local patterns mapped to the story that absorbs them. Six stay local on purpose: the tokenizer, `well_arrays`, the encoder handoff, streaming evaluation, the labelled-example filter and the window dataset's pending story.
- [x] Task 5: `tests/test_component_catalog.py`: rules hold, the catalog is current, all three kinds are present, Examples execute, and each validator rule has a negative test.
- [x] Task 6: Public surface. `dsio.train.trainer`, `dsio.train.artifacts` and `dsio.experimental.telemetry` declare `__all__`. The README gains "Find a component before writing one", linking the catalog, conventions and admission process.

- [x] Task 7: Review follow-ups. The review's critical finding: the stable rule never saw real code, because `collect()` only walked the spine list and experimental. Fixes:
  - [x] 7.1 `collect()` walks **every** `dsio` module. Stable modules are spine only if listed in `SPINE_MODULES`, and everything else is a component, so a promoted block cannot skip the stable rules. A test drops a spine module and asserts its names become checked components.
  - [x] 7.2 Legacy is **frozen**. Only the `LEGACY` set in `tools/catalog.py` may be legacy, and a legacy component with a real use fails until it drops legacy and gains sections.
  - [x] 7.3 Real uses must be Kaggle consumers (`reference_projects/kaggle/...`). The unrelated-use group is **derived from the path**, so both CMI consumers count once. The consumer must reference the component, or its `via`. Only uses that pass every check count toward groups.
  - [x] 7.4 Every experimental module declares `__all__`, and any public class or function missing from it is reported. The missing `autocorrelation`, `embed` and `Encodable` are now catalogued, as legacy.
  - [x] 7.5 Examples must execute a statement that calls the component itself. `calibrate_training_execution` now has a runnable Example (a real `DsioModule` and a small data module), which is also its first direct test.
  - [x] 7.6 Docstrings corrected against the code:
    - `calibrate_execution`'s measurement keys (`peak_process_tree_rss_bytes` only when a host budget is set; there is no `peak_host_memory_bytes`);
    - `calibrate_training_execution` takes `data_module_factory(candidate)`, memory budgets are fractions (0.8/0.5), it accepts `accelerator="auto"`, and requires a bare optimizer, a sized dataset and `sample_id`.
  - [x] 7.7 Rendering:
    - components are grouped by package;
    - candidates show a use count;
    - summaries use the first paragraph;
    - reST backticks are converted for Markdown;
    - `--check` prints a diff of a stale catalog;
    - repeated section headers are reported;
    - `tools/` is type-checked by mypy.

## Dev Notes

- **Section scope, a deliberate decision.** Full sections are required only for non-legacy components. The 42 legacy components are catalogued with summary and status, and gain sections when a warehouse story reshapes them for their first real use. Writing sections now for components Epics 7–9 will replace would be waste, which is why Story 6.5 went first. Spine entries are catalogued for discovery; they are not warehouse components (`CONTEXT.md`).
- **Evidence outside `src/dsio`.** The admission audit scans every source string for consumer names, so evidence in docstrings would fail it.
- **Tooling location.** `tools/` sits at the repository root, outside `src/dsio`, so hatch never packages it. The test imports it as `tools.catalog`; the repo root is on `sys.path` via `tests/conftest.py`.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

### File List

- `tools/__init__.py`, `tools/catalog.py` (new)
- `docs/component-warehouse/{catalog.md,evidence.yaml,candidates.yaml}` (new)
- `tests/test_component_catalog.py` (new)
- `src/dsio/experimental/{execution,training,telemetry}.py`, `src/dsio/experimental/model/components.py`
- `src/dsio/train/{trainer,artifacts}.py`
- `README.md`, `_bmad-output/implementation-artifacts/sprint-status.yaml`
