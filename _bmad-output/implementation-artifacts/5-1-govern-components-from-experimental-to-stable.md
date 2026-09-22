---
baseline_commit: 26d4125
---

# Story 5.1: Govern Components from Experimental to Stable

Status: in-progress

## Story

As a DSio maintainer,
I want reusable components admitted through one strict experimental-to-stable path,
so that projects can share battle-tested code without turning DSio into an unrestricted plugin system.

## Acceptance Criteria

1. New reusable components first live under the documented `dsio.experimental` namespace and are selected by named import paths or an existing governed closed dispatcher.
2. Stable promotion requires focused unit and integration tests, determinism/provenance behavior, generic contracts, reuse evidence, agentic review, and explicit human approval; failed criteria keep the component experimental.
3. Admission checks reject consumer-project branching, private project dependencies, private DSIO dependencies, and runtime registry mutation with an identified rule.
4. Incompatible stable changes follow semantic versioning and migration/deprecation policy rather than silently changing project behavior.
5. Novel project components are contributed and tested through this path; project-side runtime registration is not introduced.

## Tasks / Subtasks

- [ ] Establish one experimental namespace and admission guide (AC: 1-5)
  - [ ] Document entry, evidence, review, promotion, rejection, and semantic-versioning rules.
  - [ ] Keep selection to import paths or existing closed dispatchers; expose no component registry or plugin API.
- [ ] Add one static admission audit for generic source boundaries (AC: 2-3, 5)
  - [ ] Require the `dsio.experimental` namespace and reject private/project dependency imports.
  - [ ] Reject explicit consumer-project branching and DSIO runtime registry mutation.
  - [ ] Return plain rule messages and raise one actionable boundary error; add no admission result model.
- [ ] Prove valid generic source, each rejection rule, named importability, policy completeness, and installed-distribution inclusion through tests and release gates (AC: 1-5)

## Dev Notes

### Minimal shape

- Add `dsio.experimental` and one small `admission` module. Do not add a plugin manager, component manifest/dataclass, runtime registry, promotion service, CLI, or approval database.
- Static checks cover mechanically provable genericity/dependency rules. The documented PR checklist owns evidence that requires judgment: real reuse, determinism, provenance, tests, review, migration notes, and explicit human approval.
- The audit returns a tuple of rule messages; `require_admissible_source` raises one error when enforcement is needed.

### Compatibility policy

- Experimental APIs may change in a minor release through this same reviewed path.
- Stable additive behavior is minor, compatible fixes are patch, and incompatible stable behavior requires a major release or a documented minor-release deprecation followed by a major removal.
- Promotion moves the component into its responsibility-named stable package and adds public compatibility tests; no project is silently redirected.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 5.1, FR35, FR36]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Component admission and compatibility]
- [Source: `src/dsio/config/components.py`]
- [Source: `src/dsio/config/registry.py`]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 4.4 at `26d4125`; selected one static audit plus a review checklist over any runtime governance system.

### Completion Notes List

### File List

- `_bmad-output/implementation-artifacts/5-1-govern-components-from-experimental-to-stable.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

### Change Log

- 2026-09-22: Created Story 5.1 and started implementation.
