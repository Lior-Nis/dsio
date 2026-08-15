# 1. Configuration structure lives in Python, not YAML

Status: accepted (2026-08-15)

## Context

FORGE (`acc_base`) ran Hydra with Pydantic validation. Two things went wrong, and both
are worth naming precisely because neither is really a Hydra defect.

Every leaf schema was declared `extra="allow"` with a bare `_target_: str`, so Pydantic
validated essentially nothing at the level where it mattered. A misspelled field was
silently discarded and training proceeded with a default. Real errors surfaced inside
`hydra.utils.instantiate`, after data loading had already started.

Second, the repo reached 340 YAML files despite a written anti-proliferation policy in
both CLAUDE.md and README.md — including `adamw_differential_lr_1e5.yaml`, `_1e6.yaml`,
and `_1e7.yaml`. The policy lost because the system made a new file the easy way to
express a new value.

Separately, the wider ecosystem has moved: DeepMind's config files are Python
(`ml_collections`, then Fiddle), Meta's newer work uses dataclasses, and Hydra itself now
has a bus factor of roughly one. The `hydra-optuna-sweeper` plugin's stable release is
from 2022 and pins `optuna<3.0`.

## Decision

Configuration structure lives in Python. Presets are functions returning a validated
`RunConfig`. Component selection resolves through a decorator registry rather than a
reflective import of a `_target_` string. YAML and CLI tokens may override *leaf values*
only; they can never introduce structure or name a class.

Every schema is `frozen=True, extra="forbid"`, leaves included.

Every run writes `config.resolved.yaml` plus its sha256 into the run record. YAML remains
the recording format — it stops being the authoring format.

## Consequences

Renaming a config field is caught by mypy at every use site. An unknown component name
fails at parse time with a did-you-mean suggestion, before any data is read. A new
variant is a function argument, so there is no path by which `lr=1e-6` becomes a checked-in
file.

We give up Hydra's `defaults:` composition, multirun, and per-run output directories. The
run ledger replaces output directories. The job matrix replaces multirun. Composition
becomes ordinary Python.

We also take on a dotted-override parser and its scalar-parsing rules. That is deliberate:
delegating to PyYAML would read `3e-4` as the *string* `"3e-4"` under YAML 1.1, and a
learning rate silently becoming a string is exactly the failure class this system exists
to prevent.
