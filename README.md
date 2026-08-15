# dsio

A reproducible ML/DL experimentation spine, distributed as a Copier template.

`dsio` owns the parts every project rebuilds badly: typed configuration, staged data with
content-addressed caching, leakage-safe splits, a run ledger that makes results
reconstructible, evaluation with honest verdicts, and a resumable job matrix. It does not
own your models — those stay idiomatic. A tabular task uses a real scikit-learn
`Pipeline`, a deep task a real `LightningModule`, forecasting real Nixtla objects. There is
no universal `Model` wrapper to fight.

## Start a project

```bash
uvx copier copy gh:<you>/dsio my-project
cd my-project
uv sync
uv run dsio presets
uv run dsio run <module>_baseline --summary
```

Later, pull spine improvements without losing your work:

```bash
copier update && uv sync && uv run pytest
```

That last command is why this is a template rather than a repo you fork. A fork can only
merge; a Copier project records its answers and re-applies template changes on top of your
local edits — which is the difference between a fix reaching every project and rotting in
whichever repo it was written.

## Shape

A generated project is a `uv` workspace with two distributions:

```
pyproject.toml         your project — depends on dsio
src/<module>/          your code: ruff only, tests optional, hack freely
lib/dsio/              the shared spine: ruff + mypy + import contracts + tests
tests/                 your tests
runs/                  the run ledger (gitignored; the records are the source of truth)
stores/ views/         canonical data and derived indices (manifests committed)
```

`project → dsio`, never the reverse. That direction is enforced by packaging: the spine has
no way to name your project. Treat `lib/dsio/` as read-only and send fixes upstream.

## Principles

**Structure lives in Python, not YAML.** Configs are typed Pydantic objects composed by
preset functions; YAML is a recorded *output* of every run, never an authored input.
Variants are function arguments, so there is no path by which `lr=1e-6` becomes a file you
check in. (`torchtitan` migrated from TOML to this same shape — see `docs/adr/0001`.)

**Store once, index many.** A corpus is stored once as continuous signal; windowing
produces an index of offsets, not a copy. Materialize only what is expensive *and*
deterministic; index everything cheap and combinatorial.

**Never block, always reconstructible.** A dirty working tree does not stop a run — the
diff is captured as an artifact, so even a dirty run reproduces exactly. The clean-tree
gate sits at model-registry promotion, where it belongs.

**Correctness is structural.** Leakage walls are import-linter contracts, not review
conventions.

## Developing the spine

`lib/dsio/` is a standalone installable package with its own test suite:

```bash
cd lib/dsio
uv sync && uv run pytest && uv run ruff check . && uv run mypy && uv run lint-imports
```

This repository is a template, so it has no root `pyproject.toml` — `pyproject.toml.jinja`
owns that name. Decisions and their reasons live in `docs/adr/`.
