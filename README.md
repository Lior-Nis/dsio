# dsio

A reproducible ML/DL experimentation spine.

`dsio` owns the parts every project rebuilds badly: typed configuration, staged data with
content-addressed caching, leakage-safe splits, a run ledger that makes results
reconstructible, and evaluation with honest verdicts. It does not own your models — those
stay idiomatic. A tabular task uses a real scikit-learn `Pipeline`, a deep task a real
`LightningModule`, forecasting real Nixtla objects. There is no universal `Model` wrapper to
fight.

## Start a project

```bash
git clone https://github.com/<you>/dsio my-project
cd my-project
git remote rename origin upstream
uv sync --extra gpu           # or --extra cpu on a machine without an NVIDIA GPU
uv run dsio run                # lists the presets
uv run dsio run spine_baseline
```

Selecting an extra is mandatory: a bare `uv sync` installs no torch at all, and that is
deliberate — it fails loudly instead of silently pulling several gigabytes you may not
want. CI uses `cpu`. `gpu` currently floats on the default PyPI index because that
platform's default wheel is already CUDA-enabled (`torch 2.13.0+cu130`, `cuda.is_available()
== True` on an RTX 5070 Ti, next to `2.13.0+cpu` from `cpu`); if that ever stops being true,
`gpu` needs an explicit CUDA index again.

Add your components under `src/dsio/` — a backbone in `model/`, a preset in `presets.py`.
Later, pull spine improvements without losing your work:

```bash
git fetch upstream && git merge upstream/main
```

That is the whole propagation story. It works because the package name is fixed, so paths
line up between your clone and upstream and a merge conflicts only in files you both
edited. The previous Copier template existed to rename the package per project, which is
precisely what stopped a plain merge from working.

## Shape

A single `uv` project rooted at one package:

```
pyproject.toml    the project
src/dsio/         the package: config, data, splits, train, eval, runs, cli, presets
tests/            its test suite
runs/             the run ledger (gitignored; the records are the source of truth)
stores/ views/    canonical data and derived indices (manifests committed)
```

`docs/adr/0018-a-repository-not-a-template.md` records why this replaced the earlier
two-distribution Copier layout, and the one guarantee that move gives up: `project → dsio`,
never the reverse, used to be enforced by packaging and is now an import-linter contract
instead.

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

## Developing

```bash
uv sync --extra cpu
uv run --extra cpu pytest -q && uv run --extra cpu ruff check . && uv run --extra cpu mypy && uv run --extra cpu lint-imports
```

Decisions and their reasons live in `docs/adr/`.
