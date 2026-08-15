# 4. What the prior art changes

Status: accepted (2026-08-15)

A survey of ML framework templates, lab training codebases, SSL frameworks, and
content-addressed pipeline tools. Recorded because several findings validate decisions we
made from first principles, and four change the plan.

## Validated

**Config as Python factory functions.** `pytorch/torchtitan` — Meta's current reference LLM
training codebase — **migrated away from TOML files** to
`torchtitan/models/<family>/config_registry.py`: named zero-argument functions
(`llama3_8b()`, `llama3_debugmodel_float8()`) resolved by `--module llama3 --config
llama3_8b`, with `tyro` layering CLI overrides. That is our preset registry, arrived at
independently by the team that shipped Hydra. They also nest each config class inside the
component it configures, as we do with `TabularTask` in `train/tabular.py`.

**A pure vocabulary leaf.** `composer/core/` (algorithm, callback, engine, event, state,
time) is `dsio/contracts/`.

**Content-addressed run identity.** `marin-community/marin`'s `execution/` module is the
closest prior art to our ledger: canonical JSON of config → hash → output path suffix, a
`.executor_status` file holding `RUNNING|SUCCESS|FAILED|DEP_FAILED`, and a lock file for
distributed writes. Content-addressed output paths give dedup, resume, and provenance from
one mechanism — which is exactly why the ledger earns its place. (Calibration in the other
direction: Karpathy *deleted* his ledger module from nanochat, −540 LOC, "it just bloats
the code." A ledger that only records is not worth it; one that also drives resume is.)

**Online probe as a callback.** `galilai-group/stable-pretraining` attaches evaluation as
callbacks — `OnlineProbe`, `OnlineKNN`, `RankMe`, `LiDAR` — that tap training state without
touching the loop. `solo-learn` puts the probe directly in its base class with
`feats.detach()` and a separate `classifier_lr` param group. Both are in-process, which
confirms rejecting FORGE's subprocess-shelling `DownstreamTaskCallback`.

## Changed

**1. Cache invalidation gets early cutoff, and code hashing becomes the default.**

The plan specified explicit `version=` bumps with a source-drift warning. Surveying how
others do it changes the balance:

| Tool | Hashes code? | Escape hatch |
|---|---|---|
| Hamilton | yes — source hash ignoring docstrings/comments | `@cache(behavior=RECOMPUTE\|DISABLE)` |
| joblib | yes — warns "Function X has changed", then clears | `ignore=[...]` |
| Snakemake | yes — plain-text equality on the rule body | `--rerun-triggers` |
| Flyte | **no** — human owns `cache_version` | (the version *is* the hatch) |
| DVC | only if you list the `.py` in `deps` | `--force`, `always_changed` |

Two documented failure modes bound the design. Hamilton states outright that it "will not
version nested function calls" — edit a helper and you get a false hit. And Flyte's own
docstring concedes the opposite failure: "You should also manually bump this version if
the function body has changed, but the signature hasn't."

Revised design: **hash the code automatically** so nobody has to remember, keep an explicit
`version=` as an override for what hashing cannot see (an upgraded dependency, changed
remote data), and add **early cutoff** — after recomputing a stage, hash its *output*; if
the output is unchanged, do not invalidate downstream stages. Early cutoff is what Nix's
`ca-derivations` and Bazel buy, and it is what makes aggressive code hashing safe: a
reformat recomputes one stage and stops, instead of cascading through the graph.

**2. The canonical store follows Megatron's two-file layout.**

`megatron/core/datasets/indexed_dataset.py`: a prefix maps to `.bin` (raw concatenated
payload) and `.idx` (magic header `MMIDIDX\x00\x00`, explicit version int, then
`sequence_lengths`, `sequence_pointers`, `document_indices`). Three structural rules to
follow: version the header from day one; keep the builder entirely out of the read path
(`_IndexWriter`/`_IndexReader` are private, `IndexedDatasetBuilder` never appears in
reads); and abstract the **storage backend**, not the index — their `_BinReader` is an ABC
with mmap, plain-file, and S3 implementations behind identical `.idx` semantics. That last
one is how `StageIO` avoids leaking the Zarr-vs-Lance choice.

Megatron also uses **three index layers**, not one: storage pointers → sample windowing →
ordering. Only the first is canonical; the other two are derived, cached as `.npy`, and
keyed by a hash of the config that produced them. Levanter's `JaggedArrayStore` is the same
shape (`offsets` + `data` + `shapes`, append-only, resumable).

**3. Shuffling is a function, not a materialized array.**

Levanter's `data/_prp.py` implements `LcgPermutation` and `FeistelPermutation` — a
stateless O(1) pseudo-random permutation where order is a pure function of `(seed, index)`.
Resuming needs no dataloader state and no seeking, because position *is* the state.
Megatron materializes a `shuffle_index` instead. For a system whose headline promise is
reproduction, the stateless version is strictly better and costs about fifty lines.

**4. An SSL pretext task is a function, not a class.**

`stable-pretraining` defines each method as a plain `forward(self, batch, stage) -> dict` —
`simclr`, `byol`, `vicreg`, `dino` are functions, composed as
`Module(forward=spt.forward.simclr, backbone=..., projector=...)`. Returning a dict makes
every intermediate tensor loggable without a schema per method.

The counter-example is instructive. VISSL — now **archived**, and pinned to `numpy==1.19.5`
so it will not install — spread each method across at least four places: a block in a
1,737-line global `defaults.yaml`, a loss, a hook, and a head. That is FORGE's
change-amplification problem (seven places per new SSL method) in another codebase. A
function plus a loss is the smaller surface.

**5. The source-drift warning fails unsafe. Reversed.**

The plan said: warn loudly when a stage's source changed without a `version=` bump, but
serve the cached result anyway. That is Hamilton's silent-staleness bug with a log line
attached. A warning in a scrollback buffer is not a safety mechanism.

Compare how the two automatic-hashing systems fail. HuggingFace `datasets` fails **safe** —
when `dill` cannot hash a transform it substitutes a random fingerprint and recomputes
everything. Hamilton fails **unsafe** — an edited helper yields a false cache hit and a
stale artifact.

Revised: drift is an **error by default**, overridable with `--accept-drift`, and the
recommended resolution is to recompute. Combined with early cutoff (change 1), recomputing
is cheap, because an unchanged output stops the invalidation from propagating.

**6. Windowed views are a leakage generator unless the group key is mandatory.**

This was missed entirely. Overlapping windows that straddle a split boundary put
near-identical rows in both train and test — simultaneously Kapoor & Narayanan's L1.4
(duplicates) and L3.2 (non-independence) — and it is *invisible*, because the index is only
offsets.

Three requirements, all enforced in the constructor rather than documented:

- The windowed-view constructor takes a **required** `group_by` argument. Optional means
  omitted.
- Splitting happens on groups (subject, device, session, symbol), **never** on window
  index.
- Temporal splits carry a mandatory `gap`, purging windows whose lookback crosses the
  boundary. `sklearn.model_selection.TimeSeriesSplit(gap=...)` exists for exactly this.

Related: prefer *Repeatable Splitting* (ML Design Patterns #22) — assign by a hash of a
stable ID, e.g. `sha256(subject_id) % 100 < 80` — over a materialized split file. Splits
then survive corpus growth and need no artifact to version, which suits a
content-addressed store.

**7. The fixed component chain is a contrastive-family helper, not the spine.**

`preprocessors → augmentor → transform → backbone → head` is precisely the shape VISSL and
solo-learn baked in, and it breaks on the first JEPA-family method. Masking is not
augmentation — it is a sampler producing context and target index sets, which is why
`vjepa2/src/masks/` is a top-level concern. JEPA has two encoders plus a predictor, not a
backbone plus a head. Multi-crop (DINO, SwAV) emits a variable-length list of views at
different resolutions that a single `transform` slot cannot type. And TF-C — a headline
*time-series* method, so this bites in our primary modality — needs two parallel encoders
over time-domain and frequency-domain views on day one.

What survives all of these is the **batch contract, not the chain**: a stage-tagged
`dict → dict`. Adopt `stable-pretraining`'s shape — the method owns its topology, dsio
fixes only the I/O. FORGE's chain remains available as a composition helper for the
contrastive family.

**8. The online probe needs a label-free counterpart.**

The plan's online linear probe requires labels *during pretraining* — which is exactly what
we do not have on the large unlabeled corpus. `stable-pretraining` ships `RankMe` and
`LiDAR` callbacks precisely for this, and they belong alongside the probe.

Two related corrections: V-JEPA 2 evaluates with **attentive** probes, and linear probing
systematically under-reads JEPA-family representations — so offer both. And third-cohort
evaluation should be **n-seed with mean±std**, not a single number; NeurIPS Q7 and REFORMS
§7 both require uncertainty.

**9. Masking is a first-class component, and we did not have one.**

I-JEPA and V-JEPA both ship `src/masks/` as a top-level concern, and MOMENT's entire
pretraining objective is masked reconstruction. For time series and tabular — our primary
modalities — **masking is the dominant pretext family**, not a niche one. It fits neither
slot we had: it is not an augmentor (it changes the loss *target*, not just the input) and
it is not a transform. `dsio.ssl.masks` becomes its own module.

**10. The third cohort gets sealed, and unseals get counted.**

With SSL first-class, many pretraining runs will be ranked before anything touches labels,
so evaluation overfitting on the held-out cohort is a live risk rather than a theoretical
one. Put the third cohort behind a token that must be explicitly unsealed, and log every
unseal into the run record.

Together with the mandatory `group_by` above, most of the leakage taxonomy collapses into
two API invariants worth stating plainly:

> You cannot obtain a fitted transform except through a split-bound API, and you cannot
> obtain test data except through a sealed accessor.

Both are enforceable by import-linter contracts, which is where they belong. Add one
mechanical assertion at view-construction time: **no raw offset may appear in two splits.**
A framework that cannot produce a leaky split is worth more than any amount of caching.

**11. Determinism is neither free nor total.**

Seeding the process is not enough. Implemented in `runs/seeding.py`:
`torch.use_deterministic_algorithms(True, warn_only=True)` (cuDNN flags alone do not cover
nondeterministic ATen kernels), `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic cuBLAS
reductions on CUDA ≥ 10.2, and `dataloader_kwargs()` supplying a seeded `generator` plus a
`worker_init_fn` — without which shuffle order depends on worker count, making results a
function of a performance knob.

Be honest in the docs about the guarantee: PyTorch does not promise identical results
across releases, commits, platforms, or CPU-vs-GPU. dsio delivers **reproducible**, not
**replicable**.

**12. Decide now what does *not* go in the cache key.**

Every mature system has this decision and it is always contested — Flyte exposes it
explicitly as `ignored_inputs` and `salt`. Our key is (stage, version, config subtree,
upstream key), so `num_workers`, device, batch size, and log level must be classified
before the first stage is written. Wrong in one direction and the cache thrashes on every
machine change; wrong in the other and artifacts are silently reused across incompatible
hardware.

Provisional rule: anything that changes the *output bytes* is in the key; anything that
changes only *how fast you got there* is out. Batch size is the hard case — it is a
throughput knob for inference and a semantic one for training, so it lives in the key for
training stages and outside it for pure transforms.

**13. Verify the artifact on load, not just the key on lookup.**

Kedro's `--only-missing-outputs` checks existence only, which means a truncated or
half-written artifact stays "valid" forever. The model registry already verifies a digest
on load and fails closed; the stage cache must do the same. Content-address the outputs, not
just the inputs — which is also what makes early cutoff (change 1) possible, since early
cutoff needs an output hash anyway.

**14. A `dsio doctor` lint scored by minimum, not by count.**

Breck et al.'s ML Test Score aggregates by taking the **minimum** across its four sections,
so one neglected axis caps the whole score. That is a better forcing function than a list of
warnings, because it cannot be gamed by fixing the easy axis. Their Infra-1 is literally
"Training is reproducible," which is this repo's thesis.

## Noted

**Time-series SSL has no framework.** `thuml/Time-Series-Library` (12.7k stars) returns
zero hits for "self-supervised"; `tsai` has exactly one SSL callback; `aeon`'s
`self_supervised` namespace contains exactly two estimators. TS2Vec and TF-C are per-paper
repos. MOMENT dispatches pluggable *heads* over one fixed pretext task. The gap is real and
unfilled, which is worth knowing given `pdm` and `algua` are both panel-shaped and
label-scarce.

**Every SSL framework except one is dead.** VISSL archived; mmselfsup frozen 2023-06;
mmpretrain effectively unmaintained since 2024; solo-learn no feature work since 2024-01.
`lightly` ships monthly but deliberately has no method abstraction, and its investment is
moving to AGPL-licensed `lightly-train`. Depend on none of them.

**LeJEPA is worth a close look for label-scarce work.** Repo is
`galilai-group/lejepa` (the `rbalestr-lab` org was renamed). SIGReg is ~50 lines with a
single hyperparameter, no stop-gradients, no teacher-student, no schedulers — and the paper
reports 94%+ Spearman correlation between training loss and downstream performance, which
would enable *label-free model selection*. That is directly relevant when you have 100:1
unlabeled-to-labeled data.

## Deferred

Two structural suggestions worth deciding before Phase 2, not adopted here:

- **Copier instead of "fork it".** `mila-iqia/ResearchTemplate` uses Copier so downstream
  projects can run `copier update` to pull template improvements. That is a purpose-built
  answer to fork drift, versus our `git fetch upstream && git merge`.
- **`uv` workspace with two distributions.** Making `dsio` and `project` separate packages
  would enforce the dependency direction (`project → dsio`, never back) through packaging
  rather than an import-linter contract. Note that `src/<framework>/` + `src/<project>/` in
  one repo is otherwise not a pattern anyone in the survey uses; MosaicML's equivalent is
  two repos (`composer` and `llm-foundry`).
