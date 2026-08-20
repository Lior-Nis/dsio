# 6. Splits are committed YAML lists of group IDs

Status: accepted (2026-08-17)
Amends: ADR 0004, item 6, which recommended *Repeatable Splitting* by hash of a stable ID.

Amended (2026-08-20): Generation moved out of the package. `SplitSpec`, `StratifyKey`,
`BalanceReport` and the `stratified_kfold` scheme described below no longer exist in
`dsio` — a project now writes its own offline generation script (sklearn's splitters cover
the non-temporal schemes) and commits the YAML it produces. The committed group-list
format this ADR argues for is unchanged, `dsio` still reads and validates it exactly as
described, and the purged/embargoed walk-forward maths in "Temporal splits are the other
half" is unaffected — that scheme has no sklearn equivalent and stays in the package. The
example split file's `# scheme: stratified_kfold, ...` header line is accordingly no
longer something `SplitFile.to_yaml()` can emit; a project's own generator is free to
record its own provenance in `notes` instead.

## Context

ADR 0004 proposed deriving splits from `hash(subject_id) % 100 < 80`, on the grounds that
splits would then survive corpus growth with no artifact to version, and that FORGE's 145
committed split YAMLs were config sprawl.

That was wrong on the substance, and the objection conflated two different things.

**The sprawl was combinatorial generation, not materialisation.** FORGE had 145 files
because it enumerated family × context × fold. The *files* are tiny — they name 128
patient IDs, not 230,555 window offsets.

**Hashing cannot stratify.** Assigning subjects to folds by hash cannot balance a
rare-event rate. FORGE stratified serpentine by FOG count precisely because random
assignment across 128 patients with wildly varying rates produced badly imbalanced folds.
Reproducibility is not the only property a split needs.

**Hashing cannot express leave-one-group-out.** "Leave patient *i* out" is a list. FORGE's
FogAtHome LOPO ran 12 folds over 12 patients; there is no hash function for that.

**A split is scientific provenance.** A paper must state which subjects were in test. A
YAML file in git is that statement — diffable, reviewable, citable. A hash requires also
pinning the exact ID set and the hash implementation, and then trusting both.

And the design this now sits on makes materialisation cheap rather than costly: one
memory-mapped store plus a small list of group IDs resolves to any fold on the fly, so a
split costs a boolean mask instead of a dataset.

## Decision

A split is a committed YAML file naming **groups**, never windows:

```yaml
# dsio split: patient_3fold (fold 0)
# store: kaggle_defog
# group key: group  <- the leakage boundary
# scheme: stratified_kfold, stratified by fog_count, seed 42
# counts: test=33, train=32, val=33
parts:
  train: [0489dc, 0e0908, ...]
  test:  [...]
  val:   [...]
store_manifest_sha256: a3f9c1...
```

The group is **the most leaky key** — the coarsest identifier that can make two windows
near-identical. Subject, machine, well, symbol. It is the smallest unit that may be
assigned to one side of a split, and the store records it per entity as a required field.

Files are **generated from a declarative `SplitSpec`, never hand-written** — the half of
the original objection worth keeping. Schemes: `holdout`, `kfold`, `stratified_kfold`,
`leave_one_group_out`, `explicit`. Stratified assignment is serpentine.

## Validation, which is the point

`SplitFile` rejects at construction:

- **parts that are not mutually disjoint** — the check FORGE's `SplitsConfig` lacked. It
  validated duplicates *within* each list but never *across* them, so a patient in both
  train and test would have passed silently.
- duplicates within a part;
- an empty split.

`resolve()` additionally rejects a split whose store name or manifest digest does not match
the store it is being applied to, and — unless explicitly waived — one that leaves any
group in the index unassigned, since silently dropping windows makes a fold train on less
data than its name claims.

## Why row-level overlap is not checked at resolve time

The property is structural, not incidental:

1. A window never crosses an entity boundary (`SignalStore.read` refuses).
2. Every entity belongs to exactly one group.
3. Split parts are disjoint over groups.

Therefore no raw row can appear in two parts. `assert_no_row_overlap()` verifies the
conclusion directly, and a test confirms it also *fires* when overlap is introduced
artificially — a check that never fails proves nothing. But it materialises every covered
row, so it belongs in tests and an explicit check command, not on the training path where
it would be O(windows × length) on 42M windows.

## Stratification is multi-key

One-key stratification is the easy half. The case that actually recurs in research is that
several metadata features matter at once — the label, plus protocol, site, device, sex —
and any of them confounded with the fold invalidates the result. Balancing them jointly
over ~100 groups is a combinatorial problem with no exact solution.

`SplitSpec.stratify` takes a list of `StratifyKey`, each with a kind (categorical or
numeric), a weight, and an explicit `aggregate` reducing an entity attribute to a
group-level value — summing an event count across a subject's recordings is right, summing
their age is not, so the reduction is never implicit. `stratify_by` remains as shorthand
for a single numeric key and expands into the general form immediately, so it does not
become a second code path.

Three properties worth stating:

**Numeric keys are balanced on distribution *and* mass.** A fold can hold the right number
of high-event subjects while holding the wrong total number of events. Both are in the
objective. Numeric levels come from quantile bins, not equal-width — a long-tailed event
count puts nearly every subject in one equal-width bin and balances nothing.

**Assignment is greedy by rarity, then refined by pairwise local search.** Groups carrying
the rarest level are placed first, while there is still freedom to place them. This
replaces the earlier serpentine dealer, which handled exactly one key.

**The result reports what it could not balance.** With twelve subjects and five keys,
perfect balance is impossible; `BalanceReport` records per-key level and mass deviation,
flags keys it could not bring within 15%, and warns when a level has fewer groups than
folds. The report is written into the split file header, so a reviewer sees what
stratification achieved without rerunning anything:

```
# stratify ok   events: levels dev 8%, mass dev 3%
# stratify POOR device: levels dev 50%
# WARNING device: level 'z' has fewer than 3 groups, so it cannot appear in every fold
```

Silently returning an unbalanced split is how a confound reaches a paper. Reporting it is
the difference between a split that can be judged and one that has to be trusted.

## Consequences

Splits survive as reviewable artifacts, LOOCV is one scheme rather than a special case, and
cross-validation over a single store costs masks rather than copies. Regenerating after the
corpus grows is a deliberate act that produces a git diff — which is the correct amount of
friction for changing what "test set" means.

## Temporal splits are the other half

A group list cannot express a temporal split, because the unit being divided is *time*. So
`SplitFile` carries an optional `temporal` section alongside `parts`, and a split may be
group-based, time-based, or both — the last being a purged walk-forward *within* a held-out
cohort, which is what a strategy needs when it must generalise over symbols and over time.

A naive time cut is wrong in two named ways, both of which silently inflate results:

**Purging.** A training window whose *label* extends into the test period has seen the test
period. With a label horizon of 5 bars, a window ending one bar before the test opens still
resolves inside it. Those windows are dropped, not merely ordered earlier.

**Embargo.** Serial correlation means a training window starting immediately after the test
period leaks the same autocorrelated regime. A gap is required, not optional.

The band removed by purge and embargo belongs to **neither** part and is genuinely
discarded — reassigning it to train reintroduces exactly the leak it was meant to remove.
The discarded count is reported in the file header, because if it is large the label
horizon is eating the dataset, and that should be a knowing decision rather than something
inferred from a training curve.

Time is measured **relative to each entity**, not by global row offset. Entity 2's row 1000
is not simultaneous with entity 1's row 1000, and cutting on global offsets would slice
every recording at a different point in its own history. `time_unit="epoch_s"` derives
absolute time from per-entity `t_start` and `sample_rate`, and fails loudly when they are
absent rather than falling back to row order and producing a plausible, wrong split.

Irregular per-row timestamps are **not** supported; both units assume a fixed rate within
an entity. That covers sensors and bar data. Tick data would need a per-row time array,
which is a further storage decision rather than a split one.
