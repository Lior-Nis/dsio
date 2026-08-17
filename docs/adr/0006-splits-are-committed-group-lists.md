# 6. Splits are committed YAML lists of group IDs

Status: accepted (2026-08-17)
Amends: ADR 0004, item 6, which recommended *Repeatable Splitting* by hash of a stable ID.

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

Temporal splits with purge and embargo do not fit a list of group IDs and are **not yet
implemented**. They need row or time boundaries rather than group membership, and will be a
distinct scheme in the same file format.
