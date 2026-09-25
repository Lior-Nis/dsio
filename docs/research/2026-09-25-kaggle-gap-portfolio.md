# Kaggle portfolio for DSIO's remaining experiment gaps

Date: 2026-09-25

## Recommendation

There are two lanes. Start the first immediately while a human accepts the rules for the
second.

### Start here

1. **Store Sales — Time Series Forecasting** (124.76 MB): establish the first explicit
   multi-series forecasting contract. Access is already verified.
2. **Automated Essay Scoring 2.0** (36.2 MB, executed): cheaply prove variable-length text,
   ordinal prediction, and a nontrivial metric.
3. **ROGII Wellbore Geology Prediction** (1.33 GB, download verified): exercise multi-source
   variable-length sequences and hidden-tail regression by well.
4. **Parkinson's Freezing of Gait Prediction** (70.59 GB, download verified): use a bounded
   subset first, then make it the first full-scale ragged/windowed sequence and mask test.

The account is already entered in Parkinson, ROGII, and Essay Scoring; authenticated downloads
were verified. Store Sales access and download were subsequently verified as well.

### Gated expansion: accept rules first

5. **Child Mind Institute — Problematic Internet Use**: multimodal joins and missing modalities.
6. **OTTO — Multi-Objective Recommender System**: streaming sessions and ranked/set output.
7. **HMS — Harmful Brain Activity Classification**: paired signals, soft targets, GPU
   augmentation, and final DDP stress.
8. **Google — Isolated Sign Language Recognition**: use only if Parkinson leaves ragged
   batching, missing-landmark handling, or many-file loading insufficiently tested.

**Ventilator Pressure Prediction** is an optional fixed-length sequence-to-sequence bridge.
Skip it if Parkinson and Store Sales already expose the necessary contracts. ASL is likewise a
conditional stress test rather than part of the minimal core because Parkinson already covers
ragged sequence loading at scale.

As of 2026-09-25, authenticated downloads for ASL, CMI, OTTO, HMS, and Ventilator return HTTP
403 because this Kaggle account has not accepted their rules. Their official manifests remain
visible and their pages offer Late Submission. This is a manual legal-acceptance prerequisite,
not a DSIO failure.

All claims below come from official Kaggle competition pages and the official authenticated
Kaggle API.

## Executed evidence

Store Sales completed its representative-data gate on 2026-09-25 against DSIO `v0.2.0`:

- 3,000,888 official training rows were read with bounded per-series history;
- 1,782 store-family series produced five rolling origins each;
- one real Lightning model trained on the latest governed fold and exported as an immutable
  MLflow Predictor;
- held-out RMSLE was **0.5282**, compared with **0.6170** for the weekly seasonal-naive
  baseline;
- inference produced all **28,512** predictions as a `[1782, 16]` forecast matrix and restored
  official row identity in `submission.csv`;
- the final evaluation [is visible in MLflow](https://pop.tailee691f.ts.net:8443/#/experiments/54/runs/07f286491d3a4c78921e02a8f01b695a).

The run corrected the planning assumption from a 15-step to a 16-step horizon and removed an
unbounded list of all training row IDs from the streaming CSV reader. Neither finding required
a DSIO-core abstraction change.

Automated Essay Scoring 2.0 then completed its representative-data gate:

- 17,307 official training essays were staged as variable-length token sequences, capped at
  512 tokens for this bounded baseline;
- dynamic batch padding and masked mean pooling trained through the canonical DataModule and
  Lightning module with two loader workers;
- held-out accuracy was **0.4493** and Quadratic Weighted Kappa was **0.4738**;
- inference preserved Kaggle's three raw test IDs even though all three overlap training IDs;
- the final evaluation [is visible in MLflow](https://pop.tailee691f.ts.net:8443/#/experiments/55/runs/b79fc3fa38134dc59666bad9f98c1bed).

The official data corrected two synthetic assumptions: essays may contain surrounding
whitespace and train/test raw IDs need not be globally disjoint. Text is now normalized and
internal sample identity is source-namespaced without changing submission identity. QWK and
the collator stay consumer-local as first-use candidates; neither warrants a DSIO-core change
before an unrelated second use. The two-worker run did expose one generic defect: Linux fork
workers were unsafe inside Prefect's multithreaded process. DSIO loaders now use a spawn context
when workers are enabled, and both the focused regression and official flow pass with it.

## Ready-now competitions

### Parkinson's Freezing of Gait Prediction

Source: [official competition](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction).

- **Data:** 70.59 GB, 1,044 CSV/parquet files; variable-duration 100/128 Hz three-axis series.
- **Target/metric:** three dense per-timestep event targets (`StartHesitation`, `Turn`, and
  `Walking`); mean Average Precision across event classes.
- **Why it matters:** it simultaneously tests ragged/windowed IO, ignored or unannotated regions,
  masks, subject-disjoint splits, identity-preserving dense predictions, and full-data loading.
- **Pass condition:** subject boundaries survive staging/splitting; ignored timestamps affect
  neither loss nor metric; full-data iteration has bounded memory; export restores every
  series/timestep identity.

### ROGII Wellbore Geology Prediction

Source: [official competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction).

- **Data:** 1.33 GB, 2,327 CSV/PNG/PPTX files; horizontal trajectories join variable reference
  logs by well.
- **Target/metric:** hidden-tail true vertical depth (`TVT`) sequence regression; RMSE.
- **Why now:** it is a manageable multi-source/ragged test with explicit per-well identity and
  windowed output, without the scale and code-submission constraints of ASL/HMS.
- **Pass condition:** joins are identity-safe, folds are well-disjoint, hidden tails never leak,
  and inference reconstructs depth-indexed predictions.

### Automated Essay Scoring 2.0

Source: [official competition](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2).

- **Data:** 36.2 MB and 17,307 labelled variable-length essays in the downloaded bundle.
- **Target/metric:** ordinal scores 1–6; Quadratic Weighted Kappa.
- **Why now:** the cheapest proof that variable-length text, ordinal schemas, and QWK fit the
  existing spine. It also provides an independent QWK use before CMI.
- **Pass condition:** length handling is deterministic, QWK matches a fixed fixture, and export
  preserves essay identity and ordinal bounds.

## Coverage

| Gap | First proof | Independent/stress proof |
|---|---|---|
| Forecasting and rolling-origin splits | Store Sales | ROGII hidden-tail regression |
| Variable-length input and masking | Essay Scoring / ROGII | Parkinson; ASL only if needed |
| Dense sequence output | Parkinson | Ventilator only if needed |
| Multimodal and missing modalities | CMI | HMS |
| Ordinal, soft, and ranked targets | Essay Scoring / CMI | HMS / OTTO |
| Bounded-memory loading at scale | Parkinson | OTTO / HMS |
| Distributed sampler correctness | synthetic two-process test | Parkinson or HMS on real GPUs |

## 1. Store Sales — Time Series Forecasting

Sources: [official overview/evaluation](https://www.kaggle.com/competitions/store-sales-time-series-forecasting),
[official data description](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).

- **Task:** predict grocery-family sales by date and store for the 16 dates present in the
  downloaded test bundle (2017-08-16 through 2017-08-31). Kaggle's page describes this as 15
  days after the training period, but the data contract is authoritative for tensor shape.
- **Data:** 124.76 MB, seven CSV files. Besides train/test, there are store, transaction,
  oil-price, holiday/event, and promotion data.
- **Target/metric:** non-negative `sales`; Root Mean Squared Logarithmic Error.
- **Access/status:** ongoing Getting Started competition with a rolling leaderboard; Join and
  rule acceptance are required.
- **DSIO evidence:** rolling-origin split generation, fixed 16-step windows, known-future
  covariates, multi-series identities, vector predictions, and leakage-safe replay.
- **Constraint:** the competition is intentionally approachable. It validates semantics, not
  scale or distributed execution.

**Pass condition:** a seasonal-naive reference and one Lightning model use the same project
flow; two deterministic rolling origins are leakage-free; a 16-step forecast survives export
and inference; RMSLE and downstream-only reevaluation are visible in MLflow.

## 2. Google Brain — Ventilator Pressure Prediction

Sources: [official overview/evaluation](https://www.kaggle.com/competitions/ventilator-pressure-prediction),
[official data description](https://www.kaggle.com/competitions/ventilator-pressure-prediction/data).

- **Task:** predict airway pressure at each time step of an approximately three-second breath
  from two control-signal series and lung attributes.
- **Data:** three CSV data files totaling about 666 MB in the official API listing. Rows carry
  a global `id`, `breath_id`, timestamp, controls, lung resistance/compliance, and pressure.
- **Target/metric:** dense per-timestep `pressure`; Mean Absolute Error.
- **Access/status:** closed Research Prediction Competition with Late Submission.
- **DSIO evidence:** grouped sequence assembly, no cross-breath windows, sequence-to-sequence
  prediction identity, masked phase loss if the project chooses it, and fast inference over
  many short sequences.

**Pass condition:** shuffled source rows reconstruct the same breaths; no batch/window crosses
a `breath_id`; exported predictions map exactly back to row IDs; checkpoint resume reproduces
the next epoch's sample order.

**Skip rule:** do not promote a Ventilator-specific abstraction. If existing window/collation
APIs handle it cleanly and ASL is ready, record the pass and move on.

## 3. Google — Isolated Sign Language Recognition

Sources: [official data description](https://www.kaggle.com/competitions/asl-signs/data),
[official dataset card](https://www.kaggle.com/competitions/asl-signs/overview/description).

- **Task:** classify one isolated sign from a variable-duration landmark sequence drawn from
  a 250-sign vocabulary.
- **Data:** 56.43 GB and 94,479 parquet/CSV/JSON files. The corpus contains roughly 100,000
  processed videos from 21 signers. Frames contain face, pose, left-hand, and right-hand
  MediaPipe landmarks; detections may be missing.
- **Target/metric:** one `sign` class per sequence; classification accuracy.
- **Access/status:** closed Research Code Competition with Late Submission. Kaggle expects a
  TensorFlow Lite submission, but DSIO training may remain PyTorch/Lightning.
- **DSIO evidence:** variable-length sampling, padding/masks or length bucketing, participant-
  grouped splits, missing landmark values, many-small-file throughput, and single-device vs
  two-device sampler correctness.
- **Constraint:** TensorFlow Lite conversion is a consumer export adapter, not justification
  for adding TensorFlow or mobile deployment concerns to DSIO.

**Pass condition:** a batch contains different sequence lengths without padded frames affecting
loss or accuracy; signer-disjoint folds are reproducible; each example is visited according to
the declared sampler policy; predictions preserve `sequence_id`; DDP smoke metrics match one
device within a documented tolerance.

## 4. Child Mind Institute — Problematic Internet Use

Source: [official data description](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data).

- **Task:** predict the ordinal Severity Impairment Index (`sii`, 0–3).
- **Data:** 6.73 GB and 1,002 parquet/CSV files. Participant tabular measurements join to
  optional per-participant actigraphy series spanning as much as 30 days. Many measures and
  some training targets are missing.
- **Target/metric:** ordinal `sii`; Quadratic Weighted Kappa.
- **Access/status:** closed Code Competition with Late Submission; the complete test set is
  hidden and supplied only during Kaggle notebook reruns.
- **DSIO evidence:** identity-safe tabular/series joins, optional modalities, long variable
  histories, missing-target filtering, multimodal collation, and reusable ordinal metrics.
- **Constraint:** local held-out evaluation is the DSIO acceptance signal. Kaggle notebook
  packaging and hidden-test execution remain project concerns.

**Pass condition:** tabular-only and tabular-plus-actigraphy models share the same DSIO spine;
missing series are explicit rather than fabricated; QWK matches a hand-worked fixture; staging,
training, inference, and MLflow retain participant identity and modality availability.

## 5. OTTO — Multi-Objective Recommender System

Sources: [official overview/evaluation](https://www.kaggle.com/competitions/otto-recommender-system),
[official data description](https://www.kaggle.com/competitions/otto-recommender-system/data).

- **Task:** from a truncated e-commerce session, predict up to 20 products for each of clicks,
  carts, and orders.
- **Data:** 11.89 GB of JSONL plus sample submission. Each session contains a time-ordered,
  variable-length list of product ID, timestamp, and event type.
- **Target/metric:** three ranked/set outputs per session; Recall@20 weighted 0.10 for clicks,
  0.30 for carts, and 0.60 for orders.
- **Access/status:** closed Prediction Competition; official pages expose the manifest but
  rule acceptance is required for download.
- **DSIO evidence:** streaming JSONL ingestion, ragged histories, time-truncation split logic,
  multi-head prediction schemas, ranked output, and full-data loader memory behavior.
- **Constraint:** candidate generation and recommender-specific negative sampling belong in
  the consumer. DSIO should not acquire a recommender framework.

**Pass condition:** representative and full-data tiers have bounded memory; truncation never
leaks future events; Recall@20 handles duplicates and empty targets correctly; output restores
session/type identity; throughput and peak memory are logged to MLflow.

## 6. HMS — Harmful Brain Activity Classification

Sources: [official overview/evaluation](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification),
[official data description](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data).

- **Task:** classify harmful brain activity from raw EEG and matched spectrogram views.
- **Data:** 26.4 GB and 28,463 parquet/PDF/CSV files. Training metadata identifies offset
  windows within consolidated EEG and spectrogram recordings. EEG is sampled at 200 Hz;
  annotators reviewed 50-second EEG and matched 10-minute spectrogram contexts.
- **Target/metric:** a six-class expert-vote probability distribution; Kullback-Leibler
  divergence. Predicted probabilities must sum to one.
- **Access/status:** closed Research Code Competition with Late Submission; only sample test
  data are downloadable, while the full test set is injected for notebook scoring.
- **DSIO evidence:** paired signal modalities with different time spans, offset-aware sampling,
  patient-group splitting, soft-label metrics/loss, GPU-side signal augmentation, large-scale
  loading, and final DDP/resume validation.
- **Constraint:** this is the last test because it combines several hard dimensions. Avoid
  diagnosing every failure as a missing DSIO abstraction; first isolate loading, collation,
  augmentation, model, and distributed concerns.

**Pass condition:** raw-only, spectrogram-only, and paired models use one spine; augmentation
is deterministic under seed and runs at the intended device boundary; KL is correct for soft
targets; two-device training has no duplicate patient/window leakage; interrupted training
resumes with equivalent sampler and MLflow lineage.

## Scope-creep guardrails

The consumer owns each competition's Prefect DAG, feature engineering, model choice, loss,
candidate generation, Kaggle adapter, and submission packaging. DSIO changes are eligible only
when the experiment exposes a generic invariant, such as:

- a reusable split strategy;
- identity-preserving padding/masking/bucketing;
- a generally useful metric or prediction schema;
- scalable loading behind the existing data interface;
- correct Lightning/DDP sampling, logging, checkpointing, or inference behavior.

Do not add competition result dataclasses, a task registry, a universal multimodal container,
a recommender system, a forecasting framework, TensorFlow/TFLite support, or Kaggle notebook
orchestration. One passing competition is evidence for an experimental component; a stable
component still requires an unrelated second use under `docs/component-admission.md`.

## What Kaggle will not prove

Even a clean sweep leaves several claims untested:

- true unbounded/online `IterableDataset` streams and backpressure;
- multi-node networking, node loss, elastic recovery, and preemption under real infrastructure;
- remote object-store consistency, credentials, bandwidth, and partial failures;
- production serving latency, concurrency, request batching, and deployment rollback;
- weeks-long checkpoint reliability and MLflow server/database failure recovery;
- the declared Python-version/platform matrix;
- project lifecycle, scheduling, and deployment, which are intentionally outside DSIO.

Test those separately with synthetic fault-injection and infrastructure acceptance tests. Do
not distort a Kaggle consumer to simulate deployment concerns DSIO does not own.

## Execution protocol

Each competition gets three tiers:

1. **Contract:** tiny deterministic fixture, one batch, one epoch, local MLflow.
2. **Representative:** real subset, multiple workers, validation, export/inference, checkpoint
   resume, and downstream-only reevaluation.
3. **Scale:** full training data, with throughput, peak RAM/VRAM, utilization, and run evidence
   logged. Add DDP at ASL as a smoke test and claim support only after HMS independently passes.

A competition is complete only after an actual model trains, browser-visible MLflow runs show
metrics and artifacts, exported inference reproduces held-out predictions, and a clean
environment resolves the recorded package/lock/code evidence. Leaderboard submission is useful
external feedback, not a substitute for these checks.

## Immediate next action

Build the ROGII Wellbore Geology consumer against DSIO `v0.2.0`. Use it as the unrelated
variable-length sequence test for the padding/masking seam, while keeping well joins,
geological features, hidden-tail model, and depth-indexed submission packaging consumer-local.
Only propose a DSIO component if the real flow demonstrates the same invariant with a second
task family.
