# Kaggle consumer portfolio

These are independent, unshipped consumer projects. They exercise public DSIO APIs;
they are not DSIO modes, templates, or leaderboard claims.

| Project | Official competition slug | Expected local files |
| --- | --- | --- |
| Titanic | `titanic` | `train.csv`, `test.csv` |
| Bike Sharing Demand | `bike-sharing-demand` | `train.csv`, `test.csv` |
| Digit Recognizer | `digit-recognizer` | `train.csv`, `test.csv` |
| Store Sales — Time Series Forecasting | `store-sales-time-series-forecasting` | `train.csv`, `test.csv` |
| Automated Essay Scoring 2.0 | `learning-agency-lab-automated-essay-scoring-2` | `train.csv`, `test.csv`, `sample_submission.csv` |
| ROGII Wellbore Geology Prediction | `rogii-wellbore-geology-prediction` | `train/`, `test/`, `sample_submission.csv` |
| Parkinson's Freezing of Gait Prediction | `tlvmc-parkinsons-freezing-gait-prediction` | metadata CSVs, `train/{defog,tdcsfog}/`, `test/{defog,tdcsfog}/`, `sample_submission.csv` |
| Child Mind Institute — Problematic Internet Use | `child-mind-institute-problematic-internet-use` | `train.csv`, `test.csv`, `sample_submission.csv`, `series_{train,test}.parquet/id=*/part-0.parquet` |

Accept each competition's rules and configure Kaggle credentials yourself. Downloads are
deliberately opt-in and outside every flow:

```bash
kaggle competitions download -c titanic -p data/titanic
kaggle competitions download -c bike-sharing-demand -p data/bike-sharing
kaggle competitions download -c digit-recognizer -p data/digit-recognizer
kaggle competitions download -c store-sales-time-series-forecasting -p data/store-sales
kaggle competitions download -c learning-agency-lab-automated-essay-scoring-2 -p data/essay-scoring
kaggle competitions download -c rogii-wellbore-geology-prediction -p data/rogii
kaggle competitions download -c tlvmc-parkinsons-freezing-gait-prediction -p data/parkinsons-fog
kaggle competitions download -c child-mind-institute-problematic-internet-use -p data/child-mind
```

Unzip each archive, then call the project-owned flows directly:

```python
from reference_projects.kaggle.titanic.flow import titanic_flow
from reference_projects.kaggle.bike_sharing.flow import bike_sharing_flow
from reference_projects.kaggle.digit_recognizer.flow import digit_recognizer_flow
from reference_projects.kaggle.store_sales.flow import store_sales_flow
from reference_projects.kaggle.essay_scoring.flow import essay_scoring_flow
from reference_projects.kaggle.rogii.flow import rogii_flow
from reference_projects.kaggle.parkinsons_fog.flow import parkinsons_fog_flow
from reference_projects.kaggle.child_mind.flow import child_mind_flow

titanic_flow("data/titanic", "work/titanic", seed=19)
bike_sharing_flow("data/bike-sharing", "work/bike-sharing", seed=19)
digit_recognizer_flow("data/digit-recognizer", "work/digit-recognizer", seed=19)
store_sales_flow("data/store-sales", "work/store-sales", seed=19)
essay_scoring_flow("data/essay-scoring", "work/essay-scoring", seed=19)
rogii_flow("data/rogii", "work/rogii", seed=19)
parkinsons_fog_flow("data/parkinsons-fog", "work/parkinsons-fog", seed=19)
child_mind_flow("data/child-mind", "work/child-mind", seed=19)
```

The Store Sales reference keeps the last 110 training days per store-family series, creates
five 30-day-context rolling origins, and predicts the 16 dates present in the official test
file. This is a bounded representative experiment, not a full-history leaderboard model.

The Essay Scoring reference hashes at most 512 tokens per essay, pads only at collation, and
uses masked mean pooling for ordinal 1–6 predictions. It shares DSIO's registered QWK metric
with Child Mind; its task-specific collator remains consumer-local.

The ROGII reference joins each horizontal well to its typewell, excludes train wells whose raw
IDs also occur in the official test set, and pads hidden tails only at collation. Its learned
model is a bounded residual around the last known TVT, and evaluation compares it with that
explicit baseline. The collator and masked RMSE remain consumer-local pending component review.

The Parkinson's reference streams each signal CSV into at most 512-row windows, groups splits
by subject, and excludes every training recording belonging to an official test subject. The
defog `Valid AND Task` mask governs both loss and metric calculation; tdcsfog rows are fully
valid. Inference restores all three probabilities to the official row IDs and ordering. Its
default call above remains the small fixture/subset flow. An explicit scale call additionally
accepts the centralized labelled and daily roots plus a CUDA `TrainerConfig`; it resolves the
official inventory before timing or staging and refuses unavailable CUDA rather than falling
back to CPU.

## Official Parkinson scale evidence

The 2026-09-28 run is recorded in tailnet-only [MLflow experiment
57](https://pop.tailee691f.ts.net:8443/#/experiments/57). The verified claim is **full labelled
training plus full-directory bounded iteration**:

- inventory resolved 91 defog and 833 tdcsfog training recordings, 46 non-task CSVs, 65 daily
  Parquets, and both test files. It ignored 46 unrelated defog and 206 unrelated tdcsfog files;
- the [scan run](https://pop.tailee691f.ts.net:8443/#/experiments/57/runs/e5f5ba230c3d46f69efea8d3045ed181)
  read all 4,230,953,001 non-supervised rows and 69,176,980,663 source bytes in 65,536-row
  batches. The daily lane alone contributed 4,220,701,887 rows and 68,337,598,024 bytes;
- scanning took 174.045 seconds (397,466,549 source bytes/s) with uncontrolled OS page cache
  and 3,394,113,536 bytes maximum sampled process-tree RSS, below the 4 GiB acceptance budget.
  The sampler read Linux `VmRSS` at 0.1-second intervals and recursively followed every
  thread's child PIDs;
- ingestion resolved all 924 labelled sources, preserved the 18-recording test-subject
  exclusion, and staged 12,228,916 points in 24,285 windows. The 23,724 eligible labelled
  windows then trained for two epochs on one RTX 5070 Ti through `DsioDataModule` and
  `DsioModule`;
- the two completed fit epochs took 114.204 seconds; checkpoint and artifact I/O were outside
  that interval. Training peaked at 7,925,714,944 maximum sampled process-tree RSS bytes,
  2,485,248 CUDA-allocated bytes, 4,194,304 CUDA-reserved bytes, and 10% sampled GPU
  utilization. Mean utilization across 959 samples was 0.084%, so this tiny convolution is not
  evidence of GPU saturation. CUDA evidence resolved physical device
  `GPU-e498255a-65f9-b6ea-967a-89ce193dde50`;
- evaluation recorded 0.053587 mean Average Precision against a 0.046423 prevalence baseline,
  and inference restored all 286,370 official rows. The 22,368,376-byte submission has SHA-256
  `25530255563a893726f5229007ed7a55a95b9341dc856bc16dd301896fbad77e`.

The inventory manifest identity is
`ae5b027376f0ef3a84ef0d9dac4eaac44e01f54cdcaf1382b0f1a9dfafdbb0cf`; it covers lane,
recording ID, size, and nanosecond mtime, rather than file content. The separate full-scan
content checksum is `0065465932ace6be473bd579c5791c00ffeec2c399f5a2eee61c4fe43771e21e`.
The lineage runs are ingest `8cb303b863954fed98776d0f9cf0aef3`, split
`152df4d3bffa4ba7a34a9eede6676c3a`, train `2171f2c90b9445cbb8e1ea254595a0cc`, export
`d35b335bde6a4e9ebf06f4796247de10`, evaluation `fe92d87e688c44c88c9dedf64336e349`, and
inference `9f848519c420479d8f90578f758e9ccc`. This does not mean the model trained on the 68.34
GB daily lane: supervised training used only the labelled recordings. It is not a DDP,
leaderboard-quality, production-performance, or deployment claim.

The Child Mind reference requires the optional `pyarrow` data dependency (`uv sync --extra
data`). It reads every participant Parquet partition in bounded batches and stages only
deterministic summary features, never a second copy of the raw sensor corpus. One shared
stratified participant split feeds both a tabular-only model and a fused model. Missing values
and absent sensor partitions remain explicit in the packed input; preprocessing statistics are
fit from the training partition only. Evaluation records overall QWK, QWK by sensor
availability, and a same-participant sensor ablation before emitting `id,sii` submissions.

## Official Child Mind evidence

The 2026-09-27 full-corpus replay is recorded in tailnet-only [MLflow experiment
58](https://pop.tailee691f.ts.net:8443/#/experiments/58). Both attempts processed 2,736
labelled participants, all 996 train sensor partitions, 315,008,875 sensor rows, and
6,727,640,239 source bytes. They produced the same dataset and split digests, execution
identities, predictions, metrics, and submission bytes under distinct attempt IDs.

| Evidence | First attempt | Replay attempt |
| --- | --- | --- |
| Ingest / split | `e076185457e548dabe5169707556ec6d` / `a817ac11f439429486614c83dd159aa5` | `1f7d6cf32c3a43ce8331be63cc00d17c` / `eb49eb9345794a9d968ba7abd69c1c0d` |
| Tabular train / evaluation | `70f124de90934df38f027976d6005104` / `f5bc42c4385545bab494e90ca65f7f7e` | `b6261224a867434693e2aef3222133d3` / `21d317d6eaa74c3da3638f89696bdf06` |
| Fused train / evaluation | `a337ba7ae58c43de91d54704d6bc2a04` / `e77f00a2c6374b7eaa1caf2ff58ce50a` | `5405199dd0bf4d649f32aa77db1b79e6` / `ff6624fa3c774ef9abd53e3410a897e7` |

The tabular model recorded accuracy `0.464912` and QWK `0.359652`. The fused model recorded
accuracy `0.482456`, overall QWK `0.368758`, sensor-present QWK `0.361094`, sensor-missing
QWK `0.373657`, ablated QWK `0.333343`, and an ablation delta of `+0.027751`. The replay's
ready exported models are `models:/m-d567a34769a440d9a7d6d01813822ab7` (tabular) and
`models:/m-2df3b580a33b40188b00327fd5a558bf` (fused). Inference attempts
`31996d8284984c26aabfba6847ba3fb2` and `e4bbb57d6f9a433fbbaa387a4fce077c`
each retain a 20-row `id,sii` submission; their replay-stable SHA-256 digests are
`c82e90366a53ceb188445af0f8b7b01ad2254f91553159ee919849e1243f99f5` and
`9d9d6e73cc657333e16c0eb0e4e5a9378b4238bca135c05dba48050990a819af` respectively.

The checked-in tests generate tiny deterministic CSVs with the official schemas. Their
metrics prove only plumbing, leakage controls, replay, evidence, and submission shape. They
do not validate model quality on the official datasets and make no leaderboard claim.

No flow downloads data, reads Kaggle credentials, submits predictions, or includes official
rows. A run writes `submission.csv` locally for human inspection and optional manual use.
