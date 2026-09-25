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

titanic_flow("data/titanic", "work/titanic", seed=19)
bike_sharing_flow("data/bike-sharing", "work/bike-sharing", seed=19)
digit_recognizer_flow("data/digit-recognizer", "work/digit-recognizer", seed=19)
store_sales_flow("data/store-sales", "work/store-sales", seed=19)
essay_scoring_flow("data/essay-scoring", "work/essay-scoring", seed=19)
rogii_flow("data/rogii", "work/rogii", seed=19)
parkinsons_fog_flow("data/parkinsons-fog", "work/parkinsons-fog", seed=19)
```

The Store Sales reference keeps the last 110 training days per store-family series, creates
five 30-day-context rolling origins, and predicts the 16 dates present in the official test
file. This is a bounded representative experiment, not a full-history leaderboard model.

The Essay Scoring reference hashes at most 512 tokens per essay, pads only at collation, and
uses masked mean pooling for ordinal 1–6 predictions. QWK and the collator remain local to the
consumer until an unrelated second project justifies admission to DSIO.

The ROGII reference joins each horizontal well to its typewell, excludes train wells whose raw
IDs also occur in the official test set, and pads hidden tails only at collation. Its learned
model is a bounded residual around the last known TVT, and evaluation compares it with that
explicit baseline. The collator and masked RMSE remain consumer-local pending component review.

The Parkinson's reference streams each signal CSV into at most 512-row windows, groups splits
by subject, and excludes every training recording belonging to an official test subject. The
defog `Valid AND Task` mask governs both loss and metric calculation; tdcsfog rows are fully
valid. Inference restores all three probabilities to the official row IDs and ordering. The
full competition download is about 70 GB, so use Kaggle's `-f` option for a bounded subset
unless you explicitly intend to run the scale tier.

The checked-in tests generate tiny deterministic CSVs with the official schemas. Their
metrics prove only plumbing, leakage controls, replay, evidence, and submission shape. They
do not validate model quality on the official datasets and make no leaderboard claim.

No flow downloads data, reads Kaggle credentials, submits predictions, or includes official
rows. A run writes `submission.csv` locally for human inspection and optional manual use.
