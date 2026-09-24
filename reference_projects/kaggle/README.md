# Kaggle consumer portfolio

These are three independent, unshipped consumer projects. They exercise public DSIO APIs;
they are not DSIO modes, templates, or leaderboard claims.

| Project | Official competition slug | Expected local files |
| --- | --- | --- |
| Titanic | `titanic` | `train.csv`, `test.csv` |
| Bike Sharing Demand | `bike-sharing-demand` | `train.csv`, `test.csv` |
| Digit Recognizer | `digit-recognizer` | `train.csv`, `test.csv` |

Accept each competition's rules and configure Kaggle credentials yourself. Downloads are
deliberately opt-in and outside every flow:

```bash
kaggle competitions download -c titanic -p data/titanic
kaggle competitions download -c bike-sharing-demand -p data/bike-sharing
kaggle competitions download -c digit-recognizer -p data/digit-recognizer
```

Unzip each archive, then call the project-owned flows directly:

```python
from reference_projects.kaggle.titanic.flow import titanic_flow
from reference_projects.kaggle.bike_sharing.flow import bike_sharing_flow
from reference_projects.kaggle.digit_recognizer.flow import digit_recognizer_flow

titanic_flow("data/titanic", "work/titanic", seed=19)
bike_sharing_flow("data/bike-sharing", "work/bike-sharing", seed=19)
digit_recognizer_flow("data/digit-recognizer", "work/digit-recognizer", seed=19)
```

The checked-in tests generate tiny deterministic CSVs with the official schemas. Their
metrics prove only plumbing, leakage controls, replay, evidence, and submission shape. They
do not validate model quality on the official datasets and make no leaderboard claim.

No flow downloads data, reads Kaggle credentials, submits predictions, or includes official
rows. A run writes `submission.csv` locally for human inspection and optional manual use.
