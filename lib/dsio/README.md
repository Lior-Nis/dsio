# dsio

The reproducible ML/DL experimentation spine.

`dsio` owns the parts every project rebuilds badly: typed configuration, staged data with
content-addressed caching, leakage-safe splits, a run ledger that makes results
reconstructible, evaluation with honest verdicts, and a resumable job matrix. It does not
own your models — a tabular task uses a real scikit-learn `Pipeline`, a deep task a real
`LightningModule`, forecasting real Nixtla objects. There is no universal `Model` wrapper.

This package is the shared spine of the workspace. Treat it as read-only inside a fork:
send fixes upstream so every project gets them, and pull them back with `copier update`.
Project code lives in the workspace root and depends on this package, never the reverse —
a direction now enforced by packaging.

See the repository root README for the design principles, and `docs/adr/` for the
decisions and their reasons.
