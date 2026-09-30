# Beyond Warehouse v1

Deferred scope. Nothing here is v1 work. Each item needs its own spec update or spec before
implementation.

## v2 — pretrained weights

- **Catalog entries for trained weights.** A weight is referenced by an immutable MLflow Logged Model or artifact URI and is never shipped inside the wheel. Each entry records:
  - architecture and configuration;
  - training dataset and split lineage;
  - expected preprocessing;
  - output schema;
  - benchmark metrics;
  - DSio version.
- **Encoder load-and-freeze/fine-tune from an immutable URI, with digest verification.** The first use is digit_recognizer's hand-rolled handoff. It builds on the existing `export_encoder`.
- **Projects decide whether weights fit their use.** DSio does not own promotion policy.

## Breadth consumers

Each of these proves new blocks under evidence gating:

| Consumer | Blocks it would prove |
|---|---|
| OTTO | Ranked/set prediction outputs, Recall@k, streaming ragged sessions, time-truncation splits |
| HMS | Paired modalities with different spans, soft-label KL objectives and metrics, accelerator augmentation (the first real use of `Jitter`/`RandomScale`), DDP and resume |
| Any consumer that needs attention/recurrent models | Transformer, GRU/LSTM and TCN backbones. None has a real use yet, so none is in v1 |
| A real self-supervised consumer | `NTXent`, `VICReg`, masking strategies, `OnlineProbe`, `RankMe` (currently unproven) |

## Deferred interface widenings

- **Multi-field model inputs.** Allow `x` plus declared extra fields (e.g. a separate validity mask or per-modality tensors) through `predict_step`, Predictor and export. v1 keeps the single-`x` contract.
- **Bounded-memory streaming evaluation.** Let `evaluate` consume batches instead of whole arrays. This would absorb cmi-seq's streaming evaluation and its window → participant aggregation (cohort #28).

## Discovery API

Add `list_models()`/`describe()`-style programmatic discovery only after the generated
catalog becomes too large to navigate. It must read the same docstring sections the catalog
uses, never a second registry.

## Integration API for Algua and Pulse

This milestone starts after v1 ships. Its target: one bounded real experiment runs through
ingestion, split, training, evaluation, predictor export and inference, using only a pinned
DSio release, with complete MLflow lineage. The stable surface is declared only after two
real consuming projects use the package independently.

## Scale and infrastructure claims still unproven

- Multi-GPU and multi-node DDP.
- Unbounded online streams.
- Remote object stores.
- Serving latency.
- Long-horizon checkpoint reliability.

Prove these with synthetic fault injection and infrastructure acceptance, not by distorting
reference consumers.
