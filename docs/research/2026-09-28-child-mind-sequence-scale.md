# Child Mind raw-sequence scale acceptance

Date: 2026-09-28

## Scope

The Child Mind Institute reference project trained a multimodal ordinal classifier from
the complete actigraphy corpus rather than precomputed aggregates. The experiment used
DSIO's `SignalStore`, grouped split generation, `DsioDataModule`, `DsioModule`, MLflow
evidence, and project-owned Prefect tasks.

- 314,569,149 source sensor rows from 996 participants
- 1,740 labelled participants without the sensor modality
- 2,736 labelled participants in total
- 40,646 windows of 8,192 rows
- 328,881,177 stored rows, including padding and missing-modality placeholders
- eight raw channels: `X`, `Y`, `Z`, `enmo`, `anglez`, `non-wear_flag`, `light`, and
  `battery_voltage`
- participant-disjoint `stratified_group_kfold` split
- inverse-window-count weighting so long recordings do not dominate training

The store is at
`/home/liornisimov/Datasets/dsio-runs/cmi-sequence-scale-20260928/a6ed6acfcb1345aa9da88a2ac9cd45e7/child-mind-sequences`.

## Reproducibility result

Two independent optimized executions selected the same calibrated training
configuration and produced identical scientific artifacts.

| Evidence | Acceptance run | Replay |
| --- | --- | --- |
| Data attempt | `379dfe2aaead422c9c7ae66499fda637` | `af5c0c879a004af682815420add844df` |
| Split attempt | `b271939a7a6445799c870fef1e15587c` | `41443f46417543efbcca0c7d10b5911a` |
| Training attempt | `d471559820e94a3ba6e0df666c91a05d` | `06a95ac5d05c463caa84a52793b2e923` |
| Export attempt | `5dcefcfbc5fa41caba0292a727dde4f9` | `835498304dc74063bd209c01e1ae6967` |
| Evaluation attempt | `2863c486a033455395a137a49ce014d0` | `9eb36719bf084710a9641740f3f57bde` |
| Logged Model | `models:/m-6f08d81e6a9c41b5ae9da3d196f711cf` | `models:/m-3b358978774a488995d1febb7236a20c` |
| Dataset digest | `45086bb798693d798843f3fab9740e32b91f4d8538ca3bf00df98f9a178cacae` | identical |
| Split digest | `2bcbd2021d776cc13dd4425d7c4a42ab6d1e90dd07705532a20a0a4d079651b4` | identical |
| Checkpoint digest | `0e3dc87fbbcfd12e8e5f8c9e12b8be1432cc087cb24c877257dbe91675d82380` | identical |

Attempt identifiers differ as expected. Data, split, export, and evaluation execution
identities are identical. The training identity differs because calibration records the
live available-host-memory budget, which changed between executions; both runs nevertheless
selected the same execution configuration and produced byte-identical checkpoints. Dataset,
split, checkpoint, and metric evidence are exact replays.

## Metrics

Both executions recorded the same values:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.5519911504 |
| Quadratic weighted kappa | 0.3792779674 |
| QWK, sensor present | 0.3776216134 |
| QWK, sensor missing | 0.3801402772 |
| QWK, sensor ablated | 0.3292040274 |
| Sensor ablation delta | 0.0484175859 |

The ablation delta demonstrates that the model consumed useful information from the raw
sensor modality while retaining an executable missing-modality path.

## Hardware calibration and throughput

Both runs used an NVIDIA GeForce RTX 5070 Ti with PyTorch 2.13.0+cu130. Calibration measured
ten effective batches per candidate and used a 10% tie band to resist timing noise. It
selected a micro-batch of 64, gradient accumulation of 2, pinned memory, and zero loader
workers in both executions. Eight-worker candidates were rejected because their process-tree
RSS exceeded the host-memory budget.

| Measurement | Acceptance run | Replay |
| --- | ---: | ---: |
| Training elapsed seconds | 33.744 | 32.590 |
| Mean GPU utilization | 8.62% | 9.06% |
| Peak GPU utilization | 30% | 29% |
| Peak CUDA allocation | 220,717,568 bytes | 220,711,424 bytes |
| Peak CUDA reservation | 532,676,608 bytes | 532,676,608 bytes |
| Peak process-tree RSS | 13,678,899,200 bytes | 13,899,141,120 bytes |
| Window-epochs per second | 2,409.09 | 2,494.40 |

Low GPU utilization is an observed property of this deliberately small reference model,
not evidence that the raw path was skipped. The experiment executes the complete raw
window corpus for two epochs and records CUDA allocation, utilization, and checkpoint
evidence. A larger model is a component benchmark question, not a prerequisite for the
pipeline acceptance proved here.

## Conclusion and limits

This run proves full-corpus staging, raw-window CUDA training, grouped split safety,
missing-modality training and evaluation, MLflow export, resource-aware calibration,
and exact replay. It does not claim competitive Kaggle quality or convergence: two
epochs and the compact reference network are a systems acceptance workload.

The evidence is visible in tailnet-only
[MLflow experiment 59](https://pop.tailee691f.ts.net:8443/#/experiments/59).
