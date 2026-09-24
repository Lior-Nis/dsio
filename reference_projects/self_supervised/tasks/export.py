"""Export an SSL checkpoint through the shared inference contract."""

from __future__ import annotations

from typing import Any

import torch
from prefect import task

from dsio.config.components import ComponentConfig, resolve_component
from dsio.inference import build_predictor, log_predictor
from dsio.tracking import attempt, record_provenance
from dsio.train.artifacts import ArtifactRef
from reference_projects.self_supervised.components import (
    EmbeddingNorm,
    TinyEmbedding,
    validate_embedding_norm,
)
from reference_projects.supervised.components import evaluation_arrays


@task(persist_result=False)
def export_model(
    data: dict[str, Any],
    split: dict[str, Any],
    training: dict[str, Any],
    experiment_id: str,
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        inputs, _ = evaluation_arrays(data["store_path"], split["assignments"]["test"])
        reference = ArtifactRef.model_validate(training["checkpoint"])
        preprocessor_config: ComponentConfig = {
            "reference": "reference_projects.supervised.components:TimeMajorToChannelFirst",
            "parameters": {
                "channels": TinyEmbedding.input_shape[0],
                "time": TinyEmbedding.input_shape[1],
            },
        }
        preprocessor = resolve_component(preprocessor_config, expected=torch.nn.Module)
        identity = record_provenance(
            run.info.run_id,
            {
                "checkpoint_digest": reference.digest,
                "dataset_digest": data["dataset_digest"],
                "export_form": "pyfunc",
                "preprocessor": preprocessor_config,
                "split_digest": split["split_digest"],
            },
            components={
                "builder": "dsio.inference.predictor:build_predictor",
                "model": "reference_projects.self_supervised.components:TinyEmbedding",
                "normalizer": "reference_projects.self_supervised.components:EmbeddingNorm",
                "preprocessor": preprocessor_config["reference"],
                "validator": (
                    "reference_projects.self_supervised.components:validate_embedding_norm"
                ),
            },
        )
        input_example = {
            "sample_id": inputs["sample_id"].tolist(),
            "x": torch.from_numpy(inputs["x"]),
        }
        predictor = build_predictor(
            reference,
            model=TinyEmbedding(),
            preprocessor=preprocessor,
            normalizer=EmbeddingNorm(),
            validator=validate_embedding_norm,
            input_example=input_example,
        )
        info = log_predictor(
            predictor,
            run_id=run.info.run_id,
            input_example=input_example,
            forms=("pyfunc",),
            name="self-supervised-reference",
        )["pyfunc"]
        return {
            "export_run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "checkpoint_digest": reference.digest,
            "identity": identity,
        }
