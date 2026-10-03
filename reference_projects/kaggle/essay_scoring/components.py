"""Importable text, model, metric, and prediction components for Essay Scoring."""

from __future__ import annotations

import hashlib
import re

from dsio.config.components import ComponentConfig

MAX_TOKENS = 512
VOCAB_SIZE = 4096
_TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[int]:
    """Map text to stable, dependency-free hash buckets; zero remains padding."""
    values = []
    for token in _TOKEN.findall(text.casefold())[:MAX_TOKENS]:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        values.append(int.from_bytes(digest, "little") % (VOCAB_SIZE - 1) + 1)
    if not values:
        raise ValueError("an essay must produce at least one token")
    return values


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "int64"},
        "y": {
            "from": "attribute",
            "attribute": "target",
            "dtype": "int64",
            "offset": -1,
        },
    },
}
INPUTS: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {"x": {"from": "data", "dtype": "int64"}},
}
COLLATOR: ComponentConfig = {
    "reference": "dsio.experimental.data.padding:PadCollator",
    "parameters": {
        "padded_fields": {"x": 0},
        "fixed_fields": ["y"],
        "emit_mask": False,
    },
}
MODEL: ComponentConfig = {
    "reference": "dsio.experimental.model.compositions:Chain",
    "parameters": {
        "backbone": {
            "reference": "dsio.experimental.model.tokens:EmbeddingEncoder",
            "parameters": {"vocab_size": VOCAB_SIZE, "embed_dim": 32},
        },
        "head": {
            "reference": "torch.nn:Linear",
            "parameters": {"in_features": 32, "out_features": 6},
        },
    },
}
OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {
        "loss": {"reference": "torch.nn:CrossEntropyLoss"},
        "target_dtype": "int64",
    },
}
OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs.multiclass:MulticlassOutput",
    "parameters": {"classes": 6, "scores": True, "label_offset": 1},
}
