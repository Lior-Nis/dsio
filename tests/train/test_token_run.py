"""A token corpus must reach the model through `dsio run`, not only through Python.

`payload_dtype` landed on `WindowDataset` but not on `TorchTask`, so a token-id corpus
could be trained from a notebook and not from the entry point. That is not a smaller
version of the same thing: the runner is what captures provenance, writes the fold's
`predictions.npz`, and logs to MLflow. A path that skips it produces a model with no run
record, which is the one thing this spine exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("torch")
pytest.importorskip("lightning")

from dsio.config.schema import RunConfig  # noqa: E402
from dsio.data.adapters import entity_examples  # noqa: E402
from dsio.data.store import DATA_ROOT_ENV, SignalStore  # noqa: E402
from dsio.data.views import WindowSpec  # noqa: E402
from dsio.eval.contract import PREDICTIONS_FILE  # noqa: E402
from dsio.model.components import EmbeddingEncoder  # noqa: E402
from dsio.model.registry import LABELS, labels  # noqa: E402
from dsio.runs.record import start_run  # noqa: E402
from dsio.splits.models import SplitFile, SplitFold  # noqa: E402
from dsio.train.runner import execute  # noqa: E402
from dsio.train.torch_task import Component, TorchTask, TrainerConfig  # noqa: E402

VOCAB = 32


@pytest.fixture
def token_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Four documents per class; the class is which half of the vocabulary they draw from."""
    root = tmp_path / "stores"
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    rng = np.random.default_rng(0)
    with SignalStore.builder(root / "docs", channels=1, dtype="int32") as builder:
        for doc in range(8):
            positive = doc % 2 == 0
            low, high = (0, VOCAB // 2) if positive else (VOCAB // 2, VOCAB)
            ids = rng.integers(low, high, size=(600, 1))
            builder.add(
                f"d{doc}", ids, group=f"d{doc}", attrs={"positive": int(positive)}
            )

    if "topic" not in LABELS:

        @labels("topic")
        def _topic(store: SignalStore) -> np.ndarray:
            out = np.zeros(store.n_rows, dtype=np.float32)
            for entity in store.entities:
                out[entity.start_row : entity.end_row] = float(entity.attrs["positive"])
            return out

    store = SignalStore(root / "docs")
    SplitFile(
        store=store.path.name,
        store_manifest_sha256=entity_examples(store).digest,
        name="k1",
        folds=[
            SplitFold(
                index=0,
                counts={"train": 4, "val": 2, "test": 2},
                parts={
                    "train": ["d0", "d1", "d2", "d3"],
                    "val": ["d4", "d5"],
                    "test": ["d6", "d7"],
                },
            )
        ],
    ).save(tmp_path / "splits" / "k1" / "split.yaml")
    return tmp_path


def _task(root: Path, **overrides: object) -> TorchTask:
    defaults = dict(
        store="docs",
        window=WindowSpec(length=64, stride=32, label_policy="majority"),
        labels="topic",
        split="k1",
        fold=0,
        splits_root=root / "splits",
        backbone=Component(name="embedding", params={"vocab_size": VOCAB, "out_dim": 8}),
        head=Component(name="linear", params={"out_dim": 2}),
        loss=Component(name="cross_entropy", params={"threshold": 0.5}),
        payload_dtype="long",
        batch_size=16,
        metrics=("accuracy",),
        trainer=TrainerConfig(max_epochs=1, accelerator="cpu", devices=1, checkpoint=False),
    )
    return TorchTask(**{**defaults, **overrides})  # type: ignore[arg-type]


# --- the config surface -------------------------------------------------------------------


def test_a_task_can_ask_for_an_integer_payload(token_corpus: Path) -> None:
    assert _task(token_corpus).payload_dtype == "long"


def test_the_float_path_is_still_the_default(token_corpus: Path) -> None:
    """Signal is float regardless of how its bytes are packed; nothing existing moves."""
    assert _task(token_corpus, payload_dtype=None).payload_dtype is None
    assert TorchTask.model_fields["payload_dtype"].default is None


def test_an_unknown_payload_dtype_is_refused_at_config_time(token_corpus: Path) -> None:
    """A typo must cost microseconds, not the time it takes to load a corpus."""
    with pytest.raises(ValidationError):
        _task(token_corpus, payload_dtype="int65")


# --- the whole point: it runs from the entry point -----------------------------------------


def test_the_model_receives_integer_ids_through_the_runner(
    token_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the *model* is handed, observed inside a real `execute`.

    Asserting only that the run completes proves nothing here, and this test was written
    that way first and did not bite: `EmbeddingEncoder` accepts a float payload on
    purpose, casting back with a documented 2**24 bound, so a run with the dtype never
    threaded through trains perfectly happily on floats. The difference is invisible from
    outside -- which is the whole reason to look from inside.
    """
    seen: list[str] = []
    original = EmbeddingEncoder.forward

    def recording_forward(self: EmbeddingEncoder, x: object) -> object:
        seen.append(str(x.dtype))  # type: ignore[attr-defined]
        return original(self, x)  # type: ignore[arg-type]

    monkeypatch.setattr(EmbeddingEncoder, "forward", recording_forward)

    config = RunConfig(name="tokens", seed=0, task=_task(token_corpus))
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    execute(config, run)

    assert seen, "the backbone never ran"
    assert set(seen) == {"torch.int64"}, f"the model was handed {sorted(set(seen))}"


def test_a_token_corpus_trains_through_the_runner(token_corpus: Path) -> None:
    """`execute` is the path `dsio run` takes, so this is the claim "NLP works" rests on.

    The dtype is the test above; this one is the plumbing around it -- provenance, the
    fold's predictions, the metrics a pooled comparison later reads.
    """
    config = RunConfig(name="tokens", seed=0, task=_task(token_corpus))
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    metrics = execute(config, run)

    assert "accuracy" in metrics
    predictions = run.artifacts_dir / PREDICTIONS_FILE
    assert predictions.is_file(), "the fold's predictions must land like any other run's"
    with np.load(predictions, allow_pickle=False) as data:
        assert data["y_true"].size > 0
