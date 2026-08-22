"""`execute()` seeds every RNG it can touch, so calling it directly is reproducible.

`dsio run` (``src/dsio/cli/run_cmd.py``) already called ``seed_everything`` before
starting a run, but nothing seeded a *direct* call to ``execute()`` — which is what every
runner test, and any caller that is not going through the CLI, actually does. A backbone's
weights are drawn from torch's global RNG at construction time, so two runs of the same
config used to disagree unless something upstream had happened to seed it. That silent gap
is what ``execute`` seeding at its own entry closes — this is dispatcher-level behaviour
(``dsio.train.runner.execute``), not specific to any one task kind.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch  # noqa: E402

from dsio.config.schema import RunConfig  # noqa: E402
from dsio.data.adapters import entity_examples  # noqa: E402
from dsio.data.store import DATA_ROOT_ENV, SignalStore  # noqa: E402
from dsio.data.views import WindowSpec  # noqa: E402
from dsio.model.registry import LABELS, labels  # noqa: E402
from dsio.runs.record import RunLedger  # noqa: E402
from dsio.splits.models import SplitFile, SplitFold  # noqa: E402
from dsio.train import load_runners  # noqa: E402
from dsio.train.runner import execute  # noqa: E402
from dsio.train.torch_task import Component, TorchTask, TrainerConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _runners() -> None:
    load_runners()


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A single tiny group pair — just enough for one fold to train on."""
    root = tmp_path / "stores"
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    rng = np.random.default_rng(0)
    with SignalStore.builder(root / "tone", channels=1) as builder:
        for group in range(4):
            positive = group % 2 == 0
            t = np.arange(200) / 100.0
            signal = (rng.standard_normal((200, 1)) * 0.5).astype("float32")
            if positive:
                signal[:, 0] += (np.sin(2 * np.pi * 5 * t) * 2.0).astype("float32")
            builder.add(
                f"p{group}", signal, group=f"p{group}", attrs={"positive": int(positive)}
            )

    if "tone" not in LABELS:

        @labels("tone")
        def _tone(store: SignalStore) -> np.ndarray:
            out = np.zeros(store.n_rows, dtype=np.float32)
            for entity in store.entities:
                out[entity.start_row : entity.end_row] = float(entity.attrs["positive"])
            return out

    store = SignalStore(root / "tone")
    digest = entity_examples(store).digest
    parts = {"test": ["p0"], "val": ["p1"], "train": ["p2", "p3"]}
    SplitFile(
        store=store.path.name,
        store_manifest_sha256=digest,
        name="k1",
        folds=[
            SplitFold(
                index=0,
                counts={part: len(members) for part, members in parts.items()},
                parts=parts,
            )
        ],
    ).save(tmp_path / "splits" / "k1" / "fold0.yaml")
    return tmp_path


def _run_once(config: RunConfig, runs_root: Path) -> dict[str, float]:
    ledger = RunLedger(runs_root)
    run = ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with run:
        return execute(config, run)


@pytest.fixture(autouse=True)
def _dirty_ambient_rng() -> None:
    """Every test process starts from the same fixed, default RNG state. Two direct calls
    to ``execute()`` can then look reproducible purely because both happened to start from
    that identical default state -- not because ``execute()`` actually reseeds anything.
    That is exactly the gap that let this test pass with ``execute()``'s own
    ``seed_everything`` call deleted, as long as it ran alone: nothing had touched the
    global RNGs yet, so both calls started level regardless. Perturbing every global RNG to
    a fixed, non-default value before the test removes that accident -- the assertion below
    can only hold if ``execute()`` resets the state itself, in isolation or not."""
    import random

    random.seed(20260821)
    np.random.seed(20260821)
    torch.manual_seed(20260821)


def test_execute_is_deterministic_for_the_same_config_and_seed(corpus: Path) -> None:
    """The headline promise — same config, same seed, identical metrics — for a caller
    that never goes through `dsio run`. A freshly constructed backbone is the thing that
    actually exercises the gap: its weights draw from torch's global RNG with no seed of
    its own, so they only reproduce if something upstream seeded it — and nothing here
    calls `seed_everything` directly, only `execute()` does."""
    config = RunConfig(
        name="det",
        seed=7,
        task=TorchTask(
            store="tone",
            window=WindowSpec(length=64, stride=32, label_policy="majority"),
            labels="tone",
            split="k1",
            fold=0,
            splits_root=corpus / "splits",
            backbone=Component(name="conv1d", params={"hidden": 4, "out_dim": 4, "depth": 1}),
            head=Component(name="linear", params={"out_dim": 2}),
            loss=Component(name="cross_entropy", params={"threshold": 0.5}),
            transform=Component(name="instance_standardize"),
            batch_size=4,
            # ``log_loss`` (continuous) is the sensitive half of this assertion --
            # ``accuracy`` alone saturates to 1.0 on this trivially separable toy corpus
            # regardless of the backbone's random initialisation, so a match on accuracy
            # proves nothing about whether the weights themselves reproduced.
            metrics=("accuracy", "log_loss"),
            trainer=TrainerConfig(
                max_epochs=2, accelerator="cpu", devices=1, checkpoint=False,
                enable_progress_bar=False,
            ),
        ),
    )
    first = _run_once(config, corpus / "runs_a")
    # Advance the global RNGs by an arbitrary amount that the first call would not itself
    # have consumed, so the two calls provably start from *different* ambient states. A
    # match below can then only be an effect of `execute()` resetting that state itself --
    # not an accident of both calls happening to start from wherever the first one left off.
    torch.rand(97)
    np.random.random(97)
    second = _run_once(config, corpus / "runs_b")
    assert first == second
