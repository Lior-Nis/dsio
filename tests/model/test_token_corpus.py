"""A tokenized-text corpus, end to end: store -> context windows -> split -> training.

dsio's store format was already a text format — the ``.bin``/``.idx`` split in
``dsio.data.format`` is Megatron-LM's, and Megatron built it for token ids before anyone
borrowed it for signal. What was missing was not a format, an index or a loader; a spike
proved all three already carry a token corpus. What was missing was two things, and this
file is the proof that they close the gap:

**``WindowDataset`` used to hand every payload to the model as float32**, unconditionally.
Token ids arrived as ``39.0, 2.0, 41.0`` — which ``nn.Embedding`` refuses outright and
which the ``conv1d`` backbone happily convolved into meaningless numbers, the worse of the
two failures because it trains and converges. ``payload_dtype`` is the opt-in that stops it.

**There was no backbone that consumes ids.** ``EmbeddingEncoder`` is that backbone, and it
is an ordinary configured one: the split, leakage proof, loader, composition, objective,
and module below are shared machinery.

A document is an entity, a training sequence is a window, and the leakage boundary is the
document — which is why the split here is by group and why ``assert_no_row_overlap``, the
same detector a sensor corpus uses, is what proves no token reached two sides of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lightning = pytest.importorskip("lightning")

from torch import nn  # noqa: E402

from dsio.data.adapters import SignalExamples, entity_examples  # noqa: E402
from dsio.data.loading import build_loader  # noqa: E402
from dsio.data.splits.folds import folds_from_splits  # noqa: E402
from dsio.data.splits.models import SplitFile, SplitFold  # noqa: E402
from dsio.data.splits.resolve import assert_no_row_overlap, resolve  # noqa: E402
from dsio.data.store import SignalStore  # noqa: E402
from dsio.data.views import WindowSpec, build_index  # noqa: E402
from dsio.experimental.data import WindowDataset  # noqa: E402
from dsio.experimental.model import Chain, SupervisedObjective  # noqa: E402
from dsio.experimental.model.tokens import EmbeddingEncoder  # noqa: E402
from dsio.model.module import DsioModule  # noqa: E402

VOCAB, LENGTH, STRIDE, DIM = 64, 32, 16, 16
PARTS = {"train": ["doc0", "doc1", "doc2"], "val": ["doc3"], "test": ["doc4", "doc5"]}


@pytest.fixture
def corpus(tmp_path: Path) -> SignalStore:
    """Six documents of int32 token ids, of different lengths.

    Channel zero is the token stream and channel one is structural validity. Lengths differ
    because real documents do; the index, not the store, makes uniform sequences.
    """
    path = tmp_path / "corpus"
    rng = np.random.default_rng(0)
    with SignalStore.builder(path, channels=2, dtype="int32") as builder:
        for doc, length in enumerate((300, 260, 240, 220, 200, 180)):
            values = np.ones((length, 2), dtype=np.int32)
            values[:, 0] = rng.integers(0, VOCAB, size=length, dtype=np.int32)
            builder.add(f"doc{doc}", values, group=f"doc{doc}")
    return SignalStore(path)


@pytest.fixture
def index(corpus: SignalStore):
    """Overlapping context windows: stride below length, so each window shares half its
    tokens with its neighbour. That overlap is exactly what makes a naive split leak."""
    return build_index(corpus, WindowSpec(length=LENGTH, stride=STRIDE))


@pytest.fixture
def split(corpus: SignalStore) -> SplitFile:
    return SplitFile(
        store=corpus.path.name,
        store_manifest_sha256=entity_examples(corpus).digest,
        name="by_document",
        folds=[
            SplitFold(
                index=0,
                counts={part: len(members) for part, members in PARTS.items()},
                parts=PARTS,
            )
        ],
    )


def _labels(index) -> np.ndarray:
    """One class per document, aligned with the whole index — a document-level label, the
    shape a topic or sentiment corpus actually has."""
    return (index.entity_codes % 2).astype(np.int64)


def test_the_corpus_becomes_overlapping_context_windows(corpus: SignalStore, index) -> None:
    """No copy of the text: 32-token contexts at stride 16 are offsets into the one store."""
    assert len(index) > len(corpus.entities)
    assert index.spec.stride < index.spec.length, "the windows must actually overlap"
    first = corpus.read(int(index.starts[0]), LENGTH)
    second = corpus.read(int(index.starts[1]), LENGTH)
    assert np.array_equal(first[STRIDE:], second[:-STRIDE]), "half the rows are shared"


def test_a_document_split_leaks_no_token(corpus: SignalStore, index, split: SplitFile) -> None:
    """The leakage proof, run by the same detector a sensor corpus uses.

    Overlapping windows that straddle a document boundary would put near-identical token
    spans in train and test at once. Splitting by group — a document is a group — is what
    prevents it, and this is the direct check rather than the assumption.
    """
    assert_no_row_overlap(resolve(SignalExamples(corpus, index), split, split.fold(0)))


def test_the_float_default_is_what_blocked_an_embedding(corpus: SignalStore, index) -> None:
    """The measured blocker, kept as a test so it cannot come back silently.

    The default payload is float32 — correct for signal, and the reason token ids could not
    reach ``nn.Embedding`` at all before ``payload_dtype`` existed.
    """
    batch = next(iter(build_loader(WindowDataset(corpus, index), batch_size=4)))
    assert batch["x"].dtype is torch.float32
    with pytest.raises(RuntimeError, match="Long, Int"):
        nn.Embedding(VOCAB, 8)(batch["x"])


def _model() -> Chain:
    return Chain(
        preprocessor={
            "reference": "dsio.experimental.model.layout:ChannelFirstToTimeMajor",
            "parameters": {"channels": 2},
        },
        backbone={
            "reference": "dsio.experimental.model.tokens:EmbeddingEncoder",
            "parameters": {"vocab_size": VOCAB, "embed_dim": 8},
        },
        head={
            "reference": "torch.nn:Linear",
            "parameters": {"in_features": 8, "out_features": 2},
        },
    )


def _module() -> DsioModule:
    return DsioModule(
        model=_model(),
        objective=SupervisedObjective(loss={"reference": "torch.nn:CrossEntropyLoss"}),
    )


def test_the_gradient_reaches_the_embedding_table(corpus: SignalStore, index) -> None:
    """embedding -> head -> loss, through the shared step every other paradigm uses."""
    torch.manual_seed(0)
    dataset = WindowDataset(corpus, index, labels=_labels(index), payload_dtype=torch.long)
    batch = next(iter(build_loader(dataset, batch_size=8)))
    module = _module()

    loss = module._common_step(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()
    assert isinstance(module.model, Chain)
    assert isinstance(module.model.backbone, EmbeddingEncoder)
    assert module.model.backbone.embedding.weight.grad is not None
    assert module.model.backbone.embedding.weight.grad.abs().sum() > 0


def test_a_token_corpus_trains_through_lightning(
    corpus: SignalStore, index, split: SplitFile
) -> None:
    """The whole chain, over a real fold and a real Trainer: nothing here is text-specific
    except the backbone and one constructor keyword."""
    torch.manual_seed(0)
    labels = _labels(index)
    fold = folds_from_splits(SignalExamples(corpus, index), [split])[0]
    train_loader = build_loader(
        WindowDataset(corpus, index, fold.train, labels=labels, payload_dtype=torch.long),
        batch_size=8,
        shuffle=True,
    )
    val_loader = build_loader(
        WindowDataset(corpus, index, fold.val, labels=labels, payload_dtype=torch.long),
        batch_size=8,
    )
    module = _module()
    assert isinstance(module.model, Chain)
    assert isinstance(module.model.backbone, EmbeddingEncoder)
    before = module.model.backbone.embedding.weight.detach().clone()

    lightning.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    ).fit(module, train_loader, val_loader)

    assert not torch.equal(before, module.model.backbone.embedding.weight.detach())
