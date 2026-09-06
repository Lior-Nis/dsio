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
is an ordinary registered one: the split, the leakage proof, the loader, the head, the loss
and the module below are all the existing signal machinery, unchanged.

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
from dsio.data.store import SignalStore  # noqa: E402
from dsio.data.views import WindowSpec, build_index  # noqa: E402
from dsio.dataset.dataset import make_loader, train_dataset, val_dataset  # noqa: E402
from dsio.model.components import CrossEntropy, EmbeddingEncoder, linear_head  # noqa: E402
from dsio.model.module import DsioModule  # noqa: E402
from dsio.splits.folds import folds_from_splits  # noqa: E402
from dsio.splits.models import SplitFile, SplitFold  # noqa: E402
from dsio.splits.resolve import assert_no_row_overlap, resolve  # noqa: E402

VOCAB, LENGTH, STRIDE, DIM = 64, 32, 16, 16
PARTS = {"train": ["doc0", "doc1", "doc2"], "val": ["doc3"], "test": ["doc4", "doc5"]}


@pytest.fixture
def corpus(tmp_path: Path) -> SignalStore:
    """Six documents of int32 token ids, of different lengths.

    ``channels=1`` because a document is one stream of ids, and the lengths differ because
    real documents do — the index, not the store, is what makes them uniform sequences.
    """
    path = tmp_path / "corpus"
    rng = np.random.default_rng(0)
    with SignalStore.builder(path, channels=1, dtype="int32") as builder:
        for doc, length in enumerate((300, 260, 240, 220, 200, 180)):
            ids = rng.integers(0, VOCAB, size=(length, 1), dtype=np.int32)
            builder.add(f"doc{doc}", ids, group=f"doc{doc}")
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
    assert np.array_equal(first[STRIDE:], second[:-STRIDE]), "half the tokens are shared"


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
    batch = next(iter(make_loader(val_dataset(corpus, index), batch_size=4)))
    assert batch["x"].dtype is torch.float32
    with pytest.raises(RuntimeError, match="Long, Int"):
        nn.Embedding(VOCAB, 8)(batch["x"])


def test_the_gradient_reaches_the_embedding_table(corpus: SignalStore, index) -> None:
    """embedding -> head -> loss, through the shared step every other paradigm uses."""
    torch.manual_seed(0)
    dataset = train_dataset(corpus, index, labels=_labels(index), payload_dtype=torch.long)
    batch = next(iter(make_loader(dataset, batch_size=8)))
    backbone = EmbeddingEncoder(vocab_size=VOCAB, embed_dim=8, out_dim=DIM)
    module = DsioModule(backbone=backbone, head=linear_head(DIM, 2), loss=CrossEntropy())

    loss = module._common_step(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()
    assert backbone.embedding.weight.grad is not None
    assert backbone.embedding.weight.grad.abs().sum() > 0


def test_a_token_corpus_trains_through_lightning(
    corpus: SignalStore, index, split: SplitFile
) -> None:
    """The whole chain, over a real fold and a real Trainer: nothing here is text-specific
    except the backbone and one constructor keyword."""
    torch.manual_seed(0)
    labels = _labels(index)
    fold = folds_from_splits(SignalExamples(corpus, index), [split])[0]
    train_loader = make_loader(
        train_dataset(corpus, index, fold.train, labels=labels, payload_dtype=torch.long),
        batch_size=8,
        shuffle=True,
    )
    val_loader = make_loader(
        val_dataset(corpus, index, fold.val, labels=labels, payload_dtype=torch.long),
        batch_size=8,
    )
    backbone = EmbeddingEncoder(vocab_size=VOCAB, embed_dim=8, out_dim=DIM)
    module = DsioModule(backbone=backbone, head=linear_head(DIM, 2), loss=CrossEntropy())
    before = backbone.embedding.weight.detach().clone()

    lightning.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    ).fit(module, train_loader, val_loader)

    assert not torch.equal(before, backbone.embedding.weight.detach())


# --- the vocabulary check costs one sync per run, not one per step -----------------------


def test_a_vocabulary_mismatch_is_named_rather_than_left_to_torch() -> None:
    """`nn.Embedding` says "index out of range in self", naming neither knob."""
    encoder = EmbeddingEncoder(vocab_size=10)
    with pytest.raises(ValueError, match="vocab_size is 10"):
        encoder(torch.tensor([[[0.0, 1.0, 42.0]]]))


def test_the_range_check_runs_once_rather_than_once_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`int(ids.max())` blocks the CPU until the GPU drains — once is fine, per-step is not.

    A GPU runs asynchronously; pulling a Python int out of a tensor forces a device
    synchronisation. In `forward` that is one stall per training step, and
    `assert_no_row_overlap`'s docstring already states this codebase's position: a check
    this expensive belongs outside the training path. A vocabulary mismatch is a config
    error — wrong on the first batch or never wrong — so the first batch is where it is
    worth paying for.
    """
    calls: list[int] = []
    real_max = torch.Tensor.max

    def counting_max(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return real_max(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "max", counting_max)

    encoder = EmbeddingEncoder(vocab_size=10)
    for _ in range(5):
        encoder(torch.zeros(2, 1, 4))

    assert len(calls) == 1, f"the range check synced the device {len(calls)} times in 5 steps"
