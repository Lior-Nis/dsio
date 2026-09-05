"""Fixed-size items — images — as degenerate windows: length == stride == entity length.

A whole-item modality needs no item store and no second code path. An 8x8 RGB image is an
entity of 64 rows and 3 channels, and ``WindowSpec(length=64, stride=64)`` cuts exactly one
window from it: the image itself. That already worked the day windowing did, which is
precisely the problem — nothing said so, so nothing stopped a refactor of ``build_index``'s
enumeration or ``WindowDataset``'s transpose from silently ending computer-vision support
with the whole suite green. These tests are that stop.

What they pin is the *correspondence*, not the absence of an exception: 16 images in, 16
windows out, one per entity, every pixel row covered exactly once, and a batch that
reshapes from ``(B, C, H*W)`` to ``(B, C, H, W)`` with the original pixels at the original
coordinates. Random pixels rather than a formula, because a formula symmetric in H and W
would let a transposed image pass.

**Only fixed-size items.** ``WindowSpec.length`` is one int for the whole store, so a
corpus of differently-sized images has no single length that is "the item": at length 64 a
12x12 image yields two 64-row windows that are not images at all, and its last 16 rows are
dropped. That is deliberately not pinned here — it is a bug being fixed elsewhere, not a
behaviour to lock in — but it is the boundary, and ``README.md`` states it where someone
choosing a modality will read it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dsio.data.adapters import SignalExamples, entity_examples  # noqa: E402
from dsio.data.store import SignalStore  # noqa: E402
from dsio.data.views import WindowSpec, build_index  # noqa: E402
from dsio.dataset.dataset import WindowDataset, make_loader  # noqa: E402
from dsio.splits.models import SplitFile, SplitFold  # noqa: E402
from dsio.splits.resolve import assert_no_row_overlap, resolve  # noqa: E402

HEIGHT = 8
WIDTH = 8
CHANNELS = 3
IMAGES = 16
PER_PATIENT = 4

#: The whole point: an item is a window of exactly entity length. ``n_rows - length`` is then
#: zero, so each entity yields one window and the stride never gets the chance to land inside
#: an image — it is set to the length regardless, so the spec states the item size twice and a
#: store that later admits a longer entity slides over it a whole item at a time rather than
#: silently overlapping.
WHOLE_IMAGE = WindowSpec(length=HEIGHT * WIDTH, stride=HEIGHT * WIDTH)


@pytest.fixture
def images() -> np.ndarray:
    """Sixteen 8x8x3 uint8 images, held here so every assertion below compares pixels.

    Random, not generated from a formula in (row, column): a formula symmetric in the two
    axes cannot tell a correctly laid-out image from a transposed one, which is the single
    most likely way a row-major flattening breaks.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(IMAGES, HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)


@pytest.fixture
def store(tmp_path: Path, images: np.ndarray) -> SignalStore:
    """One entity per image: H*W rows of C channels, row-major.

    Four patients, four images each — the grouping is the reason to store images this way
    rather than as files in a directory, since it is what a split can then act on.
    """
    path = tmp_path / "scans"
    with SignalStore.builder(path, channels=CHANNELS, dtype="uint8") as builder:
        for i, image in enumerate(images):
            builder.add(
                f"img{i:02d}",
                image.reshape(HEIGHT * WIDTH, CHANNELS),
                group=f"patient{i // PER_PATIENT}",
            )
    return SignalStore(path)


@pytest.fixture
def index(store: SignalStore):
    return build_index(store, WHOLE_IMAGE)


# --- the correspondence -------------------------------------------------------------


def test_every_image_yields_exactly_one_window_starting_at_its_own_first_row(
    store: SignalStore, index
) -> None:
    """1:1, not merely "the right count".

    Sixteen windows could also be four images cut into four pieces each with twelve images
    missed entirely, which is what a stride that no longer tracks the entity boundary
    produces. Asserting that the starts *are* the entity start rows, in order, is what
    makes the count mean one window per image.
    """
    assert len(index) == IMAGES
    assert index.starts.tolist() == [entity.start_row for entity in store.entities]
    assert index.entity_ids.tolist() == [entity.entity_id for entity in store.entities]


def test_the_windows_cover_every_pixel_row_exactly_once(store: SignalStore, index) -> None:
    """No pixel dropped, no pixel counted twice — the fixed-size case's whole claim.

    ``covered_rows`` is unique, so it can only prove nothing was dropped; the length check
    against the window count is what rules out two windows overlapping on the same image.
    """
    assert index.covered_rows().tolist() == list(range(store.n_rows))
    assert store.n_rows == IMAGES * HEIGHT * WIDTH
    assert len(index) * WHOLE_IMAGE.length == store.n_rows


def test_an_item_is_the_image_it_came_from_pixel_for_pixel(
    store: SignalStore, index, images: np.ndarray
) -> None:
    """The flattening must be row-major and channels-first, and both must be reversible.

    A window arrives as ``(C, H*W)`` because ``WindowDataset`` transposes the store's
    ``[time, channels]`` for torch; reshaping that to ``(C, H, W)`` is only the image again
    if the store's rows were written in row-major pixel order. Comparing values, not
    shapes, is what catches a transpose that happens to be square.
    """
    dataset = WindowDataset(store, index)
    for i, image in enumerate(images):
        x = dataset[i]["x"]
        assert x.shape == (CHANNELS, HEIGHT * WIDTH)
        expected = torch.from_numpy(image.transpose(2, 0, 1).astype("float32"))
        assert torch.equal(x.reshape(CHANNELS, HEIGHT, WIDTH), expected)


def test_a_loader_batch_reshapes_to_b_c_h_w(store: SignalStore, index, images: np.ndarray) -> None:
    """What a convolutional backbone is handed, end to end.

    Each item reports the index position it came from, so the batch is checked against the
    images that position names rather than against loader order — the same identity
    discipline predictions are realigned by.
    """
    loader = make_loader(WindowDataset(store, index), batch_size=4, shuffle=False)
    seen = 0
    for batch in loader:
        x = batch["x"]
        assert x.shape == (4, CHANNELS, HEIGHT * WIDTH)
        picture = x.reshape(4, CHANNELS, HEIGHT, WIDTH)
        for slot, position in enumerate(batch["row"].tolist()):
            expected = images[position].transpose(2, 0, 1).astype("float32")
            assert torch.equal(picture[slot], torch.from_numpy(expected))
        seen += x.shape[0]
    assert seen == IMAGES


# --- the reason to store images this way at all --------------------------------------


def test_a_patient_split_puts_no_image_in_two_parts(store: SignalStore, index) -> None:
    """The leakage discipline medical imaging usually lacks, proved rather than asserted.

    Two scans of one patient are near-identical, so a per-image split scores a model on
    patients it trained on. Splitting on the group and then proving it at the row level —
    ``assert_no_row_overlap`` materialises every pixel row each part touches — is the
    check a group-only comparison cannot make: it would still pass if a window had
    straddled an image boundary.
    """
    split = SplitFile(
        store=store.path.name,
        store_manifest_sha256=entity_examples(store).digest,
        name="by_patient",
        folds=[
            SplitFold(
                index=0,
                counts={"train": 2, "test": 2},
                parts={"train": ["patient0", "patient1"], "test": ["patient2", "patient3"]},
            )
        ],
    )
    parts = resolve(SignalExamples(store, index), split, split.fold(0))

    assert_no_row_overlap(parts)
    assert {part: len(subset) for part, subset in parts.items()} == {"train": 8, "test": 8}
    assert set(parts["train"].index.entity_ids) & set(parts["test"].index.entity_ids) == set()
