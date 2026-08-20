"""A Dataset over the memory-mapped store, built from fold positions.

This is where Phase 2 pays for itself. A fold is integer positions into a
:class:`~dsio.data.views.WindowIndex`; a loader over it reads windows from the mmap on
demand. Nothing is copied, so five folds over 42M windows cost five position arrays rather
than five datasets — and switching window length costs a new index, not a new corpus.

The store's reader is keyed by process id and its live handles are dropped on pickling, so
worker processes reopen their own. That was verified under real DataLoader concurrency in
ADR 0005's worker-scaling benchmark before anything depended on it.

**The dataset owns the paradigm.** A pretext objective like masked reconstruction used to
be a stochastic slot the model applied only when ``self.training`` was set — a runtime flag
that a validation loop could get wrong without anything in a config file revealing it. Here
it is a constructor argument instead: a dataset built with ``mask=`` always masks, one built
without it never does, and which dataset a stage gets is a fact settled once when
``train_dataset``/``val_dataset`` hand loaders to Lightning — not a branch re-evaluated on
every batch.

**The target carries a NaN sentinel, not the whole original window.** Averaging a
reconstruction loss over every position — visible ones included — lets a model win by
copying the visible input, which is indistinguishable from having learned something. NaN is
the continuous analogue of MLM's ``-100``: a masked item's target holds the true value at
every position the mask hid, and NaN everywhere it did not, so a mask-aware loss reads
``(prediction, target)`` and nothing else to know which positions to score — no batch dict,
no subclass. See :class:`~dsio.nn.components.MaskedMSE`.

**A contrastive target only exists once a whole batch does.** SimCLR's negatives are "every
other window in the batch" and VICReg's invariance term needs to know which two rows are one
window's pair — neither is a fact about one item, so neither can live on the dataset the way
MAE's sentinel does. :class:`TwoViewCollate` builds both at collate time instead: it
augments each raw window twice and stacks the ``2 * batch`` views, with the target carrying
each row's pair index. See :class:`~dsio.nn.components.NTXent` and
:class:`~dsio.nn.components.VICReg`, which read that index and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dsio.data.store import SignalStore
from dsio.data.views import WindowIndex
from dsio.nn.masking import apply_mask
from dsio.runs.seeding import dataloader_kwargs

#: What a masking strategy from :mod:`dsio.nn.masking` looks like from here: called with
#: one window at a time (a fake batch of one) plus an optional generator, returning a
#: boolean hide-mask of the same shape's time axis. Every strategy in that module already
#: accepts ``generator`` — this widened signature just states what was already true, and
#: ``WindowDataset`` below is the first caller that actually passes one.
MaskStrategy = Callable[[torch.Tensor, torch.Generator | None], torch.Tensor]


class WindowDataset(Dataset[dict[str, Any]]):
    """Windows of one store, restricted to a set of index positions.

    ``positions`` are offsets into ``index``, which is what a
    :class:`~dsio.eval.contract.Fold` carries. Each item reports the position it came from
    so predictions can be realigned by identity rather than by trusting loader ordering.

    ``mask``, when given, turns this into a pretext dataset: ``__getitem__`` returns the
    masked window as ``x`` and, under ``target_key``, the original window with every
    *visible* position replaced by NaN — the sentinel a mask-aware loss selects on before
    it computes anything, so it never sees a position the model was allowed to look at.
    ``labels`` is ignored (a pretext window has no label to speak of; the target *is* the
    signal). Without ``mask`` an item is ``(x, label)`` exactly as before. Building a
    training dataset with a mask and a validation dataset without one is what makes "never
    mask a validation batch" a fact about which object was constructed, rather than a flag
    checked at call time.

    ``normalize_target`` (only meaningful alongside ``mask``) standardises the *target*
    per window, per channel — ``(x - mean) / (std + eps)`` computed over the whole
    original window before masking — while ``x`` itself stays raw. Reconstructing raw
    amplitude makes the loss dominated by whichever channel happens to have the largest
    units, so the model spends its capacity on the loudest sensor instead of learning
    something that generalises across channels. This is the same normalisation the deleted
    ``MaskedReconstruction.step()`` used to apply; it moved here because the target it
    normalises is now built here.

    ``mask_seed`` (only meaningful alongside ``mask``), when given, makes the mask a
    function of the item rather than of call order: each ``__getitem__`` builds a fresh
    ``torch.Generator`` seeded from ``mask_seed ^ position`` and hands it to ``mask``, so
    the same position always draws the same hidden span, regardless of which epoch, which
    worker, or how many times it has been fetched before. Left ``None`` (the default), the
    mask strategy draws from the process's global RNG, which is what training wants —
    fresh randomness every epoch is the point of a stochastic mask. A validation loader
    needs the opposite: :func:`~dsio.train.ssl_task.build_loaders` sets this only for the
    loader Lightning's validation loop consumes, so ``val/loss`` measures the same
    held-out reconstruction problem every time it is computed rather than a freshly
    redrawn one.
    """

    def __init__(
        self,
        store: SignalStore,
        index: WindowIndex,
        positions: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        *,
        channels_first: bool = True,
        mask: MaskStrategy | None = None,
        target_key: str = "y",
        normalize_target: bool = True,
        mask_seed: int | None = None,
    ) -> None:
        if index.store_name != store.path.name:
            raise ValueError(
                f"index was built for store {index.store_name!r}, not {store.path.name!r}"
            )
        self.store = store
        self.index = index
        self.positions = (
            np.arange(len(index), dtype=np.int64)
            if positions is None
            else np.asarray(positions, dtype=np.int64)
        )
        source = index.labels if labels is None else labels
        self.labels = None if source is None else np.asarray(source)
        if self.labels is not None and len(self.labels) != len(index):
            raise ValueError(
                f"labels has {len(self.labels)} entries for {len(index)} windows; they must "
                "be aligned with the whole index, not with this fold"
            )
        self.channels_first = channels_first
        self.mask = mask
        self.target_key = target_key
        self.normalize_target = normalize_target
        self.mask_seed = mask_seed

    def __len__(self) -> int:
        return int(self.positions.size)

    def __getitem__(self, i: int) -> dict[str, Any]:
        position = int(self.positions[i])
        start = int(self.index.starts[position])
        window = self.store.read(start, self.index.spec.length)
        # The store is [time, channels]; torch convolutions want [channels, time]. The copy
        # is required because the mmap slice is a view, and a view handed to a worker
        # process outlives the read it came from.
        array = np.ascontiguousarray(window.T if self.channels_first else window)
        x = torch.from_numpy(array).float()
        # row is carried regardless of which branch below runs: predictions are aligned to
        # folds by this identity, not by trusting the order a DataLoader hands batches back.
        item: dict[str, Any] = {"row": position}
        if self.mask is not None:
            # One window at a time, so the mask strategy sees a batch of one. The target
            # carries the true value at every hidden position and NaN everywhere else —
            # apply_mask with the mask inverted, since apply_mask fills where its mask is
            # True and here that is "visible", the opposite of what x's masking used.
            generator = None
            if self.mask_seed is not None:
                # Deterministic per position, not per call: the same window always draws
                # the same mask regardless of epoch, worker, or how many times it has
                # already been fetched. XOR rather than addition so the derived seed does
                # not walk monotonically with position, which would correlate neighbouring
                # windows' masks in a store where position order is meaningful.
                generator = torch.Generator().manual_seed(self.mask_seed ^ position)
            hidden = self.mask(x.unsqueeze(0), generator)
            item["x"] = apply_mask(x.unsqueeze(0), hidden).squeeze(0)
            target_source = x
            if self.normalize_target:
                # Per-window, per-channel standardisation of the target only -- x (what
                # the encoder sees) stays raw. Computed from the whole original window,
                # before masking, exactly like the deleted step()'s norm_target math.
                mean = x.mean(dim=-1, keepdim=True)
                std = x.std(dim=-1, keepdim=True) + 1e-6
                target_source = (x - mean) / std
            item[self.target_key] = apply_mask(
                target_source.unsqueeze(0), ~hidden, value=float("nan")
            ).squeeze(0)
        else:
            item["x"] = x
            if self.labels is not None:
                value = self.labels[position]
                item["y"] = torch.as_tensor(value)
        return item

    @property
    def groups(self) -> np.ndarray:
        """Group per item, for verifying a loader never mixes a split boundary."""
        return self.index.groups[self.positions]


def train_dataset(
    store: SignalStore,
    index: WindowIndex,
    positions: np.ndarray | None = None,
    *,
    labels: np.ndarray | None = None,
    channels_first: bool = True,
    mask: MaskStrategy | None = None,
    target_key: str = "y",
    normalize_target: bool = True,
    mask_seed: int | None = None,
) -> WindowDataset:
    """Build the dataset a training loader gets. The only builder that can mask.

    Paired with :func:`val_dataset`, whose signature has no ``mask`` parameter at all —
    "never mask a validation batch" is then a fact about which function a caller reached
    for, not a convention a caller has to remember to uphold by leaving an argument unset.

    ``mask_seed`` is left unset by a training caller (fresh randomness every epoch) and set
    by a caller building the loader Lightning's validation loop consumes over a masked
    fold — see :class:`WindowDataset`'s docstring.
    """
    return WindowDataset(
        store,
        index,
        positions,
        labels=labels,
        channels_first=channels_first,
        mask=mask,
        target_key=target_key,
        normalize_target=normalize_target,
        mask_seed=mask_seed,
    )


def val_dataset(
    store: SignalStore,
    index: WindowIndex,
    positions: np.ndarray | None = None,
    *,
    labels: np.ndarray | None = None,
    channels_first: bool = True,
    target_key: str = "y",
) -> WindowDataset:
    """Build the dataset a validation (or other never-masked) loader gets.

    There is no ``mask=`` keyword to pass here — the mistake this closes is not "someone
    remembered not to", it is "there was nothing to remember".
    """
    return WindowDataset(
        store,
        index,
        positions,
        labels=labels,
        channels_first=channels_first,
        target_key=target_key,
    )


class TwoViewCollate:
    """Batch collate that builds two augmented views per window, with a target carrying
    each row's pair index.

    Where MAE's pretext target lives on the dataset (a NaN sentinel written per item), a
    contrastive one only exists once a whole batch does: SimCLR's negatives are "every other
    row in the batch" and VICReg's invariance term needs to know which two rows are one
    window's pair. Both are facts about the *collated* batch, not about one item, so this is
    where they are built — the item-level dataset underneath is unchanged: whatever
    :func:`val_dataset` would hand a downstream classifier, raw and unmasked, since building
    two views needs no per-item mask.

    Each raw window is augmented twice, independently (``augment`` is stochastic), and the
    resulting ``2 * batch`` views are stacked so window ``i``'s two views sit at positions
    ``i`` and ``i + batch``. ``target[i]`` names ``i``'s partner: exactly the
    ``(arange(2 * batch) + batch) % (2 * batch)`` computation a contrastive loss used to
    compute inside its own training step. Moving it here is what lets that loss become an
    ordinary ``(prediction, target)`` loss — no batch dict, no subclass — because a
    ``(prediction, target)`` loss already receives the whole batch's predictions, and the
    negatives are just "every row that is not my partner". See
    :class:`~dsio.nn.components.NTXent` and :class:`~dsio.nn.components.VICReg`.

    ``seed``, when given, makes the two views a function of the batch's own rows rather
    than of call order: ``augment`` is called inside ``torch.random.fork_rng``, reseeded
    with ``seed`` XORed with the sum of the batch's row positions, so the same batch of
    rows always draws the same two views and the outer (global) RNG stream is left exactly
    as it was found. The augmentors here (``Jitter``, ``RandomScale``, ...) take no
    generator of their own — unlike a masking strategy — so this is the collate-level
    equivalent of ``WindowDataset``'s ``mask_seed``: unset for training (fresh augmentation
    every epoch), set only for the loader Lightning's validation loop consumes. It relies
    on that loader never shuffling, so the same rows land in the same batch, in the same
    order, on every call — true of every validation loader :func:`~dsio.train.ssl_task.
    build_loaders` builds.
    """

    def __init__(self, augment: nn.Module, target_key: str = "y", seed: int | None = None) -> None:
        self.augment = augment
        self.target_key = target_key
        self.seed = seed

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty batch")
        x = torch.stack([item["x"] for item in items])
        row = torch.as_tensor([item["row"] for item in items])
        batch = x.shape[0]
        views = self._augment_twice(x, row)
        pair = (torch.arange(2 * batch, device=views.device) + batch) % (2 * batch)
        return {"x": views, self.target_key: pair, "row": row.repeat(2)}

    def _augment_twice(self, x: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
        if self.seed is None:
            with torch.no_grad():
                return torch.cat([self.augment(x), self.augment(x)], dim=0)
        derived = self.seed ^ int(row.sum().item())
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(derived)
            return torch.cat([self.augment(x), self.augment(x)], dim=0)


def make_loader(
    dataset: WindowDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    drop_last: bool = False,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> DataLoader[dict[str, Any]]:
    """Build a DataLoader whose *shuffle order* does not depend on the worker count.

    Seeding the process is not enough: each worker gets its own RNG, so without an explicit
    generator and ``worker_init_fn`` the shuffle order varies with ``num_workers`` — which
    would make a result depend on a performance knob and put it straight into dsio's list
    of things that must never change an answer. ``dataloader_kwargs`` closes that.

    **This does not, on its own, make a mask or an augmented view worker-count-independent.**
    ``_seed_worker`` seeds Python's and NumPy's per-worker RNGs deterministically, but
    torch's own global RNG is seeded to ``base_seed + worker_id`` — different per worker by
    construction — so anything a masking strategy or ``collate_fn`` draws from torch's
    global RNG (the default when no generator is passed) genuinely varies with
    ``num_workers``. Measured, not assumed: a ``WindowDataset`` built with an unseeded mask
    reads a different hidden span at ``num_workers=0`` than at ``num_workers=2`` for the
    same position, and an unseeded :class:`TwoViewCollate` produces different views. That is
    fine for training — fresh randomness every epoch is the point — and exactly the problem
    for validation, which is why ``WindowDataset.mask_seed`` and ``TwoViewCollate.seed``
    exist: both draw from a generator built fresh from an explicit seed rather than from
    torch's global state, which is what actually makes them independent of worker count
    (and of call order, and of epoch) — verified in
    ``tests/nn/test_data.py::test_loader_result_does_not_depend_on_worker_count``.
    """
    kwargs: dict[str, Any] = dict(dataloader_kwargs(seed))
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **kwargs,
    )
