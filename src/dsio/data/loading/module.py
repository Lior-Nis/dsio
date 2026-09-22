"""The one Lightning data composition root."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset

from dsio.data.examples import Examples
from dsio.data.loading.collation import Collate
from dsio.data.loading.datasets import DatasetFactory, IdentityDataset, LoadingError
from dsio.data.loading.loaders import build_loader, validate_loader_options
from dsio.data.splits.models import SplitFile
from dsio.data.splits.validation import validate
from dsio.data.store import SignalStore

Phase = Literal["train", "validate", "test", "predict"]
_PHASES: tuple[Phase, ...] = ("train", "validate", "test", "predict")
_STAGES: dict[str | None, tuple[Phase, ...]] = {
    None: _PHASES,
    "fit": ("train", "validate"),
    "validate": ("validate",),
    "test": ("test",),
    "predict": ("predict",),
}


class DsioDataModule(LightningDataModule):
    """Replay one governed fold through deterministic, identity-preserving loaders."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del kwargs
        raise TypeError(
            "DsioDataModule cannot be subclassed; vary loading through injected components"
        )

    def __init__(
        self,
        store: SignalStore,
        examples: Examples,
        split: SplitFile,
        *,
        fold: int,
        roles: Mapping[str, str],
        dataset_factory: DatasetFactory,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        shuffle: Mapping[str, bool] | None = None,
        collate_fn: Collate | None = None,
    ) -> None:
        super().__init__()
        self.store = store
        self.examples = examples
        self.split = split
        self.fold_index = fold
        self.roles = _phase_mapping(roles)
        if not callable(dataset_factory):
            raise LoadingError("dataset_factory must be callable")
        self.dataset_factory = dataset_factory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = _shuffle_flags(shuffle)
        self.collate_fn = collate_fn
        self._loaders: dict[Phase, DataLoader[dict[str, Any]]] = {}
        validate_loader_options(batch_size=batch_size, num_workers=num_workers, seed=seed)

    def setup(self, stage: str | None = None) -> None:
        try:
            phases = _STAGES[stage]
        except KeyError:
            raise LoadingError(
                f"unsupported Lightning setup stage {stage!r}; expected fit, validate, "
                "test, predict, or None"
            ) from None
        validate(self.examples, self.split)
        fold = self.split.fold(self.fold_index)
        for phase in phases:
            if phase not in self.roles or phase in self._loaders:
                continue
            role = self.roles[phase]
            try:
                sample_ids = tuple(fold.assignments[role])
            except KeyError:
                raise LoadingError(
                    f"mapped role {role!r} is absent from split {self.split.name!r} "
                    f"fold {fold.index}; available roles: {sorted(fold.assignments)}"
                ) from None
            dataset = self._dataset(phase, sample_ids)
            self._loaders[phase] = build_loader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle[phase],
                num_workers=self.num_workers,
                seed=self.seed,
                collate_fn=self.collate_fn,
            )

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("validate")

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("test")

    def predict_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("predict")

    def _dataset(self, phase: Phase, sample_ids: tuple[str, ...]) -> IdentityDataset:
        try:
            dataset = self.dataset_factory(self.store, self.examples, sample_ids)
        except LoadingError:
            raise
        except Exception as error:
            raise LoadingError(
                f"dataset factory failed for phase {phase!r}, role {self.roles[phase]!r}: {error}"
            ) from error
        if not isinstance(dataset, Dataset):
            raise LoadingError(
                f"dataset factory for phase {phase!r} must return a torch Dataset, got "
                f"{type(dataset).__name__}"
            )
        if isinstance(dataset, IterableDataset):
            raise LoadingError(
                f"dataset factory for phase {phase!r} must return a map-style Dataset, "
                "not IterableDataset"
            )
        return IdentityDataset(dataset, sample_ids)

    def _loader(self, phase: Phase) -> DataLoader[dict[str, Any]]:
        try:
            return self._loaders[phase]
        except KeyError:
            raise LoadingError(
                f"{phase} loader has not been set up; map the phase and call setup() first"
            ) from None


def _phase_mapping(roles: Mapping[str, str]) -> dict[Phase, str]:
    if not isinstance(roles, Mapping) or not roles:
        raise LoadingError("a non-empty phase-to-role mapping is required")
    result: dict[Phase, str] = {}
    for phase, role in roles.items():
        if phase not in _PHASES:
            raise LoadingError(
                f"unsupported Lightning phase {phase!r}; expected one of {list(_PHASES)}"
            )
        if not isinstance(role, str) or not role:
            raise LoadingError(f"role mapped to phase {phase!r} must be a non-empty string")
        result[phase] = role
    return result


def _shuffle_flags(configured: Mapping[str, bool] | None) -> dict[Phase, bool]:
    result: dict[Phase, bool] = {phase: False for phase in _PHASES}
    result["train"] = True
    if configured is not None and not isinstance(configured, Mapping):
        raise LoadingError("shuffle must be a phase-to-bool mapping")
    for phase, enabled in (configured or {}).items():
        if phase not in _PHASES:
            raise LoadingError(f"shuffle names unsupported Lightning phase {phase!r}")
        if not isinstance(enabled, bool):
            raise LoadingError(f"shuffle for phase {phase!r} must be bool")
        result[phase] = enabled
    return result
