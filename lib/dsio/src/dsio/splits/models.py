"""Split files: committed YAML listing group IDs.

A split is a list of **group** identifiers, never a list of windows. The group is the
leakage boundary — the coarsest, leakiest key in the data, typically the subject, machine,
well or symbol. Two windows that share a group can be near-identical, so the group is the
smallest unit that may be assigned to one side of a split.

Naming the groups rather than deriving them is deliberate:

**Stratification needs deliberate assignment.** Hashing a subject ID into a fold cannot
balance a rare-event rate across folds. FORGE stratified 128 patients serpentine by FOG
count precisely because random assignment produced badly imbalanced folds.

**Leave-one-group-out is not expressible as a hash.** "Leave patient *i* out" is a list.

**A split is scientific provenance.** A paper has to state which subjects were in test. A
YAML file in git is that statement, diffable and reviewable.

The file binds itself to a store by manifest digest, so a split cannot be silently applied
to a corpus it was not computed for.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dsio.contracts import DsioModel, short_digest
from dsio.splits.stratify import BalanceReport, StratifyKey

SCHEMA = "dsio.split/1"

Scheme = Literal[
    "holdout",
    "kfold",
    "stratified_kfold",
    "leave_one_group_out",
    "explicit",
]


class SplitError(ValueError):
    """Raised when a split is malformed, or contradicts the store it names."""


class SplitSpec(DsioModel):
    """Declarative description of how to generate splits.

    The spec is what you write; the YAML files are what it produces. Generation is
    reproducible from the spec, and the output is committed so that reading a result never
    requires re-running the generator.
    """

    scheme: Scheme
    group_key: str = Field(
        default="group",
        description="Which key is the leakage boundary. 'group' is the store's own.",
    )
    k: int = Field(default=1, ge=1, description="Number of folds. 1 for holdout.")
    seed: int = Field(default=42, ge=0)
    stratify: tuple[StratifyKey, ...] = Field(
        default=(),
        description=(
            "Features to balance across folds — label, protocol, site, device, whatever "
            "you know matters. Balanced jointly, not one at a time."
        ),
    )
    stratify_by: str | None = Field(
        default=None,
        description="Shorthand for a single numeric key; expands into `stratify`.",
    )
    val_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    always_train: tuple[str, ...] = Field(
        default=(),
        description="Groups pinned to train, e.g. subjects with zero positive events.",
    )

    @model_validator(mode="after")
    def _check(self) -> SplitSpec:
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"val_fraction + test_fraction must leave room for train, got "
                f"{self.val_fraction} + {self.test_fraction}"
            )
        if self.scheme in {"kfold", "stratified_kfold"} and self.k < 2:
            raise ValueError(f"{self.scheme} needs k >= 2, got {self.k}")
        if self.scheme == "stratified_kfold" and not (self.stratify or self.stratify_by):
            raise ValueError(
                "stratified_kfold requires stratify=[...] or the stratify_by shorthand"
            )
        names = [key.name for key in self.stratify]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"stratify names a key more than once: {', '.join(sorted(duplicates))}"
            )
        return self

    @property
    def keys(self) -> list[StratifyKey]:
        """All stratification keys, with the shorthand expanded.

        The shorthand exists because one numeric key is the common case; it must not become
        a second code path, so it expands into the general form immediately.
        """
        keys = list(self.stratify)
        if self.stratify_by and self.stratify_by not in {k.name for k in keys}:
            keys.append(StratifyKey(name=self.stratify_by, kind="numeric"))
        return keys

    @property
    def digest(self) -> str:
        return short_digest(self.model_dump(mode="json"))


class SplitFile(DsioModel):
    """One materialised split: named parts, each a list of group IDs.

    Parts are conventionally train/val/test, but the shape is open so a project can add
    its own (a calibration cohort, an external validation set).
    """

    schema_version: str = SCHEMA
    store: str
    store_manifest_sha256: str | None = None
    group_key: str = "group"
    name: str
    fold: int | None = None
    spec: SplitSpec | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    balance: BalanceReport | None = Field(
        default=None,
        description="What stratification achieved. Recorded so a split can be judged.",
    )
    notes: str | None = None
    parts: dict[str, list[str]]

    @model_validator(mode="after")
    def _validate_parts(self) -> SplitFile:
        """Reject the failure FORGE could not catch.

        FORGE's SplitsConfig validated that no patient appeared twice *within* a list, but
        never that the three lists were disjoint from each other. A subject present in both
        train and test would have passed silently — and that is the single most damaging
        thing a split file can get wrong.
        """
        if not self.parts:
            raise ValueError("a split must define at least one part")

        for part, groups in self.parts.items():
            duplicates = [g for g, n in Counter(groups).items() if n > 1]
            if duplicates:
                raise ValueError(
                    f"part {part!r} lists {len(duplicates)} group(s) more than once: "
                    f"{', '.join(sorted(duplicates)[:5])}"
                )

        seen: dict[str, str] = {}
        collisions: list[str] = []
        for part, groups in self.parts.items():
            for group in groups:
                if group in seen:
                    collisions.append(f"{group!r} in both {seen[group]!r} and {part!r}")
                else:
                    seen[group] = part
        if collisions:
            raise ValueError(
                "split parts must be mutually disjoint; "
                + "; ".join(sorted(collisions)[:5])
                + (f" (+{len(collisions) - 5} more)" if len(collisions) > 5 else "")
            )
        return self

    @property
    def all_groups(self) -> set[str]:
        return {group for groups in self.parts.values() for group in groups}

    def part_of(self, group: str) -> str | None:
        for part, groups in self.parts.items():
            if group in groups:
                return part
        return None

    def to_yaml(self) -> str:
        """Render with a human-readable provenance header, as FORGE's split files had."""
        header = [
            f"# dsio split: {self.name}"
            + (f" (fold {self.fold})" if self.fold is not None else ""),
            f"# store: {self.store}",
            f"# group key: {self.group_key}  <- the leakage boundary",
        ]
        if self.spec is not None:
            header.append(
                f"# scheme: {self.spec.scheme}"
                + (f", stratified by {self.spec.stratify_by}" if self.spec.stratify_by else "")
                + f", seed {self.spec.seed}"
            )
        header.append(
            "# counts: " + ", ".join(f"{part}={n}" for part, n in sorted(self.counts.items()))
        )
        if self.balance is not None:
            header.extend(self.balance.summary_lines())
        if self.notes:
            header.append(f"# {self.notes}")
        body = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True, width=100)
        return "\n".join(header) + "\n" + body

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        from dsio.contracts import atomic_write

        atomic_write(path, self.to_yaml().encode("utf-8"))

    @classmethod
    def load(cls, path: Path) -> SplitFile:
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        if data.get("schema_version") != SCHEMA:
            raise SplitError(
                f"{path} declares schema {data.get('schema_version')!r}, expected {SCHEMA!r}"
            )
        return cls.model_validate(data)
