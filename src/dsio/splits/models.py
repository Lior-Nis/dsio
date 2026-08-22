"""Split files: committed YAML listing group IDs.

A split is a list of **group** identifiers, never a list of windows. The group is the
leakage boundary — the coarsest, leakiest key in the data, typically the subject, machine,
well or symbol. Two windows that share a group can be near-identical, so the group is the
smallest unit that may be assigned to one side of a split.

Naming the groups rather than deriving them is deliberate:

**Stratification needs deliberate assignment.** Hashing a group id into a fold cannot
balance a rare-event rate across folds; with few groups, random assignment reliably
produces imbalanced ones.

**Leave-one-group-out is not expressible as a hash.** "Leave group *i* out" is a list.

**A split is provenance.** A result has to state which groups were held out. A YAML file in
git is that statement, diffable and reviewable.

The file binds itself to a store by manifest digest, so a split cannot be silently applied
to a corpus it was not computed for.

One file holds a whole split **family**: every fold, in one committed, diffable place.
``SplitFold.index`` — not the fold's position in the list — is what a fold *means*; running
folds 1 and 3 alone must not renumber them 0 and 1, because an artifact directory and a
comparison both key off that number. Holding every fold together also means the one
property that spans folds — no group tested twice across the family — can be checked by
reading the YAML, before any store is opened or anything is resolved to row positions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from dsio.contracts import DsioModel
from dsio.splits.temporal import TemporalBounds

SCHEMA = "dsio.split/2"

#: The part a pooled out-of-fold metric is computed over. Hard-coded here (rather than
#: imported from `dsio.splits.folds`, which imports this module) because the cross-fold
#: disjointness guarantee is about this specific part, not an arbitrary caller-chosen one.
TEST_PART = "test"


class SplitError(ValueError):
    """Raised when a split is malformed, or contradicts the store it names."""


class SplitFold(DsioModel):
    """One division of the corpus: named parts of group IDs, temporal bounds, or both.

    Everything a single fold carries: its own parts, its own temporal bounds, its own
    declared ``index``. ``index`` is that fold's identity — the correspondence a run's
    artifacts and a comparison key off — and is independent of where this entry sits in
    its :class:`SplitFile`'s ``folds`` list.
    """

    index: int
    counts: dict[str, int] = Field(default_factory=dict)
    parts: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Group IDs per part. Empty for a purely temporal fold.",
    )
    temporal: TemporalBounds | None = Field(
        default=None,
        description="Time spans per part, with purge and embargo. Composes with `parts`.",
    )

    @model_validator(mode="after")
    def _validate_parts(self) -> SplitFold:
        """Reject the most damaging thing a fold can get wrong.

        Validating that no group appears twice *within* a part is the obvious check and the
        insufficient one. A group present in both train and test passes that check and
        silently invalidates every number the split produces, so disjointness is verified
        *across* parts as well. (Disjointness *across folds* is a different property,
        checked once per file — see :func:`_assert_test_parts_disjoint_across_folds`.)
        """
        if not self.parts and self.temporal is None:
            raise ValueError(
                f"fold {self.index} must define group parts, temporal bounds, or both"
            )

        for part, groups in self.parts.items():
            duplicates = [g for g, n in Counter(groups).items() if n > 1]
            if duplicates:
                raise ValueError(
                    f"fold {self.index} part {part!r} lists {len(duplicates)} group(s) "
                    f"more than once: {', '.join(sorted(duplicates)[:5])}"
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
                f"fold {self.index} parts must be mutually disjoint; "
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


class SplitFile(DsioModel):
    """One committed split family: every fold it has, in order.

    Parts are conventionally train/val/test, but the shape is open so a project can add
    its own (a calibration set, an external validation cohort). ``store`` and
    ``store_manifest_sha256`` bind the whole family to one corpus; individual folds carry
    no store identity of their own, so that binding is never duplicated per fold.
    """

    schema_version: str = SCHEMA
    store: str
    store_manifest_sha256: str | None = None
    group_key: str = "group"
    name: str
    notes: str | None = None
    folds: list[SplitFold] = Field(
        default_factory=list,
        description="Every fold in this split family, ordered. Each fold's own `index` "
        "is its identity, not its position in this list.",
    )

    @model_validator(mode="after")
    def _validate_folds(self) -> SplitFile:
        if not self.folds:
            raise ValueError("a split must declare at least one fold")

        duplicate_indices = sorted(
            {index for index, n in Counter(f.index for f in self.folds).items() if n > 1}
        )
        if duplicate_indices:
            raise ValueError(f"fold indices must be unique; repeated: {duplicate_indices}")
        return self

    def fold(self, index: int) -> SplitFold:
        """The fold declaring this index — looked up by declaration, never by position."""
        for candidate in self.folds:
            if candidate.index == index:
                return candidate
        raise SplitError(
            f"split {self.name!r} has no fold {index}; it defines folds "
            f"{sorted(f.index for f in self.folds)}"
        )

    def to_yaml(self) -> str:
        """Render with a human-readable provenance header.

        A split is meant to be read in a diff without parsing it, so the header states what
        it is and how it was produced. A single-fold family (the common case) gets the
        plain `# counts: ...` / `# temporal: ...` lines; a multi-fold family prefixes each
        with the fold it belongs to.
        """
        fold_numbers = ", ".join(str(f.index) for f in self.folds)
        word = "fold" if len(self.folds) == 1 else "folds"
        header = [
            f"# dsio split: {self.name} ({word} {fold_numbers})",
            f"# store: {self.store}",
            f"# group key: {self.group_key}  <- the leakage boundary",
        ]
        # The generating script is a project concern now; `notes` carries whatever it
        # wants to say about how this file was produced.
        multi = len(self.folds) > 1
        for f in self.folds:
            prefix = f"[{f.index}] " if multi else ""
            header.append(
                f"# {prefix}counts: "
                + ", ".join(f"{part}={n}" for part, n in sorted(f.counts.items()))
            )
            if f.temporal is not None:
                header.append(
                    f"# {prefix}temporal: unit={f.temporal.time_unit}, "
                    f"label_horizon={f.temporal.label_horizon:g}, "
                    f"embargo={f.temporal.embargo:g}"
                )
                for part, spans in sorted(f.temporal.spans.items()):
                    rendered = ", ".join(f"[{s.start:g}, {s.end:g})" for s in spans)
                    header.append(f"#   {part}: {rendered}")
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
        split = cls.model_validate(data)
        _assert_test_parts_disjoint_across_folds(split.name, split.folds)
        return split


def _assert_test_parts_disjoint_across_folds(name: str, folds: Sequence[SplitFold]) -> None:
    """No group may be assigned to the test part of two folds in the same family.

    Within a fold, disjointness across parts is `SplitFold`'s own concern. Across folds it
    is a different property, and the one a pooled out-of-fold metric depends on: a group
    tested by two folds is scored twice, silently reweighting the pooled number toward
    whichever groups were duplicated. This runs at :meth:`SplitFile.load`, so a malformed
    split family is rejected by reading the YAML — before a store is opened, before
    anything is resolved to row positions, strictly earlier than a check that needs
    `Examples` can fire.

    Raises `SplitError` (not the bare `ValueError` the model's own validators raise)
    because this is not something pydantic's construction machinery calls: it runs after
    `model_validate` succeeds, so its exception type is not rewrapped into a
    `pydantic.ValidationError` on the way out.
    """
    owners: dict[str, list[int]] = {}
    for f in folds:
        for group in f.parts.get(TEST_PART, []):
            owners.setdefault(group, []).append(f.index)

    collisions = {group: idxs for group, idxs in owners.items() if len(idxs) > 1}
    if not collisions:
        return

    offending = sorted({index for idxs in collisions.values() for index in idxs})
    detail = "; ".join(
        f"{group!r} in folds {idxs}" for group, idxs in sorted(collisions.items())[:5]
    )
    raise SplitError(
        f"split {name!r} test parts are not disjoint across folds {offending}: {detail}"
        + (f" (+{len(collisions) - 5} more group(s))" if len(collisions) > 5 else "")
    )
