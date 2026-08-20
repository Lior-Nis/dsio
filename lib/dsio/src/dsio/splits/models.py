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
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from dsio.contracts import DsioModel
from dsio.splits.temporal import TemporalBounds

SCHEMA = "dsio.split/1"


class SplitError(ValueError):
    """Raised when a split is malformed, or contradicts the store it names."""


class SplitFile(DsioModel):
    """One materialised split: named parts, each a list of group IDs.

    Parts are conventionally train/val/test, but the shape is open so a project can add
    its own (a calibration set, an external validation cohort).
    """

    schema_version: str = SCHEMA
    store: str
    store_manifest_sha256: str | None = None
    group_key: str = "group"
    name: str
    fold: int | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    notes: str | None = None
    parts: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Group IDs per part. Empty for a purely temporal split.",
    )
    temporal: TemporalBounds | None = Field(
        default=None,
        description="Time spans per part, with purge and embargo. Composes with `parts`.",
    )

    @model_validator(mode="after")
    def _validate_parts(self) -> SplitFile:
        """Reject the most damaging thing a split file can get wrong.

        Validating that no group appears twice *within* a part is the obvious check and the
        insufficient one. A group present in both train and test passes that check and
        silently invalidates every number the split produces, so disjointness is verified
        *across* parts as well.
        """
        if not self.parts and self.temporal is None:
            raise ValueError(
                "a split must define group parts, temporal bounds, or both"
            )

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
        """Render with a human-readable provenance header.

        A split is meant to be read in a diff without parsing it, so the header states what
        it is and how it was produced.
        """
        header = [
            f"# dsio split: {self.name}"
            + (f" (fold {self.fold})" if self.fold is not None else ""),
            f"# store: {self.store}",
            f"# group key: {self.group_key}  <- the leakage boundary",
        ]
        # The generating script is a project concern now; `notes` carries whatever it
        # wants to say about how this file was produced.
        header.append(
            "# counts: " + ", ".join(f"{part}={n}" for part, n in sorted(self.counts.items()))
        )
        if self.temporal is not None:
            header.append(
                f"# temporal: unit={self.temporal.time_unit}, "
                f"label_horizon={self.temporal.label_horizon:g}, "
                f"embargo={self.temporal.embargo:g}"
            )
            for part, spans in sorted(self.temporal.spans.items()):
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
        return cls.model_validate(data)
