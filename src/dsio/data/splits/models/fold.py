"""One fold's groups, exact assignments, and optional temporal bounds."""

from __future__ import annotations

from collections import Counter

from pydantic import Field, model_validator

from dsio.contracts import DsioModel
from dsio.data.splits.temporal import TemporalBounds


class SplitFold(DsioModel):
    """One division of a corpus into named roles."""

    index: int
    counts: dict[str, int] = Field(default_factory=dict)
    parts: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Group IDs per part. Empty for a purely temporal fold.",
    )
    assignments: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Exact stable sample identities per named role.",
    )
    temporal: TemporalBounds | None = Field(
        default=None,
        description="Time spans per part, with purge and embargo. Composes with `parts`.",
    )

    @model_validator(mode="after")
    def _validate_parts(self) -> SplitFold:
        if not self.parts and not self.assignments and self.temporal is None:
            raise ValueError(
                f"fold {self.index} must define group parts, sample assignments, "
                "temporal bounds, or a combination"
            )

        for part, groups in self.parts.items():
            duplicates = [group for group, count in Counter(groups).items() if count > 1]
            if duplicates:
                raise ValueError(
                    f"fold {self.index} part {part!r} lists {len(duplicates)} group(s) "
                    f"more than once: {', '.join(sorted(duplicates)[:5])}"
                )

        _validate_disjoint_values(self.index, self.parts, "group", "part")
        _validate_disjoint_values(self.index, self.assignments, "sample", "role")
        return self

    @property
    def all_groups(self) -> set[str]:
        return {group for groups in self.parts.values() for group in groups}

    def part_of(self, group: str) -> str | None:
        for part, groups in self.parts.items():
            if group in groups:
                return part
        return None

    @property
    def all_assignments(self) -> set[str]:
        return {sample_id for values in self.assignments.values() for sample_id in values}


def _validate_disjoint_values(
    fold_index: int,
    roles: dict[str, list[str]],
    value_name: str,
    role_name: str,
) -> None:
    seen: dict[str, str] = {}
    for role, values in roles.items():
        duplicates = [value for value, count in Counter(values).items() if count > 1]
        if duplicates:
            raise ValueError(
                f"fold {fold_index} {role_name} {role!r} lists duplicate {value_name} "
                f"identities: {', '.join(sorted(duplicates)[:5])}"
            )
        for value in values:
            previous = seen.get(value)
            if previous is not None:
                if value_name == "group":
                    raise ValueError(
                        f"fold {fold_index} parts must be mutually disjoint; "
                        f"{value!r} in both {previous!r} and {role!r}"
                    )
                raise ValueError(
                    f"fold {fold_index} {value_name} {value!r} appears in more than one "
                    f"{role_name}: {previous!r} and {role!r}"
                )
            seen[value] = role
