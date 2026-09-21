"""A replayable split family with generation provenance and a content digest."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from dsio.contracts import DsioModel, sha256_of
from dsio.data.splits.models.fold import SplitFold

SCHEMA = "dsio.split/3"
STABLE_ALGORITHMS = frozenset(
    {
        "group_kfold",
        "group_shuffle",
        "leave_one_group_out",
        "purged_walk_forward",
        "stratified_group_kfold",
    }
)


class SplitError(ValueError):
    """Raised when a split is malformed, or contradicts the store it names."""


class SplitFile(DsioModel):
    """Every fold and all provenance for one governed split family."""

    schema_version: str = SCHEMA
    store: str
    store_manifest_sha256: str | None = None
    examples_derivation: str | None = Field(
        default=None,
        description="The Examples derivation this split was computed against.",
    )
    group_key: str = "group"
    name: str
    algorithm: str = "manual"
    algorithm_version: str = "1"
    parameters: dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(default=0, strict=True)
    dependencies: dict[str, str] = Field(default_factory=dict)
    validations: tuple[str, ...] = ()
    coverage: Literal["total", "partial"] = "total"
    required_roles: tuple[str, ...] = ("train", "test")
    notes: str | None = None
    folds: list[SplitFold] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_folds(self) -> SplitFile:
        if not self.folds:
            raise ValueError("a split must declare at least one fold")
        if (
            len(self.required_roles) != 2
            or len(set(self.required_roles)) != 2
            or any(not role for role in self.required_roles)
        ):
            raise ValueError("required_roles must contain two distinct, non-empty names")
        if self.algorithm != "manual" and self.algorithm not in STABLE_ALGORITHMS:
            raise ValueError(
                f"unknown governed split algorithm {self.algorithm!r}; admit it to DSIO first"
            )
        if self.algorithm != "manual" and any(not fold.assignments for fold in self.folds):
            raise ValueError("governed split folds must contain exact sample assignments")
        try:
            sha256_of(self.parameters)
        except ValueError as exc:
            raise ValueError(f"split parameters are not canonical: {exc}") from exc

        duplicate_indices = sorted(
            {index for index, count in Counter(f.index for f in self.folds).items() if count > 1}
        )
        if duplicate_indices:
            raise ValueError(f"fold indices must be unique; repeated: {duplicate_indices}")

        _assert_evaluation_disjoint_across_folds(self.name, self.folds, self.required_roles[1])
        return self

    @property
    def digest(self) -> str:
        """Content identity over every replay- and provenance-relevant field."""
        return sha256_of(self.model_dump(mode="json"))

    def fold(self, index: int) -> SplitFold:
        """Return the fold with the declared index, never its list position."""
        for candidate in self.folds:
            if candidate.index == index:
                return candidate
        raise SplitError(
            f"split {self.name!r} has no fold {index}; it defines folds "
            f"{sorted(f.index for f in self.folds)}"
        )

    def to_yaml(self) -> str:
        """Render the manifest with a concise human-readable provenance header."""
        fold_numbers = ", ".join(str(f.index) for f in self.folds)
        word = "fold" if len(self.folds) == 1 else "folds"
        header = [
            f"# dsio split: {_header(self.name)} ({word} {fold_numbers})",
            f"# store: {_header(self.store)}",
            f"# algorithm: {_header(self.algorithm)}/{_header(self.algorithm_version)}  "
            f"seed={self.seed}",
            f"# digest: {self.digest}",
            f"# group key: {_header(self.group_key)}  <- the leakage boundary",
        ]
        multi = len(self.folds) > 1
        for fold in self.folds:
            prefix = f"[{fold.index}] " if multi else ""
            header.append(
                f"# {prefix}counts: "
                + ", ".join(
                    f"{_header(part)}={count}" for part, count in sorted(fold.counts.items())
                )
            )
            if fold.temporal is not None:
                header.append(
                    f"# {prefix}temporal: unit={fold.temporal.time_unit}, "
                    f"label_horizon={fold.temporal.label_horizon:g}, "
                    f"embargo={fold.temporal.embargo:g}"
                )
                for part, spans in sorted(fold.temporal.spans.items()):
                    rendered = ", ".join(f"[{span.start:g}, {span.end:g})" for span in spans)
                    header.append(f"#   {_header(part)}: {rendered}")
        if self.notes:
            header.append(f"# {_header(self.notes)}")
        payload = self.model_dump(mode="json")
        payload["digest"] = self.digest
        return "\n".join(header) + "\n" + yaml.safe_dump(payload, sort_keys=True, width=100)

    def save(self, path: Path | str) -> None:
        from dsio.contracts import atomic_write

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, self.to_yaml().encode("utf-8"))

    @classmethod
    def load(cls, path: Path) -> SplitFile:
        """Parse a trusted manifest boundary and normalize failures to SplitError."""
        try:
            loaded = yaml.safe_load(Path(path).read_text())
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise SplitError(f"cannot read split manifest {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise SplitError(f"{path} must contain a YAML mapping")
        data: dict[str, Any] = loaded
        if data.get("schema_version") != SCHEMA:
            raise SplitError(
                f"{path} declares schema {data.get('schema_version')!r}, expected {SCHEMA!r}"
            )
        persisted_digest = data.pop("digest", None)
        if persisted_digest is None and data.get("algorithm", "manual") != "manual":
            raise SplitError(f"{path} has no content digest")
        try:
            manifest = cls.model_validate(data)
        except ValidationError as exc:
            messages = [str(error["msg"]).removeprefix("Value error, ") for error in exc.errors()]
            raise SplitError("; ".join(messages)) from exc
        if persisted_digest is not None and persisted_digest != manifest.digest:
            raise SplitError(
                f"{path} digest is {persisted_digest}, but its content hashes to "
                f"{manifest.digest}; the split manifest was modified"
            )
        return manifest


def _header(value: object) -> str:
    """Render arbitrary model text on exactly one YAML-comment line."""
    escaped: list[str] = []
    for character in str(value):
        codepoint = ord(character)
        if character == "\\":
            escaped.append("\\\\")
        elif character == "\r":
            escaped.append("\\r")
        elif character == "\n":
            escaped.append("\\n")
        elif codepoint < 32 or codepoint == 127 or character in {"\x85", "\u2028", "\u2029"}:
            escaped.append(f"\\u{codepoint:04x}")
        else:
            escaped.append(character)
    return "".join(escaped)


def _assert_evaluation_disjoint_across_folds(
    name: str,
    folds: list[SplitFold],
    evaluation_role: str,
) -> None:
    group_owners: dict[str, list[int]] = {}
    sample_owners: dict[str, list[int]] = {}
    for fold in folds:
        for group in fold.parts.get(evaluation_role, []):
            group_owners.setdefault(group, []).append(fold.index)
        for sample_id in fold.assignments.get(evaluation_role, []):
            sample_owners.setdefault(sample_id, []).append(fold.index)
    _raise_cross_fold_collision(name, evaluation_role, "groups", group_owners)
    _raise_cross_fold_collision(name, evaluation_role, "assignments", sample_owners)


def _raise_cross_fold_collision(
    name: str,
    role: str,
    value_name: str,
    owners: dict[str, list[int]],
) -> None:
    collisions = {value: indices for value, indices in owners.items() if len(indices) > 1}
    if not collisions:
        return
    detail = "; ".join(
        f"{value!r} in folds {indices}" for value, indices in sorted(collisions.items())[:5]
    )
    if value_name == "groups":
        offending = sorted({index for indices in collisions.values() for index in indices})
        raise ValueError(
            f"split {name!r} {role} parts are not disjoint across folds {offending}: {detail}"
        )
    raise ValueError(f"split {name!r} {role} {value_name} are not disjoint across folds: {detail}")
