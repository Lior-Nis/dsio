"""The run configuration root and the task-config base every runner extends.

``RunConfig.task`` is annotated as the base :class:`TaskConfig` but resolved through the
task registry on the way in and serialized as its concrete subclass on the way out. That
keeps the spine open — a project can register a task dsio has never heard of — while the
recorded YAML still round-trips back into the exact same typed object.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, SerializeAsAny, field_validator

from dsio.config.registry import Registry
from dsio.contracts import DsioModel, sha256_of, short_digest


class TaskConfig(DsioModel):
    """Base for a unit of work a runner knows how to execute.

    ``kind`` is the registry key. Subclasses set it as a literal default so a config
    never has to repeat it, and so the recorded YAML is self-describing.
    """

    kind: str


TASKS: Registry[type[TaskConfig]] = Registry("task")


class RunConfig(DsioModel):
    """The complete, validated description of one run.

    Everything needed to reproduce a run is reachable from here. Anything not reachable
    from here is, by definition, not reproducible.
    """

    name: str = Field(min_length=1, description="Human label; grouping key across seeds.")
    seed: int = Field(default=42, ge=0)
    tags: tuple[str, ...] = ()
    task: SerializeAsAny[TaskConfig]

    @field_validator("task", mode="before")
    @classmethod
    def _resolve_task(cls, value: Any) -> Any:
        """Rebuild a serialized task into its registered concrete subclass."""
        if isinstance(value, TaskConfig):
            return value
        if isinstance(value, dict):
            kind = value.get("kind")
            if kind is None:
                raise ValueError(
                    "task mapping has no 'kind'; cannot tell which task type to build"
                )
            if not isinstance(kind, str):
                raise ValueError(f"task 'kind' must be a string, got {type(kind).__name__}")
            return TASKS.get(kind).model_validate(value)
        raise ValueError(f"task must be a TaskConfig or mapping, got {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """Return the fully-resolved config as plain data, ready to record or hash."""
        return self.model_dump(mode="json")

    @property
    def config_hash(self) -> str:
        """Full sha256 over the resolved config. The run's content identity."""
        return sha256_of(self.to_dict())

    @property
    def short_hash(self) -> str:
        """Short digest for directory names and log lines."""
        return short_digest(self.to_dict())
