"""Plain importable component configuration without orchestration schemas."""

from dsio.config.components import ComponentConfig, ComponentError, resolve_component

__all__ = ["ComponentConfig", "ComponentError", "resolve_component"]
