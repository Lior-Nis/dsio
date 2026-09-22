"""Import and expression-name analysis for static admission."""

from dsio.experimental.admission.syntax.imports.dynamic import (
    dynamic_import_reference,
    is_dynamic_import_api,
)
from dsio.experimental.admission.syntax.imports.names import expression_name, fold_string
from dsio.experimental.admission.syntax.imports.resolution import (
    aliases,
    aliases_at,
    assigned_aliases,
    imports,
)

__all__ = [
    "aliases",
    "aliases_at",
    "assigned_aliases",
    "dynamic_import_reference",
    "expression_name",
    "fold_string",
    "imports",
    "is_dynamic_import_api",
]
