"""dsio — reproducible ML/DL experimentation spine."""

import importlib.metadata as _metadata

# pyproject.toml is the only place the version is written; provenance records the same
# installed-metadata value, so `dsio.__version__` can never disagree with run evidence.
try:
    __version__ = _metadata.version("dsio")
except _metadata.PackageNotFoundError as error:
    raise ImportError(
        "dsio is not installed; install the distribution (e.g. `uv sync`) instead of "
        "importing the source tree directly"
    ) from error
