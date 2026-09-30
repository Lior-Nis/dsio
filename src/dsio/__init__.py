"""dsio — reproducible ML/DL experimentation spine."""

from importlib.metadata import version

# pyproject.toml is the only place the version is written; provenance records the same
# installed-metadata value, so `dsio.__version__` can never disagree with run evidence.
__version__ = version("dsio")
