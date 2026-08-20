import importlib
import subprocess
import sys

from dsio.config.presets import _BUILTIN_PRESET_MODULES, load_preset_modules


def test_every_builtin_preset_module_is_importable():
    """A stale entry here would rot exactly as a stale runner entry did."""
    for name in _BUILTIN_PRESET_MODULES:
        importlib.import_module(name)


def test_discovery_registers_a_preset_in_a_clean_interpreter():
    """In-process this is unfalsifiable: conftest's autouse fixture has already
    registered one. Only a fresh interpreter proves discovery itself works."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from dsio.config.presets import PRESETS, load_preset_modules;"
            " load_preset_modules();"
            " assert PRESETS.names(), 'discovery registered nothing'",
        ],
        check=True,
    )


def test_discovery_is_idempotent():
    assert load_preset_modules() == load_preset_modules()
