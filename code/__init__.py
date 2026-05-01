"""Support triage agent package for HackerRank Orchestrate.

The repository's required package name, ``code``, shadows Python's standard
library module of the same name. Pytest imports ``pdb``, and ``pdb`` expects
stdlib ``code`` attributes to exist. Re-export those small compatibility
symbols so tooling keeps working while preserving the mandated package path.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path
from types import ModuleType


def _load_stdlib_code_module() -> ModuleType:
    """Load the standard-library code.py module under a private alias."""

    stdlib_code_path = Path(sysconfig.get_path("stdlib")) / "code.py"
    spec = importlib.util.spec_from_file_location("_stdlib_code", stdlib_code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load stdlib code module from {stdlib_code_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_code = _load_stdlib_code_module()

InteractiveInterpreter = _stdlib_code.InteractiveInterpreter
InteractiveConsole = _stdlib_code.InteractiveConsole
compile_command = _stdlib_code.compile_command
