"""Pytest bootstrap for the repo package named ``code``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def _is_repo_path(path_entry: str) -> bool:
    candidate = Path(path_entry or ".").resolve()
    return candidate == ROOT_DIR


original_sys_path = list(sys.path)
try:
    sys.path = [entry for entry in sys.path if not _is_repo_path(entry)]
    sys.modules.pop("code", None)
finally:
    sys.path = original_sys_path
    sys.modules.pop("code", None)
