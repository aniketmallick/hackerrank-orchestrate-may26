"""Pytest bootstrap.

Ensures the parent directory of this `code/` package is on ``sys.path`` so
tests can ``from code.config import X`` regardless of where pytest is invoked
from. Lets graders unzip ``code/`` anywhere and run ``pytest -q`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _PACKAGE_DIR.parent

if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))
