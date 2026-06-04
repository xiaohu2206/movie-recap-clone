from __future__ import annotations

import sys
from pathlib import Path


def add_project_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    value = str(repo_root)
    if value not in sys.path:
        sys.path.insert(0, value)

