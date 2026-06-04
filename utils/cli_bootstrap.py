from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_NAME = "clone_narration_video"


def _ensure_package_registered() -> None:
    if _PACKAGE_NAME in sys.modules:
        return
    init_path = _PROJECT_ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME,
        init_path,
        submodule_search_locations=[str(_PROJECT_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载包 {_PACKAGE_NAME}: {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    spec.loader.exec_module(module)


def add_project_to_syspath() -> None:
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _ensure_package_registered()
