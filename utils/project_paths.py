from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
MODEL_DIR = PROJECT_ROOT / "model" / "transnetv2-weights"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_output_dir(module_name: str) -> Path:
    return ensure_dir(OUTPUT_ROOT / module_name)


def relpath(path: str | Path, base: str | Path = PROJECT_ROOT) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(Path(base).resolve()))
    except ValueError:
        return str(p)


def resolve_existing_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for candidate in (p, PROJECT_ROOT / p, REPO_ROOT / p):
        if candidate.exists():
            return candidate
    return p
