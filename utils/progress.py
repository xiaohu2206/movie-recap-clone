from __future__ import annotations

import json
from typing import Any


def emit_progress(
    stage: str,
    percent: float,
    message: str,
    *,
    current: int | None = None,
    total: int | None = None,
    phase: str | None = None,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "percent": max(0.0, min(100.0, round(float(percent), 1))),
        "message": message,
    }
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)
    if phase:
        payload["phase"] = phase
    payload.update(extra)
    print(f"[progress] {json.dumps(payload, ensure_ascii=False)}", flush=True)
