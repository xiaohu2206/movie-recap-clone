from __future__ import annotations

from typing import Any


def _has_movie_subtitles(row: dict[str, Any]) -> bool:
    subtitles = row.get("movie_subtitles") or []
    if not isinstance(subtitles, list):
        return bool(str(subtitles).strip())
    for sub in subtitles:
        if isinstance(sub, dict) and str(sub.get("text") or "").strip():
            return True
        if not isinstance(sub, dict) and str(sub).strip():
            return True
    return False


def try_shortcut(segment: dict[str, Any]) -> dict[str, Any] | None:
    """Only safe rule: no movie subtitle text anywhere means pure narration."""
    ranges = segment.get("movie_time_ranges") or []
    if all(not _has_movie_subtitles(row) for row in ranges):
        return {"audio_pattern": "all_narration", "split_clip_index": None, "source": "rule"}
    return None
