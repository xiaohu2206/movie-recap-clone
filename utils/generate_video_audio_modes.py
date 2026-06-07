from __future__ import annotations

from typing import Any


VOICEOVER = "voiceover"
ORIGINAL = "original"


def round_seconds(value: Any) -> float:
    return round(float(value or 0.0), 3)


def item_key(item: dict[str, Any], fallback: str) -> str:
    return str(item.get("item_id") or item.get("segment_id") or fallback)


def item_duration_from_clips(item: dict[str, Any]) -> float:
    total = 0.0
    for clip in item.get("video_clips") or []:
        if not isinstance(clip, dict):
            continue
        try:
            start = float(clip.get("movie_start") or 0.0)
            end = float(clip.get("movie_end") or 0.0)
        except Exception:
            continue
        total += max(0.0, end - start)
    return round_seconds(total)


def audio_mode_for_item(item: dict[str, Any]) -> str:
    mode = str(item.get("audio_mode") or "").strip().lower()
    if mode == ORIGINAL:
        return ORIGINAL
    return VOICEOVER


def is_original_audio_item(item: dict[str, Any]) -> bool:
    return audio_mode_for_item(item) == ORIGINAL


def original_audio_result(item: dict[str, Any]) -> dict[str, Any]:
    duration = max(
        float(item.get("tts_duration") or 0.0),
        item_duration_from_clips(item),
        0.0,
    )
    return {
        "path": "",
        "duration": round_seconds(duration),
        "original_audio": True,
        "silent": False,
    }
