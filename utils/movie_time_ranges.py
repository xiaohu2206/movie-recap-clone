from __future__ import annotations

from typing import Any


def _round(value: float) -> float:
    return round(float(value), 3)


def ranges_overlap(a: dict[str, Any], b: dict[str, Any], *, gap: float = 0.05) -> bool:
    a0, a1 = float(a.get("start") or 0.0), float(a.get("end") or 0.0)
    b0, b1 = float(b.get("start") or 0.0), float(b.get("end") or 0.0)
    return a0 <= b1 + gap and b0 <= a1 + gap


def _union_shot_ids(prev: dict[str, Any], current: dict[str, Any]) -> list[str]:
    shot_ids: list[str] = [str(x) for x in prev.get("movie_shot_ids") or [] if str(x)]
    for shot_id in current.get("movie_shot_ids") or []:
        sid = str(shot_id)
        if sid and sid not in shot_ids:
            shot_ids.append(sid)
    return shot_ids


def merge_overlapping_movie_ranges(ranges: list[dict[str, Any]], *, gap: float = 0.05) -> list[dict[str, Any]]:
    """合并同一段内 movie 时间重叠的区间，避免重复播放同一画面。

    多个参考镜头可能因画面定位误差映射到重叠的原片时间段；此处按时间重叠合并，
    保留 match_score 更高的 source_ref_shot_id，并合并 movie_shot_ids。
    """
    if len(ranges) <= 1:
        return [dict(r) for r in ranges]

    sorted_ranges = sorted(ranges, key=lambda row: float(row.get("start") or 0.0))
    merged: list[dict[str, Any]] = []
    for row in sorted_ranges:
        current = dict(row)
        if not merged:
            merged.append(current)
            continue
        prev = merged[-1]
        if not ranges_overlap(prev, current, gap=gap):
            merged.append(current)
            continue
        prev["start"] = _round(min(float(prev.get("start") or 0.0), float(current.get("start") or 0.0)))
        prev["end"] = _round(max(float(prev.get("end") or 0.0), float(current.get("end") or 0.0)))
        prev["movie_shot_ids"] = _union_shot_ids(prev, current)
        if float(current.get("match_score") or 0.0) > float(prev.get("match_score") or 0.0):
            prev["source_ref_shot_id"] = current.get("source_ref_shot_id")
            prev["match_score"] = current.get("match_score")
            prev["confidence"] = current.get("confidence")
    return merged
