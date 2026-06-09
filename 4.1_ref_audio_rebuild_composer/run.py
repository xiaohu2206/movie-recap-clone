from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.progress import emit_progress
from clone_narration_video.utils.project_paths import default_output_dir


ACCEPTED_VISUAL_STATUSES = {"matched", "matched_low_confidence", "inferred_by_neighbors"}


def _round(value: float) -> float:
    return round(float(value), 3)


def _duration(start: Any, end: Any) -> float:
    try:
        return _round(max(0.0, float(end) - float(start)))
    except Exception:
        return 0.0


def _ref_shots(ref_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = ref_analysis.get("ref_shots") or []
    if not isinstance(rows, list):
        return []
    return sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: float(row.get("start") or 0.0),
    )


def _timeline_rows(timeline_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = timeline_data.get("ref_to_movie_timeline") or []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _match_by_ref(timeline_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("ref_shot_id")): row
        for row in _timeline_rows(timeline_data)
        if row.get("ref_shot_id")
    }


def _is_accepted(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    if row.get("status") not in ACCEPTED_VISUAL_STATUSES:
        return False
    try:
        return float(row.get("movie_end")) > float(row.get("movie_start"))
    except Exception:
        return False


def _make_clip(
    *,
    source: str,
    start: float,
    target_duration: float,
    available_end: float | None,
    source_ref_shot_id: str,
    movie_shot_ids: list[str],
    fallback: bool,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    fit_mode = "fallback_use_ref_video" if fallback else "same_duration"
    end = start + target_duration

    if available_end is not None:
        available_duration = max(0.0, available_end - start)
        if available_duration > target_duration + 0.03:
            end = start + target_duration
            fit_mode = "cut_to_ref_duration"
        elif available_duration < target_duration - 0.03:
            end = start + target_duration
            fit_mode = "extend_movie_end"
            warnings.append("movie_clip_short")
        else:
            end = available_end

    return (
        {
            "clip_id": "clip_001",
            "source": source,
            "movie_start": _round(start),
            "movie_end": _round(end),
            "duration": _round(max(0.03, end - start)),
            "source_ref_shot_id": source_ref_shot_id,
            "movie_shot_ids": movie_shot_ids,
            "fit_mode": fit_mode,
            "allocation": fit_mode,
            "is_fallback": fallback,
        },
        warnings,
    )


def _base_items(
    ref_analysis: dict[str, Any],
    timeline_data: dict[str, Any],
    *,
    ref_video_path: str,
    movie_path: str,
) -> list[dict[str, Any]]:
    by_ref = _match_by_ref(timeline_data)
    items: list[dict[str, Any]] = []
    for index, ref in enumerate(_ref_shots(ref_analysis), start=1):
        ref_id = str(ref.get("ref_shot_id") or ref.get("shot_id") or f"ref_shot_{index:03d}")
        ref_start = float(ref.get("start") or 0.0)
        ref_end = float(ref.get("end") or ref_start)
        ref_duration = _duration(ref_start, ref_end)
        if ref_duration <= 0:
            continue

        match = by_ref.get(ref_id)
        warnings: list[str] = []
        if _is_accepted(match):
            movie_start = float(match.get("movie_start") or 0.0)
            movie_end = float(match.get("movie_end") or movie_start)
            clip, warnings = _make_clip(
                source=movie_path,
                start=movie_start,
                target_duration=ref_duration,
                available_end=movie_end,
                source_ref_shot_id=ref_id,
                movie_shot_ids=[str(x) for x in match.get("movie_shot_ids") or []],
                fallback=False,
            )
            status = "ready"
            reason = ""
            confidence = str(match.get("confidence") or "low")
            match_score = float(match.get("match_score") or 0.0)
            source_status = str(match.get("status") or "")
        else:
            clip, warnings = _make_clip(
                source=ref_video_path,
                start=ref_start,
                target_duration=ref_duration,
                available_end=ref_end,
                source_ref_shot_id=ref_id,
                movie_shot_ids=[],
                fallback=True,
            )
            status = "fallback"
            reason = str((match or {}).get("status") or "ref_shot_unmatched")
            confidence = "low"
            match_score = float((match or {}).get("match_score") or 0.0)
            source_status = str((match or {}).get("status") or "missing_alignment")

        items.append(
            {
                "item_id": f"rebuild_item_{len(items) + 1:03d}",
                "audio_type": "reference_audio",
                "type": "ref_audio_movie_video" if status == "ready" else "fallback_ref_video",
                "status": status,
                "reason": reason,
                "timeline_start": 0.0,
                "timeline_end": ref_duration,
                "tts_duration": ref_duration,
                "ref_time_range": {
                    "start": _round(ref_start),
                    "end": _round(ref_end),
                    "duration": ref_duration,
                },
                "external_audio": {
                    "source": "ref_video",
                    "path": ref_video_path,
                    "start": _round(ref_start),
                    "end": _round(ref_end),
                    "duration": ref_duration,
                },
                "video_clips": [clip],
                "source_ref_shot_ids": [ref_id],
                "confidence": confidence,
                "match_score": _round(match_score),
                "source_alignment_statuses": [source_status],
                "duration_warnings": warnings,
            }
        )
    return items


def _can_merge(
    current: dict[str, Any],
    next_item: dict[str, Any],
    *,
    max_ref_gap: float,
    max_movie_gap: float,
    min_movie_gap: float,
) -> bool:
    if current.get("status") != "ready" or next_item.get("status") != "ready":
        return False
    current_clips = current.get("video_clips") or []
    next_clips = next_item.get("video_clips") or []
    if not current_clips or not next_clips:
        return False
    current_clip = current_clips[-1]
    next_clip = next_clips[0]
    if current_clip.get("source") != next_clip.get("source"):
        return False

    current_ref = current.get("ref_time_range") or {}
    next_ref = next_item.get("ref_time_range") or {}
    ref_gap = float(next_ref.get("start") or 0.0) - float(current_ref.get("end") or 0.0)
    movie_gap = float(next_clip.get("movie_start") or 0.0) - float(current_clip.get("movie_end") or 0.0)
    return ref_gap <= max_ref_gap and min_movie_gap <= movie_gap <= max_movie_gap


def _merge_into(current: dict[str, Any], next_item: dict[str, Any]) -> None:
    current_ref = current.get("ref_time_range") or {}
    next_ref = next_item.get("ref_time_range") or {}
    current_audio = current.get("external_audio") or {}
    next_audio = next_item.get("external_audio") or {}

    current["ref_time_range"] = {
        "start": _round(float(current_ref.get("start") or 0.0)),
        "end": _round(float(next_ref.get("end") or 0.0)),
        "duration": _duration(current_ref.get("start"), next_ref.get("end")),
    }
    current["external_audio"] = {
        **current_audio,
        "start": _round(float(current_audio.get("start") or 0.0)),
        "end": _round(float(next_audio.get("end") or 0.0)),
        "duration": _duration(current_audio.get("start"), next_audio.get("end")),
    }
    for clip in next_item.get("video_clips") or []:
        merged = dict(clip)
        merged["clip_id"] = f"clip_{len(current.get('video_clips') or []) + 1:03d}"
        current.setdefault("video_clips", []).append(merged)
    current.setdefault("source_ref_shot_ids", []).extend(next_item.get("source_ref_shot_ids") or [])
    current.setdefault("source_alignment_statuses", []).extend(next_item.get("source_alignment_statuses") or [])
    current.setdefault("duration_warnings", []).extend(next_item.get("duration_warnings") or [])
    current["tts_duration"] = current["external_audio"]["duration"]
    current["timeline_end"] = current["timeline_start"] + current["tts_duration"]
    if current.get("confidence") != "low" and next_item.get("confidence") == "low":
        current["confidence"] = "low"
    elif current.get("confidence") == "high" and next_item.get("confidence") == "medium":
        current["confidence"] = "medium"


def merge_adjacent_items(
    items: list[dict[str, Any]],
    *,
    enabled: bool = True,
    max_ref_gap: float = 0.3,
    max_movie_gap: float = 1.0,
    min_movie_gap: float = -0.2,
) -> list[dict[str, Any]]:
    if not enabled or not items:
        return [dict(item) for item in items]

    merged: list[dict[str, Any]] = []
    for item in items:
        clone = dict(item)
        clone["video_clips"] = [dict(clip) for clip in item.get("video_clips") or []]
        clone["source_ref_shot_ids"] = list(item.get("source_ref_shot_ids") or [])
        clone["source_alignment_statuses"] = list(item.get("source_alignment_statuses") or [])
        clone["duration_warnings"] = list(item.get("duration_warnings") or [])
        if merged and _can_merge(
            merged[-1],
            clone,
            max_ref_gap=max_ref_gap,
            max_movie_gap=max_movie_gap,
            min_movie_gap=min_movie_gap,
        ):
            _merge_into(merged[-1], clone)
        else:
            merged.append(clone)
    return merged


def _apply_timeline_cursor(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cursor = 0.0
    for index, item in enumerate(items, start=1):
        duration = float((item.get("external_audio") or {}).get("duration") or item.get("tts_duration") or 0.0)
        item["item_id"] = f"rebuild_item_{index:03d}"
        item["timeline_start"] = _round(cursor)
        item["timeline_end"] = _round(cursor + duration)
        item["tts_duration"] = _round(duration)
        cursor += duration
    return items


def _quality_report(base_items: list[dict[str, Any]], final_items: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [item for item in base_items if item.get("status") == "ready"]
    fallback = [item for item in base_items if item.get("status") == "fallback"]
    return {
        "total_ref_shots": len(base_items),
        "ready_ref_shots": len(ready),
        "fallback_ref_shots": len(fallback),
        "rebuild_items": len(final_items),
        "merged_items": max(0, len(base_items) - len(final_items)),
        "duration_warning_count": sum(len(item.get("duration_warnings") or []) for item in base_items),
    }


def compose_ref_audio_rebuild_timeline(
    ref_analysis: dict[str, Any],
    movie_shots_data: dict[str, Any],
    timeline_data: dict[str, Any],
    *,
    ref_video_path: str,
    movie_path: str,
    merge_adjacent_clips: bool = True,
    max_ref_gap_for_merge: float = 0.3,
    max_movie_gap_for_merge: float = 1.0,
    min_movie_gap_for_merge: float = -0.2,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    if progress_callback:
        progress_callback(10.0, "Building ref-audio rebuild clips")
    base_items = _base_items(
        ref_analysis,
        timeline_data,
        ref_video_path=ref_video_path,
        movie_path=movie_path,
    )
    if progress_callback:
        progress_callback(55.0, "Merging adjacent rebuild clips")
    final_items = merge_adjacent_items(
        base_items,
        enabled=merge_adjacent_clips,
        max_ref_gap=max_ref_gap_for_merge,
        max_movie_gap=max_movie_gap_for_merge,
        min_movie_gap=min_movie_gap_for_merge,
    )
    final_items = _apply_timeline_cursor(final_items)
    if progress_callback:
        progress_callback(90.0, "Writing ref-audio rebuild timeline")

    return {
        "project_id": str(ref_analysis.get("ref_video_id") or "clone_rebuild"),
        "mode": "ref_audio_rebuild",
        "description": "Use reference video audio and replace visuals with aligned movie clips.",
        "sources": {
            "ref_video": {"path": ref_video_path},
            "movie": {"path": movie_path},
        },
        "settings": {
            "timeline_duration_source": "ref_audio",
            "default_unmatched_strategy": "use_ref_video",
            "merge_adjacent_clips": bool(merge_adjacent_clips),
            "max_ref_gap_for_merge": float(max_ref_gap_for_merge),
            "max_movie_gap_for_merge": float(max_movie_gap_for_merge),
            "min_movie_gap_for_merge": float(min_movie_gap_for_merge),
        },
        "final_timeline": final_items,
        "quality_report": _quality_report(base_items, final_items),
        "metadata": {
            "input_ref_shots": len(_ref_shots(ref_analysis)),
            "input_alignment_rows": len(_timeline_rows(timeline_data)),
            "movie_shots": len(movie_shots_data.get("movie_shots") or []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="参考音频画面重组时间线生成模块")
    parser.add_argument("--ref-analysis", required=True, help="第 1 步 ref_analysis.json")
    parser.add_argument("--movie-shots", required=True, help="第 3 步 movie_shots.json")
    parser.add_argument("--timeline", required=True, help="第 4 步 ref_to_movie_timeline.json")
    parser.add_argument("--ref-video-path", required=True)
    parser.add_argument("--movie-path", required=True)
    parser.add_argument("--output-dir", default=str(default_output_dir("4.1_ref_audio_rebuild_composer")))
    parser.add_argument("--disable-merge-adjacent", action="store_true")
    parser.add_argument("--max-ref-gap-for-merge", type=float, default=0.3)
    parser.add_argument("--max-movie-gap-for-merge", type=float, default=1.0)
    parser.add_argument("--min-movie-gap-for-merge", type=float, default=-0.2)
    args = parser.parse_args()

    result = compose_ref_audio_rebuild_timeline(
        read_json(args.ref_analysis),
        read_json(args.movie_shots),
        read_json(args.timeline),
        ref_video_path=args.ref_video_path,
        movie_path=args.movie_path,
        merge_adjacent_clips=not args.disable_merge_adjacent,
        max_ref_gap_for_merge=args.max_ref_gap_for_merge,
        max_movie_gap_for_merge=args.max_movie_gap_for_merge,
        min_movie_gap_for_merge=args.min_movie_gap_for_merge,
        progress_callback=lambda percent, message: emit_progress("ref_audio_rebuild", percent, message),
    )
    out = write_json(Path(args.output_dir) / "ref_audio_rebuild_timeline.json", result)
    emit_progress("ref_audio_rebuild", 100, "Ref-audio rebuild timeline complete")
    print(out)


if __name__ == "__main__":
    main()
