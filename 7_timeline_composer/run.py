from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.project_paths import default_output_dir
from clone_narration_video.utils.shot_breakdown import build_shot_breakdown


def _round(value: float) -> float:
    return round(float(value), 3)


def _duration(start: Any, end: Any) -> float:
    try:
        return max(0.0, float(end) - float(start))
    except Exception:
        return 0.0


def _text_units(text: str) -> int:
    cleaned = re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]【】]", "", text or "")
    return max(1, len(cleaned))


def estimate_tts_duration(text: str, chars_per_second: float, min_duration: float) -> float:
    cps = max(0.1, chars_per_second)
    return _round(max(min_duration, _text_units(text) / cps))


def _movie_shot_rows(movie_shots_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not movie_shots_data:
        return []
    rows = movie_shots_data.get("movie_shots") or []
    if not isinstance(rows, list):
        return []
    return sorted(rows, key=lambda x: float(x.get("start") or 0.0))


def _clip(
    clip_index: int,
    start: float,
    end: float,
    source: str,
    *,
    source_ref_shot_id: str | None = None,
    movie_shot_ids: list[str] | None = None,
    allocation: str,
) -> dict[str, Any]:
    return {
        "clip_id": f"clip_{clip_index:03d}",
        "movie_start": _round(start),
        "movie_end": _round(end),
        "duration": _round(_duration(start, end)),
        "source": source,
        "source_ref_shot_id": source_ref_shot_id,
        "movie_shot_ids": movie_shot_ids or [],
        "allocation": allocation,
    }


def _base_ranges(item: dict[str, Any]) -> list[dict[str, Any]]:
    ranges = []
    for row in item.get("movie_time_ranges") or []:
        dur = _duration(row.get("start"), row.get("end"))
        if dur <= 0:
            continue
        ranges.append(row)
    return ranges


def _used_movie_shot_ids(ranges: list[dict[str, Any]]) -> set[str]:
    used: set[str] = set()
    for row in ranges:
        for shot_id in row.get("movie_shot_ids") or []:
            used.add(str(shot_id))
    return used


def _extend_with_adjacent_shots(
    clips: list[dict[str, Any]],
    ranges: list[dict[str, Any]],
    movie_shots: list[dict[str, Any]],
    remaining: float,
    source: str,
) -> tuple[float, bool]:
    if remaining <= 0 or not ranges or not movie_shots:
        return remaining, False
    used = _used_movie_shot_ids(ranges)
    last_end = max(float(row.get("end") or 0.0) for row in ranges)
    extended = False
    for shot in movie_shots:
        if remaining <= 0:
            break
        shot_id = str(shot.get("movie_shot_id") or "")
        start = float(shot.get("start") or 0.0)
        end = float(shot.get("end") or 0.0)
        if shot_id in used or end <= start or start < last_end - 0.05:
            continue
        take = min(end - start, remaining)
        clips.append(
            _clip(
                len(clips) + 1,
                start,
                start + take,
                source,
                movie_shot_ids=[shot_id] if shot_id else [],
                allocation="adjacent_extension",
            )
        )
        remaining = _round(remaining - take)
        extended = True
    return remaining, extended


def allocate_video_clips(
    item: dict[str, Any],
    *,
    tts_duration: float,
    source: str,
    movie_shots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    ranges = _base_ranges(item)
    if not ranges:
        return [], "missing_visual_ranges"

    available = sum(_duration(row.get("start"), row.get("end")) for row in ranges)
    clips: list[dict[str, Any]] = []

    if tts_duration <= available:
        ratio = tts_duration / available if available else 0.0
        allocated = 0.0
        for idx, row in enumerate(ranges, start=1):
            start = float(row.get("start") or 0.0)
            original_duration = _duration(row.get("start"), row.get("end"))
            if idx == len(ranges):
                take = max(0.0, tts_duration - allocated)
            else:
                take = original_duration * ratio
            if take <= 0.03:
                continue
            clips.append(
                _clip(
                    len(clips) + 1,
                    start,
                    start + take,
                    source,
                    source_ref_shot_id=row.get("source_ref_shot_id"),
                    movie_shot_ids=[str(x) for x in row.get("movie_shot_ids") or []],
                    allocation="trimmed" if ratio < 0.98 else "original",
                )
            )
            allocated += take
        return clips, "trimmed_to_tts" if tts_duration < available * 0.98 else "kept_original"

    for row in ranges:
        clips.append(
            _clip(
                len(clips) + 1,
                float(row.get("start") or 0.0),
                float(row.get("end") or 0.0),
                source,
                source_ref_shot_id=row.get("source_ref_shot_id"),
                movie_shot_ids=[str(x) for x in row.get("movie_shot_ids") or []],
                allocation="original",
            )
        )

    remaining = _round(tts_duration - available)
    remaining, used_adjacent = _extend_with_adjacent_shots(clips, ranges, movie_shots, remaining, source)
    if remaining > 0.03 and clips:
        clips[-1]["movie_end"] = _round(float(clips[-1]["movie_end"]) + remaining)
        clips[-1]["duration"] = _round(float(clips[-1]["duration"]) + remaining)
        clips[-1]["allocation"] = "synthetic_extension"
        return clips, "extended_last_clip"
    return clips, "extended_with_adjacent_shots" if used_adjacent else "kept_original"


def _confidence(item: dict[str, Any], clips: list[dict[str, Any]], allocation_status: str) -> str:
    if not clips:
        return "low"
    if allocation_status in {"missing_visual_ranges", "extended_last_clip"}:
        return "low"
    range_conf = [str(x.get("confidence") or "low") for x in item.get("movie_time_ranges") or []]
    if any(x == "low" for x in range_conf):
        return "medium"
    if str(item.get("rewrite_status")) != "ai_rewritten":
        return "medium"
    return "high"


def compose_timeline(
    rewritten_script: list[dict[str, Any]],
    *,
    source: str,
    movie_shots_data: dict[str, Any] | None,
    chars_per_second: float,
    min_duration: float,
) -> dict[str, Any]:
    movie_shots = _movie_shot_rows(movie_shots_data)
    final_timeline = []
    cursor = 0.0
    for idx, item in enumerate(rewritten_script, start=1):
        narration = str(item.get("new_text") or item.get("old_text") or "")
        tts_duration = estimate_tts_duration(narration, chars_per_second, min_duration)
        clips, allocation_status = allocate_video_clips(
            item,
            tts_duration=tts_duration,
            source=source,
            movie_shots=movie_shots,
        )
        ref_range = item.get("ref_time_range") or {}
        ref_shot_ids = [
            str(row.get("source_ref_shot_id"))
            for row in item.get("movie_time_ranges") or []
            if row.get("source_ref_shot_id")
        ]
        final_timeline.append(
            {
                "item_id": f"item_{idx:03d}",
                "segment_id": item.get("segment_id"),
                "narration": narration,
                "tts_duration": tts_duration,
                "timeline_start": _round(cursor),
                "timeline_end": _round(cursor + tts_duration),
                "video_clips": clips,
                "ref_source": {
                    "ref_start": ref_range.get("start"),
                    "ref_end": ref_range.get("end"),
                    "ref_shot_ids": ref_shot_ids,
                },
                "confidence": _confidence(item, clips, allocation_status),
                "allocation_status": allocation_status,
            }
        )
        cursor = _round(cursor + tts_duration)
    return {
        "final_timeline": final_timeline,
        "timeline_backend": {
            "tts_duration_estimator": "char_count",
            "chars_per_second": chars_per_second,
            "min_duration": min_duration,
            "movie_shot_extension": bool(movie_shots),
        },
    }


def _merge_with_script_mapping(
    rewritten_rows: list[dict[str, Any]],
    script_mapping_data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not script_mapping_data:
        return rewritten_rows
    mapping = {
        str(row.get("segment_id")): row
        for row in script_mapping_data.get("script_mapping") or []
        if row.get("segment_id")
    }
    merged = []
    for row in rewritten_rows:
        seg_id = str(row.get("segment_id") or "")
        base = dict(mapping.get(seg_id) or {})
        base.update(row)
        if not base.get("movie_time_ranges") and mapping.get(seg_id):
            base["movie_time_ranges"] = mapping[seg_id].get("movie_time_ranges") or []
        merged.append(base)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="生成新视频脚本时间轴")
    parser.add_argument("--rewritten-script", required=True, help="第 6 步输出的 rewritten_script.json")
    parser.add_argument("--script-mapping", help="可选：第 5 步输出，用于补齐画面绑定信息")
    parser.add_argument("--movie-shots", help="可选：第 3 步输出，用于文案变长时扩展相邻镜头")
    parser.add_argument("--movie-source", default="movie.mp4")
    parser.add_argument("--output-dir", default=str(default_output_dir("7_timeline_composer")))
    parser.add_argument("--chars-per-second", type=float, default=4.2)
    parser.add_argument("--min-duration", type=float, default=1.2)
    parser.add_argument("--ref-analysis", help="可选：第 1 步 ref_analysis.json")
    parser.add_argument("--output-root", help="可选：outputs 根目录，用于自动查找流水线数据")
    args = parser.parse_args()

    rewritten_data = read_json(args.rewritten_script)
    rows = rewritten_data.get("rewritten_script") or []
    if not isinstance(rows, list):
        raise SystemExit("rewritten_script.json 缺少 rewritten_script 数组")

    mapping_data = read_json(args.script_mapping) if args.script_mapping else None
    movie_shots_data = read_json(args.movie_shots) if args.movie_shots else None
    rows = _merge_with_script_mapping(rows, mapping_data)
    result = compose_timeline(
        rows,
        source=args.movie_source,
        movie_shots_data=movie_shots_data,
        chars_per_second=args.chars_per_second,
        min_duration=args.min_duration,
    )
    out = write_json(Path(args.output_dir) / "final_timeline.json", result)
    print(out)

    output_root = Path(args.output_root).resolve() if args.output_root else Path(args.output_dir).resolve().parent
    ref_analysis_data = read_json(args.ref_analysis) if args.ref_analysis else None
    shot_breakdown = build_shot_breakdown(
        result.get("final_timeline") or [],
        ref_analysis=ref_analysis_data,
        script_mapping=mapping_data,
        output_root=output_root,
    )
    breakdown_out = write_json(Path(args.output_dir) / "shot_breakdown.json", shot_breakdown)
    print(breakdown_out)


if __name__ == "__main__":
    main()
