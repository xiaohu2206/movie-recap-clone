from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.progress import emit_progress
from clone_narration_video.utils.project_paths import default_output_dir, relpath
from clone_narration_video.utils.subtitle_tools import extract_srt_with_bcut, parse_srt, write_srt


def _duration(start: Any, end: Any) -> float:
    try:
        return max(0.0, float(end) - float(start))
    except Exception:
        return 0.0


def _round(value: float) -> float:
    return round(float(value), 3)


def _normalize_subtitle(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": _round(float(entry.get("start") or 0.0)),
        "end": _round(float(entry.get("end") or 0.0)),
        "text": str(entry.get("text") or "").strip(),
    }


def _overlaps(start: float, end: float, sub: dict[str, Any]) -> bool:
    return float(sub.get("start") or 0.0) < end and float(sub.get("end") or 0.0) > start


def _subtitles_for_range(subtitles: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [_normalize_subtitle(sub) for sub in subtitles if _overlaps(start, end, sub) and str(sub.get("text") or "").strip()]


def _collect_windows(script_mapping: list[dict[str, Any]], padding: float) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    for item in script_mapping:
        for row in item.get("movie_time_ranges") or []:
            dur = _duration(row.get("start"), row.get("end"))
            if dur <= 0:
                continue
            windows.append(
                {
                    "start": max(0.0, float(row.get("start") or 0.0) - padding),
                    "end": float(row.get("end") or 0.0) + padding,
                }
            )
    windows.sort(key=lambda x: x["start"])
    merged: list[dict[str, float]] = []
    for window in windows:
        if not merged or window["start"] > merged[-1]["end"]:
            merged.append(dict(window))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], window["end"])
    return [{"start": _round(x["start"]), "end": _round(x["end"])} for x in merged]


def _filter_subtitles_for_windows(subtitles: list[dict[str, Any]], windows: list[dict[str, float]]) -> list[dict[str, Any]]:
    if not windows:
        return [_normalize_subtitle(sub) for sub in subtitles if str(sub.get("text") or "").strip()]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, float, str]] = set()
    for sub in subtitles:
        text = str(sub.get("text") or "").strip()
        if not text:
            continue
        start = float(sub.get("start") or 0.0)
        end = float(sub.get("end") or 0.0)
        if not any(start < window["end"] and end > window["start"] for window in windows):
            continue
        normalized = _normalize_subtitle(sub)
        key = (normalized["start"], normalized["end"], normalized["text"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(normalized)
    return selected


def _load_or_extract_subtitles(
    *,
    movie_path: str,
    subtitle_srt: str,
    output_srt: Path,
    work_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    if subtitle_srt:
        shutil.copyfile(subtitle_srt, output_srt)
        return parse_srt(output_srt), "srt_input"
    extract_srt_with_bcut(movie_path, output_srt, work_dir)
    return parse_srt(output_srt), "bcut"


def fill_movie_subtitles(
    script_mapping: list[dict[str, Any]],
    *,
    movie_path: str,
    subtitle_srt: str = "",
    output_dir: Path,
    padding: float = 0.3,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = _collect_windows(script_mapping, padding)
    if progress_callback:
        progress_callback(10.0, f"Collected subtitle windows {len(windows)}")

    movie_subtitle_path = output_dir / "movie_subtitle.srt"
    subtitles, provider = _load_or_extract_subtitles(
        movie_path=movie_path,
        subtitle_srt=subtitle_srt,
        output_srt=movie_subtitle_path,
        work_dir=output_dir / "work",
    )
    selected_subtitles = _filter_subtitles_for_windows(subtitles, windows)
    write_srt(movie_subtitle_path, selected_subtitles)
    if progress_callback:
        progress_callback(45.0, f"Loaded movie subtitles {len(selected_subtitles)}")

    filled: list[dict[str, Any]] = []
    total = len(script_mapping)
    for seg_idx, item in enumerate(script_mapping, start=1):
        new_item = dict(item)
        ranges: list[dict[str, Any]] = []
        for clip_index, row in enumerate(item.get("movie_time_ranges") or []):
            new_row = dict(row)
            new_row["clip_index"] = int(new_row.get("clip_index", clip_index))
            start = float(new_row.get("start") or 0.0)
            end = float(new_row.get("end") or 0.0)
            new_row["movie_subtitles"] = _subtitles_for_range(selected_subtitles, start, end)
            ranges.append(new_row)
        new_item["movie_time_ranges"] = ranges
        filled.append(new_item)
        if progress_callback and (seg_idx == 1 or seg_idx == total or seg_idx % 10 == 0):
            progress_callback(45.0 + (seg_idx / max(1, total)) * 55.0, f"Filled subtitle ranges {seg_idx}/{total}")

    return {
        "script_mapping": filled,
        "movie_subtitle_srt": relpath(movie_subtitle_path),
        "subtitle_backend": {
            "provider": provider,
            "ranges": len(windows),
            "subtitles": len(selected_subtitles),
            "padding": padding,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="原片字幕补充模块")
    parser.add_argument("--script-mapping", required=True, help="第 5 步输出的 script_mapping.json")
    parser.add_argument("--movie-path", required=True, help="原片视频路径")
    parser.add_argument("--movie-subtitle-srt", default="", help="可选：现成原片字幕 srt")
    parser.add_argument("--output-dir", default=str(default_output_dir("5.1_movie_subtitle_filler")))
    parser.add_argument("--padding", type=float, default=0.3)
    args = parser.parse_args()

    data = read_json(args.script_mapping)
    script_mapping = data.get("script_mapping") or []
    if not isinstance(script_mapping, list):
        raise SystemExit("script_mapping.json 缺少 script_mapping 数组")
    result = fill_movie_subtitles(
        script_mapping,
        movie_path=args.movie_path,
        subtitle_srt=args.movie_subtitle_srt,
        output_dir=Path(args.output_dir),
        padding=args.padding,
        progress_callback=lambda percent, message: emit_progress("subtitle", percent, message),
    )
    out = write_json(Path(args.output_dir) / "script_mapping_subtitled.json", result)
    print(out)


if __name__ == "__main__":
    main()
