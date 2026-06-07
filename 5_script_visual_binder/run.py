from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.movie_time_ranges import merge_overlapping_movie_ranges
from clone_narration_video.utils.progress import emit_progress
from clone_narration_video.utils.project_paths import default_output_dir

ACCEPTED_VISUAL_STATUSES = {"matched", "matched_low_confidence", "inferred_by_neighbors"}


def _role(index: int, total: int) -> str:
    if index == 0:
        return "hook"
    if total > 1 and index == total - 1:
        return "ending"
    return "narration"


def bind_script_visual(
    narration_segments: list[dict[str, Any]],
    ref_to_movie_timeline: list[dict[str, Any]],
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    by_ref = {str(x.get("ref_shot_id")): x for x in ref_to_movie_timeline if x.get("ref_shot_id")}
    script_mapping = []
    total = len(narration_segments)
    for idx, seg in enumerate(narration_segments):
        ranges = []
        for ref_id in seg.get("ref_shot_ids") or []:
            ref_key = str(ref_id)
            match = by_ref.get(ref_key)
            if not match or match.get("status") not in ACCEPTED_VISUAL_STATUSES:
                continue
            movie_shot_ids = [str(x) for x in match.get("movie_shot_ids") or []]
            ranges.append(
                {
                    "start": match.get("movie_start"),
                    "end": match.get("movie_end"),
                    "source_ref_shot_id": ref_key,
                    "movie_shot_ids": movie_shot_ids,
                    "confidence": match.get("confidence") or "low",
                    "match_score": match.get("match_score") or 0.0,
                }
            )
        ranges = merge_overlapping_movie_ranges(ranges)
        script_mapping.append(
            {
                "segment_id": seg.get("segment_id"),
                "old_text": seg.get("text") or "",
                "ref_time_range": {"start": seg.get("ref_start"), "end": seg.get("ref_end")},
                "movie_time_ranges": ranges,
                "text_role": seg.get("text_role") or _role(idx, total),
            }
        )
        current = idx + 1
        if progress_callback and (current == 1 or current == total or current % 10 == 0):
            progress_callback((current / max(1, total)) * 100.0, f"Bound script segments {current}/{total}")
    return {"script_mapping": script_mapping}


def main() -> None:
    parser = argparse.ArgumentParser(description="解说画面绑定模块")
    parser.add_argument("--narration-segments", required=True, help="narration_segments.json")
    parser.add_argument("--timeline", required=True, help="ref_to_movie_timeline.json")
    parser.add_argument("--output-dir", default=str(default_output_dir("5_script_visual_binder")))
    args = parser.parse_args()

    seg_data = read_json(args.narration_segments)
    timeline_data = read_json(args.timeline)
    result = bind_script_visual(
        seg_data.get("narration_segments") or [],
        timeline_data.get("ref_to_movie_timeline") or [],
        progress_callback=lambda percent, message: emit_progress("binder", percent, message),
    )
    out = write_json(Path(args.output_dir) / "script_mapping.json", result)
    print(out)


if __name__ == "__main__":
    main()

