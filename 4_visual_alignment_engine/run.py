from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.project_paths import default_output_dir
from clone_narration_video.utils.visual_features import build_frame_feature, compare_features, first_keyframe


def _id(item: dict[str, Any], key: str) -> str:
    return str(item.get(key) or item.get("shot_id") or "")


def _confidence(score: float) -> str:
    if score >= 0.82:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def _continuity(prev: dict[str, Any] | None, candidate: dict[str, Any], ref_gap: float) -> float:
    if not prev:
        return 0.5
    prev_end = float(prev.get("movie_end") or 0.0)
    expected = prev_end + max(0.0, ref_gap)
    start = float(candidate.get("start") or 0.0)
    delta = abs(start - expected)
    return math.exp(-delta / 8.0)


def align_visual_timeline(
    ref_shots: list[dict[str, Any]],
    movie_shots: list[dict[str, Any]],
    *,
    top_n: int = 8,
    min_score: float = 0.35,
) -> dict[str, Any]:
    movie_features = []
    for shot in movie_shots:
        path = first_keyframe(shot)
        if path:
            movie_features.append((shot, build_frame_feature(path)))

    timeline: list[dict[str, Any]] = []
    prev_match: dict[str, Any] | None = None
    prev_ref_end: float | None = None
    for ref in ref_shots:
        ref_path = first_keyframe(ref)
        ref_feature = build_frame_feature(ref_path) if ref_path else {"ok": False}
        candidates = []
        for movie, movie_feature in movie_features:
            visual = compare_features(ref_feature, movie_feature)
            ref_gap = 0.0 if prev_ref_end is None else float(ref.get("start") or 0.0) - prev_ref_end
            cont = _continuity(prev_match, movie, ref_gap)
            final = visual["score"] * 0.7 + cont * 0.3
            candidates.append(
                {
                    "movie_shot_id": _id(movie, "movie_shot_id"),
                    "movie_start": float(movie.get("start") or 0.0),
                    "movie_end": float(movie.get("end") or 0.0),
                    "visual_score": visual["score"],
                    "continuity_score": round(cont, 4),
                    "final_score": round(final, 4),
                    "detail": visual,
                }
            )
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        best = candidates[0] if candidates else None
        visual_best = max(candidates, key=lambda x: x["visual_score"], default=None)
        if best and visual_best and prev_match:
            # If the visual winner is clearly better, keep it; otherwise let continuity repair the path.
            if visual_best["visual_score"] >= best["visual_score"] + 0.12:
                best = visual_best

        status = "matched" if best and best["visual_score"] >= min_score else "unmatched"
        item = {
            "ref_shot_id": _id(ref, "ref_shot_id"),
            "ref_start": float(ref.get("start") or 0.0),
            "ref_end": float(ref.get("end") or 0.0),
            "movie_start": best["movie_start"] if best else None,
            "movie_end": best["movie_end"] if best else None,
            "movie_shot_ids": [best["movie_shot_id"]] if best else [],
            "match_score": best["visual_score"] if best else 0.0,
            "final_score": best["final_score"] if best else 0.0,
            "match_type": "temporal_continuity" if best and best["continuity_score"] > best["visual_score"] else "visual_hash",
            "confidence": _confidence(float(best["visual_score"])) if best else "low",
            "status": status,
            "candidates": candidates[:top_n],
        }
        timeline.append(item)
        if status == "matched":
            prev_match = item
        prev_ref_end = float(ref.get("end") or 0.0)

    return {"ref_to_movie_timeline": timeline}


def main() -> None:
    parser = argparse.ArgumentParser(description="画面定位模块")
    parser.add_argument("--ref-analysis", required=True, help="参考解析 JSON")
    parser.add_argument("--movie-shots", required=True, help="原电影镜头 JSON")
    parser.add_argument("--output-dir", default=str(default_output_dir("4_visual_alignment_engine")))
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.35)
    args = parser.parse_args()

    ref_data = read_json(args.ref_analysis)
    movie_data = read_json(args.movie_shots)
    result = align_visual_timeline(
        ref_data.get("ref_shots") or [],
        movie_data.get("movie_shots") or [],
        top_n=args.top_n,
        min_score=args.min_score,
    )
    out = write_json(Path(args.output_dir) / "ref_to_movie_timeline.json", result)
    print(out)


if __name__ == "__main__":
    main()

