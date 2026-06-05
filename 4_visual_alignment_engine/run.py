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
from clone_narration_video.utils.visual_features import build_shot_feature, compare_shot_features

from candidate_recall import build_candidates
from diagnostics import build_timeline_item, write_low_confidence_report
from path_solver import solve_global_path, solve_greedy_path

ALGORITHM_VERSION = "visual_alignment_v2"


def _id(item: dict[str, Any], key: str) -> str:
    return str(item.get(key) or item.get("shot_id") or "")


def _progress(
    progress_callback: Callable[[float, str], None] | None,
    percent: float,
    message: str,
) -> None:
    if progress_callback:
        progress_callback(percent, message)


def _load_manual_overrides(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    data = read_json(path)
    rows = data.get("overrides") if isinstance(data, dict) else data
    overrides: dict[str, str] = {}
    if not isinstance(rows, list):
        return overrides
    for row in rows:
        if not isinstance(row, dict):
            continue
        ref_id = row.get("ref_shot_id")
        movie_id = row.get("movie_shot_id")
        if ref_id and movie_id:
            overrides[str(ref_id)] = str(movie_id)
    return overrides


def _manual_candidate(
    ref_feature: dict[str, Any],
    movie_features: list[dict[str, Any]],
    movie_shots: list[dict[str, Any]],
    movie_id_to_index: dict[str, int],
    movie_shot_id: str,
) -> dict[str, Any] | None:
    movie_index = movie_id_to_index.get(movie_shot_id)
    if movie_index is None:
        return None
    movie = movie_shots[movie_index]
    visual = compare_shot_features(ref_feature, movie_features[movie_index])
    return {
        "movie_index": movie_index,
        "movie_shot_id": _id(movie, "movie_shot_id"),
        "movie_start": float(movie.get("start") or 0.0),
        "movie_end": float(movie.get("end") or 0.0),
        "visual_score": float(visual["score"]),
        "recall_score": float(visual["lightweight_score"]),
        "final_score": float(visual["score"]),
        "path_score": float(visual["score"]),
        "visual_rank": None,
        "final_rank": 1,
        "detail": visual,
        "diagnostics": {
            "transition_score": 1.0,
            "continuity_score": 1.0,
            "time_delta": 0.0,
            "path_penalty": 0.0,
            "path_continuous": True,
            "boosted_by_continuity": False,
        },
    }


def _append_candidate_once(candidates: list[dict[str, Any]], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    movie_shot_id = str(candidate.get("movie_shot_id") or "")
    for index, existing in enumerate(candidates):
        if str(existing.get("movie_shot_id") or "") == movie_shot_id:
            candidates[index] = {**existing, **candidate}
            return candidates
    return [candidate, *candidates]


def _summary(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(timeline)
    matched = [row for row in timeline if row.get("status") in {"matched", "matched_low_confidence", "inferred_by_neighbors"}]
    high = [row for row in timeline if row.get("confidence") == "high"]
    low_or_review = [row for row in timeline if row.get("confidence") == "low" or row.get("status") != "matched"]
    backward_count = 0
    duplicate_count = 0
    seen: set[str] = set()
    prev_start: float | None = None
    for row in timeline:
        movie_start = row.get("movie_start")
        if movie_start is not None and prev_start is not None and float(movie_start) < prev_start - 0.5:
            backward_count += 1
        if movie_start is not None:
            prev_start = float(movie_start)
        for shot_id in row.get("movie_shot_ids") or []:
            shot_id = str(shot_id)
            if shot_id in seen:
                duplicate_count += 1
            seen.add(shot_id)
    return {
        "total_ref_shots": total,
        "matched_count": len(matched),
        "matched_rate": round(len(matched) / max(1, total), 4),
        "high_confidence_count": len(high),
        "high_confidence_rate": round(len(high) / max(1, total), 4),
        "manual_review_count": len(low_or_review),
        "timeline_backward_count": backward_count,
        "duplicate_movie_shot_count": duplicate_count,
    }


def align_visual_timeline(
    ref_shots: list[dict[str, Any]],
    movie_shots: list[dict[str, Any]],
    *,
    top_n: int = 8,
    min_score: float = 0.35,
    keyframes_per_shot: int = 3,
    recall_top_k: int = 80,
    rerank_top_k: int = 20,
    feature_mode: str = "classic",
    disable_global_path: bool = False,
    manual_overrides: dict[str, str] | None = None,
    diagnostics_dir: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    if feature_mode not in {"classic", "classic_clip"}:
        raise ValueError(f"不支持的 feature_mode: {feature_mode}")

    effective_feature_mode = "classic"
    embedding_status = "disabled"
    if feature_mode == "classic_clip":
        embedding_status = "not_configured_fallback_to_classic"

    movie_features: list[dict[str, Any]] = []
    total_movie = len(movie_shots)
    for idx, shot in enumerate(movie_shots, start=1):
        movie_features.append(
            build_shot_feature(shot, max_frames=keyframes_per_shot, feature_mode=effective_feature_mode)
        )
        if idx == 1 or idx == total_movie or idx % 20 == 0:
            _progress(
                progress_callback,
                5.0 + (idx / max(1, total_movie)) * 25.0,
                f"Indexed movie shot features {idx}/{total_movie}",
            )

    ref_features: list[dict[str, Any]] = []
    candidates_by_ref: list[list[dict[str, Any]]] = []
    total_ref = len(ref_shots)
    for ref_index, ref in enumerate(ref_shots, start=1):
        ref_feature = build_shot_feature(ref, max_frames=keyframes_per_shot, feature_mode=effective_feature_mode)
        ref_features.append(ref_feature)
        candidates_by_ref.append(
            build_candidates(
                ref_feature,
                movie_features,
                movie_shots,
                recall_top_k=recall_top_k,
                rerank_top_k=rerank_top_k,
            )
        )
        if ref_index == 1 or ref_index == total_ref or ref_index % 10 == 0:
            _progress(
                progress_callback,
                30.0 + (ref_index / max(1, total_ref)) * 55.0,
                f"Recalled and reranked candidates {ref_index}/{total_ref}",
            )

    _progress(progress_callback, 88.0, "Solving visual alignment path")
    if disable_global_path:
        chosen_path = solve_greedy_path(ref_shots, candidates_by_ref)
        match_type = "temporal_continuity"
    else:
        chosen_path = solve_global_path(ref_shots, candidates_by_ref)
        match_type = "global_path"

    movie_id_to_index = {_id(shot, "movie_shot_id"): idx for idx, shot in enumerate(movie_shots)}
    manual_overrides = manual_overrides or {}
    timeline: list[dict[str, Any]] = []
    for index, ref in enumerate(ref_shots):
        candidate = chosen_path[index] if index < len(chosen_path) else None
        candidates = candidates_by_ref[index] if index < len(candidates_by_ref) else []
        ref_id = _id(ref, "ref_shot_id")
        manual = False
        if ref_id in manual_overrides:
            override = _manual_candidate(
                ref_features[index],
                movie_features,
                movie_shots,
                movie_id_to_index,
                manual_overrides[ref_id],
            )
            if override:
                candidate = override
                candidates = _append_candidate_once(candidates, override)
                manual = True

        timeline.append(
            build_timeline_item(
                ref,
                candidate,
                candidates,
                top_n=top_n,
                min_score=min_score,
                match_type=match_type,
                manual_override=manual,
            )
        )

    _progress(progress_callback, 96.0, "Writing alignment diagnostics")
    if diagnostics_dir:
        write_low_confidence_report(diagnostics_dir, timeline)

    return {
        "ref_to_movie_timeline": timeline,
        "metadata": {
            "algorithm_version": ALGORITHM_VERSION,
            "feature_mode": effective_feature_mode,
            "requested_feature_mode": feature_mode,
            "embedding_status": embedding_status,
            "keyframes_per_shot": int(keyframes_per_shot),
            "recall_top_k": int(recall_top_k),
            "rerank_top_k": int(rerank_top_k),
            "output_top_n": int(top_n),
            "min_score": float(min_score),
            "global_path_enabled": not disable_global_path,
            "manual_override_count": len(manual_overrides),
            "summary": _summary(timeline),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="画面定位模块")
    parser.add_argument("--ref-analysis", required=True, help="参考解析 JSON")
    parser.add_argument("--movie-shots", required=True, help="原电影镜头 JSON")
    parser.add_argument("--output-dir", default=str(default_output_dir("4_visual_alignment_engine")))
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--keyframes-per-shot", type=int, default=3)
    parser.add_argument("--recall-top-k", type=int, default=80)
    parser.add_argument("--rerank-top-k", type=int, default=20)
    parser.add_argument("--feature-mode", choices=["classic", "classic_clip"], default="classic")
    parser.add_argument("--disable-global-path", action="store_true")
    parser.add_argument("--manual-overrides", help="人工覆写 JSON")
    parser.add_argument("--diagnostics-dir", help="诊断输出目录；默认写到 output-dir/diagnostics")
    args = parser.parse_args()

    ref_data = read_json(args.ref_analysis)
    movie_data = read_json(args.movie_shots)
    output_dir = Path(args.output_dir)
    diagnostics_dir = Path(args.diagnostics_dir) if args.diagnostics_dir else output_dir / "diagnostics"
    result = align_visual_timeline(
        ref_data.get("ref_shots") or [],
        movie_data.get("movie_shots") or [],
        top_n=args.top_n,
        min_score=args.min_score,
        keyframes_per_shot=args.keyframes_per_shot,
        recall_top_k=args.recall_top_k,
        rerank_top_k=args.rerank_top_k,
        feature_mode=args.feature_mode,
        disable_global_path=args.disable_global_path,
        manual_overrides=_load_manual_overrides(args.manual_overrides),
        diagnostics_dir=diagnostics_dir,
        progress_callback=lambda percent, message: emit_progress("alignment", percent, message),
    )
    out = write_json(output_dir / "ref_to_movie_timeline.json", result)
    emit_progress("alignment", 100, "Visual alignment complete")
    print(out)


if __name__ == "__main__":
    main()
