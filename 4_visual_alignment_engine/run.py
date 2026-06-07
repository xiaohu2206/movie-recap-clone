from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.progress import emit_progress
from clone_narration_video.utils.project_paths import default_output_dir
from clone_narration_video.utils.video_tools import cut_clip
from clone_narration_video.utils.visual_features import build_shot_feature, compare_shot_features

from candidate_recall import build_candidates
from diagnostics import build_timeline_item, write_low_confidence_report
from path_solver import solve_greedy_path, solve_segmented_global_path
from refinement import refine_candidates_for_ref

ALGORITHM_VERSION = "visual_alignment_v3"
# 大写配置-输出分割后的镜头
EXPORT_MATCHED_SHOT_CLIPS = True              # 默认不开启；开启后按 ref 镜头建子文件夹输出配对片段
MATCHED_SHOT_CLIPS_DIRNAME = "matched_shot_clips"  # 独立文件夹；每个 ref 镜头一个子文件夹
MATCHED_SHOT_CLIPS_RATIO = 1                 # 默认只输出前 20% 的镜头

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
        "refinement": {"enabled": False, "mode": "manual_override"},
        "diagnostics": {
            "transition_score": 1.0,
            "continuity_score": 1.0,
            "time_delta": 0.0,
            "path_penalty": 0.0,
            "path_continuous": True,
            "boosted_by_continuity": False,
            "path": {
                "anchor": True,
                "segment_index": None,
                "jump_allowed": True,
                "skip_state": False,
            },
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


def export_matched_shot_clips(
    timeline: list[dict[str, Any]],
    ref_video_path: str | Path | None,
    movie_video_path: str | Path | None,
    out_dir: str | Path,
    *,
    ratio: float = 1.0,
    progress_callback: Callable[[float, str], None] | None = None,
) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ratio = min(1.0, max(0.0, float(ratio)))
    if not timeline:
        return 0
    count = len(timeline) if ratio >= 1.0 else max(1, math.ceil(len(timeline) * ratio))
    selected = timeline[:count]
    exported = 0
    for idx, item in enumerate(selected, start=1):
        ref_id = str(item.get("ref_shot_id") or f"ref_{idx:03d}")
        sub_dir = out / ref_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        if ref_video_path:
            cut_clip(
                ref_video_path,
                float(item.get("ref_start") or 0.0),
                float(item.get("ref_end") or 0.0),
                sub_dir / f"{ref_id}_ref.mp4",
            )
        movie_start = item.get("movie_start")
        movie_end = item.get("movie_end")
        if movie_video_path and movie_start is not None and movie_end is not None:
            movie_id = (item.get("movie_shot_ids") or ["movie"])[0]
            cut_clip(
                movie_video_path,
                float(movie_start),
                float(movie_end),
                sub_dir / f"{movie_id}_movie.mp4",
            )
        exported += 1
        if progress_callback:
            progress_callback(100, f"Exported matched shot clips {idx}/{len(selected)} ({ref_id})")
    return exported


def align_visual_timeline(
    ref_shots: list[dict[str, Any]],
    movie_shots: list[dict[str, Any]],
    *,
    top_n: int = 8,
    min_score: float = 0.35,
    keyframes_per_shot: int = 4,
    recall_top_k: int = 80,
    rerank_top_k: int = 20,
    refine_top_k: int = 3,
    feature_mode: str = "classic",
    alignment_mode: str = "temporal",
    temporal_radius_sec: float = 1.5,
    temporal_step_sec: float = 1.0,
    neighbor_shot_window: int = 1,
    spatial_normalize: str = "auto",
    device: str = "auto",
    feature_cache_dir: str | Path | None = None,
    save_debug_boards: bool = False,
    disable_global_path: bool = False,
    manual_overrides: dict[str, str] | None = None,
    diagnostics_dir: str | Path | None = None,
    ref_video_path: str | Path | None = None,
    movie_video_path: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    if feature_mode not in {"classic", "classic_clip"}:
        raise ValueError(f"不支持的 feature_mode: {feature_mode}")
    if alignment_mode not in {"classic", "temporal", "spatial_temporal", "topiq_temporal"}:
        raise ValueError(f"Unsupported alignment_mode: {alignment_mode}")
    if spatial_normalize not in {"auto", "off"}:
        raise ValueError(f"Unsupported spatial_normalize: {spatial_normalize}")

    effective_feature_mode = "classic"
    embedding_status = "disabled"
    if feature_mode == "classic_clip":
        embedding_status = "not_configured_fallback_to_classic"
    effective_alignment_mode = alignment_mode
    alignment_status = "enabled"
    if alignment_mode == "topiq_temporal":
        effective_alignment_mode = "temporal"
        alignment_status = "topiq_not_configured_fallback_to_temporal"

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
        candidates = build_candidates(
            ref_feature,
            movie_features,
            movie_shots,
            recall_top_k=recall_top_k,
            rerank_top_k=rerank_top_k,
        )
        if effective_alignment_mode != "classic":
            if ref_index == 1 or ref_index == total_ref or ref_index % 5 == 0:
                _progress(
                    progress_callback,
                    30.0 + (ref_index / max(1, total_ref)) * 55.0,
                    f"Refining {effective_alignment_mode} candidates {ref_index}/{total_ref}",
                )
            candidates = refine_candidates_for_ref(
                ref,
                candidates,
                movie_shots,
                alignment_mode=effective_alignment_mode,
                refine_top_k=refine_top_k,
                temporal_radius_sec=temporal_radius_sec,
                temporal_step_sec=temporal_step_sec,
                neighbor_shot_window=neighbor_shot_window,
                spatial_normalize=spatial_normalize,
                ref_video_path=ref_video_path,
                movie_video_path=movie_video_path,
                feature_cache_dir=feature_cache_dir,
                debug_dir=(Path(diagnostics_dir) / "debug_boards") if diagnostics_dir and save_debug_boards else None,
            )
        candidates_by_ref.append(candidates)
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
        chosen_path = solve_segmented_global_path(ref_shots, candidates_by_ref, min_visual_score=min_score)
        match_type = (
            "segmented_global_path"
            if effective_alignment_mode == "classic"
            else f"{effective_alignment_mode}_segmented_global_path"
        )

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
            "refine_top_k": int(refine_top_k),
            "output_top_n": int(top_n),
            "min_score": float(min_score),
            "alignment_mode": effective_alignment_mode,
            "requested_alignment_mode": alignment_mode,
            "alignment_status": alignment_status,
            "temporal_radius_sec": float(temporal_radius_sec),
            "temporal_step_sec": float(temporal_step_sec),
            "neighbor_shot_window": int(neighbor_shot_window),
            "spatial_normalize": spatial_normalize,
            "device": device,
            "feature_cache_dir": str(feature_cache_dir) if feature_cache_dir else None,
            "debug_boards_enabled": bool(save_debug_boards),
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
    parser.add_argument("--keyframes-per-shot", type=int, default=4)
    parser.add_argument("--recall-top-k", type=int, default=80)
    parser.add_argument("--rerank-top-k", type=int, default=20)
    parser.add_argument("--refine-top-k", type=int, default=3)
    parser.add_argument("--feature-mode", choices=["classic", "classic_clip"], default="classic")
    parser.add_argument("--alignment-mode", choices=["classic", "temporal", "spatial_temporal", "topiq_temporal"], default="temporal")
    parser.add_argument("--temporal-radius-sec", type=float, default=1.5)
    parser.add_argument("--temporal-step-sec", type=float, default=1.0)
    parser.add_argument("--neighbor-shot-window", type=int, default=1)
    parser.add_argument("--spatial-normalize", choices=["auto", "off"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--feature-cache-dir")
    parser.add_argument("--save-debug-boards", action="store_true")
    parser.add_argument("--disable-global-path", action="store_true")
    parser.add_argument("--manual-overrides", help="人工覆写 JSON")
    parser.add_argument("--diagnostics-dir", help="诊断输出目录；默认写到 output-dir/diagnostics")
    args = parser.parse_args()

    ref_data = read_json(args.ref_analysis)
    movie_data = read_json(args.movie_shots)
    output_dir = Path(args.output_dir)
    diagnostics_dir = Path(args.diagnostics_dir) if args.diagnostics_dir else output_dir / "diagnostics"
    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else output_dir / "cache"
    result = align_visual_timeline(
        ref_data.get("ref_shots") or [],
        movie_data.get("movie_shots") or [],
        top_n=args.top_n,
        min_score=args.min_score,
        keyframes_per_shot=args.keyframes_per_shot,
        recall_top_k=args.recall_top_k,
        rerank_top_k=args.rerank_top_k,
        refine_top_k=args.refine_top_k,
        feature_mode=args.feature_mode,
        alignment_mode=args.alignment_mode,
        temporal_radius_sec=args.temporal_radius_sec,
        temporal_step_sec=args.temporal_step_sec,
        neighbor_shot_window=args.neighbor_shot_window,
        spatial_normalize=args.spatial_normalize,
        device=args.device,
        feature_cache_dir=feature_cache_dir,
        save_debug_boards=args.save_debug_boards,
        disable_global_path=args.disable_global_path,
        manual_overrides=_load_manual_overrides(args.manual_overrides),
        diagnostics_dir=diagnostics_dir,
        ref_video_path=ref_data.get("ref_video_path"),
        movie_video_path=movie_data.get("movie_path"),
        progress_callback=lambda percent, message: emit_progress("alignment", percent, message),
    )
    out = write_json(output_dir / "ref_to_movie_timeline.json", result)
    if EXPORT_MATCHED_SHOT_CLIPS:
        exported = export_matched_shot_clips(
            result["ref_to_movie_timeline"],
            ref_data.get("ref_video_path"),
            movie_data.get("movie_path"),
            output_dir / MATCHED_SHOT_CLIPS_DIRNAME,
            ratio=MATCHED_SHOT_CLIPS_RATIO,
            progress_callback=lambda percent, message: emit_progress("alignment", percent, message),
        )
        emit_progress("alignment", 100, f"Exported {exported} matched shot clip folders")
    emit_progress("alignment", 100, "Visual alignment complete")
    print(out)


if __name__ == "__main__":
    main()
