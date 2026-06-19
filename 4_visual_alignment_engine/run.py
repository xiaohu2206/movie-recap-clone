from __future__ import annotations

import argparse
import csv
import inspect
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.progress import emit_progress
from clone_narration_video.utils.project_paths import default_output_dir
import run_shot_match_localize as shot_localizer


ALGORITHM_VERSION = "shot_match_localize_orb_v1"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
ACCEPTED_VISUAL_STATUSES = {"matched", "matched_low_confidence"}


@dataclass(frozen=True)
class ShotClip:
    index: int
    shot_id: str
    path: Path | None
    start: float
    end: float
    duration: float
    row: dict[str, Any]
    keyframe_paths: tuple[Path, ...]


def _progress(
    progress_callback: Callable[[float, str], None] | None,
    percent: float,
    message: str,
) -> None:
    if progress_callback:
        progress_callback(percent, message)


def _round(value: Any, digits: int = 3) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _rows(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = data.get(key) or []
    if not isinstance(rows, list):
        return []
    return sorted(
        (row for row in rows if isinstance(row, dict)),
        key=lambda row: float(row.get("start") or 0.0),
    )


def _shot_id(row: dict[str, Any], id_key: str, index: int) -> str:
    value = row.get(id_key) or row.get("shot_id")
    if value:
        return str(value)
    if id_key == "ref_shot_id":
        return f"ref_shot_{index + 1:03d}"
    return f"movie_shot_{index + 1:06d}"


def _find_clip_path(clip_dir: Path, shot_id: str) -> Path | None:
    for extension in VIDEO_EXTENSIONS:
        candidate = clip_dir / f"{shot_id}{extension}"
        if candidate.exists():
            return candidate
    for candidate in clip_dir.glob(f"{shot_id}.*"):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            return candidate
    return None


def build_shot_clips(
    shot_rows: list[dict[str, Any]],
    *,
    clip_dir: str | Path,
    id_key: str,
) -> list[ShotClip]:
    root = Path(clip_dir)
    clips: list[ShotClip] = []
    for index, row in enumerate(shot_rows):
        shot_id = _shot_id(row, id_key, index)
        start = float(row.get("start") or 0.0)
        end = float(row.get("end") or start)
        duration = float(row.get("duration") or max(0.0, end - start))
        keyframe_paths = tuple(
            path
            for value in (row.get("keyframes") or [])
            if isinstance(value, str) and value.strip() and (path := Path(value)).is_file()
        )
        clips.append(
            ShotClip(
                index=index,
                shot_id=shot_id,
                path=_find_clip_path(root, shot_id),
                start=start,
                end=end,
                duration=max(0.0, duration),
                row=row,
                keyframe_paths=keyframe_paths,
            )
        )
    return clips


def _has_feature_source(clip: ShotClip) -> bool:
    return clip.path is not None or bool(clip.keyframe_paths)


def _extract_features(
    clips: list[ShotClip],
    *,
    sample_count: int,
    frame_size: int,
    workers: int,
    mask_text_bands: bool,
) -> list[shot_localizer.ShotFeature]:
    if all(clip.path is not None for clip in clips):
        return shot_localizer.extract_all_features(
            [clip.path for clip in clips if clip.path],
            sample_count,
            frame_size,
            workers,
            mask_text_bands=mask_text_bands,
        )

    sources = [
        shot_localizer.KeyframeSource(
            path=clip.path or Path(f"{clip.shot_id}.keyframes"),
            frame_paths=() if clip.path else clip.keyframe_paths,
            duration=clip.duration,
        )
        for clip in clips
    ]
    return shot_localizer.extract_all_keyframe_features(
        sources,
        sample_count,
        frame_size,
        workers,
        mask_text_bands=mask_text_bands,
    )


def _escape_invalid_json_backslashes(text: str) -> str:
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    chars: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "\\":
            chars.append(text[index])
            index += 1
            continue
        run_start = index
        while index < len(text) and text[index] == "\\":
            index += 1
        run = text[run_start:index]
        next_char = text[index] if index < len(text) else ""
        chars.append(run)
        if len(run) % 2 == 1 and next_char and next_char not in valid_escapes:
            chars.append("\\")
    return "".join(chars)


def _read_json_tolerant(path: str | Path) -> Any:
    try:
        return read_json(path)
    except json.JSONDecodeError:
        text = Path(path).read_text(encoding="utf-8-sig")
        return json.loads(_escape_invalid_json_backslashes(text))


def _load_manual_overrides(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    data = _read_json_tolerant(path)
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


def _candidate_row(
    candidate: dict[str, Any],
    movie_clips: list[ShotClip],
    *,
    rank: int,
    selected_index: int,
) -> dict[str, Any] | None:
    movie_index = int(candidate.get("movie_index", -1))
    if not 0 <= movie_index < len(movie_clips):
        return None
    clip = movie_clips[movie_index]
    score = max(0.0, min(1.0, float(candidate.get("score") or 0.0)))
    return {
        "movie_shot_id": clip.shot_id,
        "movie_start": _round(clip.start),
        "movie_end": _round(clip.end),
        "visual_score": round(score, 4),
        "recall_score": round(float(candidate.get("local_score") or 0.0), 4),
        "final_score": round(score, 4),
        "coarse_visual_score": round(float(candidate.get("global_score") or 0.0), 4),
        "refinement_score": round(float(candidate.get("fine_score") or 0.0), 4),
        "visual_rank": rank,
        "final_rank": rank,
        "selected": movie_index == selected_index,
        "refinement": {
            "enabled": True,
            "mode": "orb_geometric_verification",
            "geometry_score": round(float(candidate.get("geometry_score") or 0.0), 4),
            "geometry_inliers": int(candidate.get("geometry_inliers") or 0),
            "good_matches": int(candidate.get("good_matches") or 0),
        },
        "detail": {
            "local_score": round(float(candidate.get("local_score") or 0.0), 4),
            "global_score": round(float(candidate.get("global_score") or 0.0), 4),
            "fine_score": round(float(candidate.get("fine_score") or 0.0), 4),
            "sequence_support": round(float(candidate.get("sequence_support") or 0.0), 4),
        },
    }


def _confidence(score: float, geometry_inliers: int, score_gap: float) -> str:
    if score >= 0.72 and geometry_inliers >= 20 and score_gap >= 0.03:
        return "high"
    if score >= 0.55 or geometry_inliers >= 8:
        return "medium"
    return "low"


def _timeline_item(
    ref_clip: ShotClip,
    match: dict[str, Any] | None,
    movie_clips: list[ShotClip],
    *,
    min_score: float,
    min_geometry_inliers: int,
    top_k: int,
    reason: str | None = None,
    manual_override: bool = False,
) -> dict[str, Any]:
    selected_index = int((match or {}).get("movie_index", -1))
    selected_clip = movie_clips[selected_index] if 0 <= selected_index < len(movie_clips) else None
    raw_candidates = list((match or {}).get("top_candidates") or [])
    candidates = [
        row
        for rank, candidate in enumerate(raw_candidates[: max(1, top_k)], start=1)
        if (row := _candidate_row(candidate, movie_clips, rank=rank, selected_index=selected_index))
    ]

    score = max(0.0, min(1.0, float((match or {}).get("score") or 0.0)))
    geometry_inliers = int((match or {}).get("geometry_inliers") or 0)
    good_matches = int((match or {}).get("good_matches") or 0)
    alternative_scores = [
        float(candidate.get("final_score") or 0.0)
        for candidate in candidates
        if not candidate.get("selected")
    ]
    score_gap = round(score - max(alternative_scores), 4) if alternative_scores else round(score, 4)
    confidence = _confidence(score, geometry_inliers, score_gap)

    if reason:
        status = reason
        confidence = "low"
    elif manual_override:
        status = "matched"
        confidence = "high"
    elif score < min_score:
        status = "needs_review"
    elif confidence == "low" or geometry_inliers < min_geometry_inliers:
        status = "matched_low_confidence"
    else:
        status = "matched"

    return {
        "ref_shot_id": ref_clip.shot_id,
        "ref_start": _round(ref_clip.start),
        "ref_end": _round(ref_clip.end),
        "movie_start": _round(selected_clip.start) if selected_clip else None,
        "movie_end": _round(selected_clip.end) if selected_clip else None,
        "movie_shot_ids": [selected_clip.shot_id] if selected_clip else [],
        "match_score": round(score, 4),
        "final_score": round(score, 4),
        "match_type": "manual_override" if manual_override else "independent_shot_localize",
        "confidence": confidence,
        "status": status,
        "diagnostics": {
            "score_gap": score_gap,
            "geometry_inliers": geometry_inliers,
            "good_matches": good_matches,
            "accepted_reason": reason
            or ("manual_override" if manual_override else "localized_by_run_shot_match_localize"),
            "refinement": candidates[0].get("refinement") if candidates else {"enabled": False},
        },
        "candidates": candidates,
    }


def _manual_match(
    ref_feature: shot_localizer.ShotFeature,
    movie_feature: shot_localizer.ShotFeature,
    movie_index: int,
) -> dict[str, Any]:
    score = max(0.0, min(1.0, shot_localizer.fine_score(ref_feature, movie_feature)))
    candidate = {
        "movie_index": movie_index,
        "movie_name": movie_feature.path.name,
        "score": score,
        "geometry_score": 1.0,
        "geometry_inliers": 999,
        "good_matches": 999,
        "local_score": score,
        "global_score": score,
        "fine_score": score,
        "sequence_support": 0.0,
        "movie_duration": movie_feature.duration,
    }
    return {
        "movie_index": movie_index,
        "score": score,
        "geometry_inliers": 999,
        "good_matches": 999,
        "top_candidates": [candidate],
        "manual_override": True,
    }


def _summary(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in timeline if row.get("status") in ACCEPTED_VISUAL_STATUSES]
    high = [row for row in timeline if row.get("confidence") == "high"]
    review = [
        row
        for row in timeline
        if row.get("confidence") == "low" or row.get("status") not in ACCEPTED_VISUAL_STATUSES
    ]
    backward_count = 0
    duplicate_count = 0
    previous_start: float | None = None
    seen: set[str] = set()
    for row in timeline:
        movie_start = row.get("movie_start")
        if movie_start is not None:
            current_start = float(movie_start)
            if previous_start is not None and current_start < previous_start - 0.5:
                backward_count += 1
            previous_start = current_start
        for shot_id in row.get("movie_shot_ids") or []:
            if shot_id in seen:
                duplicate_count += 1
            seen.add(shot_id)
    total = len(timeline)
    return {
        "total_ref_shots": total,
        "matched_count": len(matched),
        "matched_rate": round(len(matched) / max(1, total), 4),
        "high_confidence_count": len(high),
        "high_confidence_rate": round(len(high) / max(1, total), 4),
        "manual_review_count": len(review),
        "timeline_backward_count": backward_count,
        "duplicate_movie_shot_count": duplicate_count,
    }


def _clear_directory_children(path: Path, output_root: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    resolved_path = path.resolve()
    resolved_root = output_root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to clear directory outside output root: {path}") from exc
    if resolved_path == resolved_root:
        raise RuntimeError(f"Refusing to clear output root directly: {path}")
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def export_review_artifacts(
    timeline: list[dict[str, Any]],
    ref_clips: list[ShotClip],
    movie_clips: list[ShotClip],
    output_root: Path,
    *,
    low_score_threshold: float,
    min_geometry_inliers: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    pair_root = output_root / "pairs"
    low_conf_root = output_root / "low_confidence_pairs"
    _clear_directory_children(pair_root, output_root)
    _clear_directory_children(low_conf_root, output_root)

    ref_by_id = {clip.shot_id: clip for clip in ref_clips}
    movie_by_id = {clip.shot_id: clip for clip in movie_clips}
    match_rows: list[dict[str, Any]] = []
    for ref_index, row in enumerate(timeline):
        ref_clip = ref_by_id.get(str(row.get("ref_shot_id") or ""))
        movie_ids = [str(value) for value in row.get("movie_shot_ids") or []]
        movie_clip = movie_by_id.get(movie_ids[0]) if movie_ids else None
        if not ref_clip or not ref_clip.path or not movie_clip or not movie_clip.path:
            continue

        score = float(row.get("match_score") or 0.0)
        folder_name = f"{ref_index + 1:04d}_{ref_clip.shot_id}__{movie_clip.shot_id}__{score:.4f}"
        target_dir = pair_root / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref_clip.path, target_dir / f"reference{ref_clip.path.suffix.lower()}")
        shutil.copy2(movie_clip.path, target_dir / f"movie{movie_clip.path.suffix.lower()}")
        write_json(target_dir / "match.json", row)

        geometry_inliers = int((row.get("diagnostics") or {}).get("geometry_inliers") or 0)
        if score < low_score_threshold or geometry_inliers < min_geometry_inliers:
            low_target_dir = low_conf_root / folder_name
            shutil.copytree(target_dir, low_target_dir)

        candidates = row.get("candidates") or []
        match_row: dict[str, Any] = {
            "ref_index": ref_index,
            "ref_name": ref_clip.path.name,
            "movie_index": movie_clip.index,
            "movie_name": movie_clip.path.name,
            "score": score,
            "ref_duration": ref_clip.duration,
            "movie_duration": movie_clip.duration,
            "status": row.get("status"),
            "confidence": row.get("confidence"),
            "geometry_inliers": geometry_inliers,
            "good_matches": int((row.get("diagnostics") or {}).get("good_matches") or 0),
        }
        for candidate_index in range(3):
            prefix = f"top{candidate_index + 1}"
            candidate = candidates[candidate_index] if candidate_index < len(candidates) else {}
            match_row[f"{prefix}_movie_shot_id"] = candidate.get("movie_shot_id", "")
            match_row[f"{prefix}_score"] = candidate.get("final_score", "")
        match_rows.append(match_row)

    fieldnames = [
        "ref_index", "ref_name", "movie_index", "movie_name", "score",
        "ref_duration", "movie_duration", "status", "confidence",
        "geometry_inliers", "good_matches",
        "top1_movie_shot_id", "top1_score", "top2_movie_shot_id", "top2_score",
        "top3_movie_shot_id", "top3_score",
    ]
    with (output_root / "matches.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(match_rows)
    write_json(output_root / "matches.json", match_rows)


def write_low_confidence_report(output_dir: str | Path, timeline: list[dict[str, Any]]) -> Path:
    items = [
        {
            "ref_shot_id": row.get("ref_shot_id"),
            "status": row.get("status"),
            "confidence": row.get("confidence"),
            "match_score": row.get("match_score"),
            "movie_shot_ids": row.get("movie_shot_ids") or [],
            "diagnostics": row.get("diagnostics") or {},
        }
        for row in timeline
        if row.get("confidence") == "low" or row.get("status") not in ACCEPTED_VISUAL_STATUSES
    ]
    return write_json(Path(output_dir) / "low_confidence_report.json", {"items": items})


def align_visual_timeline(
    ref_analysis: dict[str, Any],
    movie_data: dict[str, Any],
    *,
    ref_clip_dir: str | Path,
    movie_clip_dir: str | Path,
    output_dir: str | Path | None = None,
    sample_count: int = 6,
    frame_size: int = 384,
    workers: int = 4,
    neighbor_radius: int = 2,
    candidate_count: int = 30,
    geometry_candidate_count: int = 24,
    min_score: float = 0.35,
    low_score_threshold: float = 0.35,
    min_geometry_inliers: int = 20,
    top_k: int = 3,
    manual_overrides: dict[str, str] | None = None,
    diagnostics_dir: str | Path | None = None,
    export_pairs: bool = True,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    ref_rows = _rows(ref_analysis, "ref_shots")
    movie_rows = _rows(movie_data, "movie_shots")
    if not ref_rows:
        raise RuntimeError("ref_analysis.json is missing ref_shots")
    if not movie_rows:
        raise RuntimeError("movie_shots.json is missing movie_shots")

    ref_clips = build_shot_clips(ref_rows, clip_dir=ref_clip_dir, id_key="ref_shot_id")
    all_movie_clips = build_shot_clips(movie_rows, clip_dir=movie_clip_dir, id_key="movie_shot_id")
    active_ref_clips = [clip for clip in ref_clips if _has_feature_source(clip)]
    movie_clips = [clip for clip in all_movie_clips if _has_feature_source(clip)]
    if not active_ref_clips:
        raise RuntimeError(
            f"No reference shot clips found in {ref_clip_dir}, and ref_analysis.json has no usable keyframes"
        )
    if not movie_clips:
        raise RuntimeError(
            f"No movie shot clips found in {movie_clip_dir}, and movie_shots.json has no usable keyframes"
        )

    _progress(progress_callback, 2.0, f"Scanned {len(ref_clips)} reference shots and {len(all_movie_clips)} movie shots")
    _progress(progress_callback, 5.0, "Extracting reference shot features")
    ref_features = _extract_features(
        active_ref_clips,
        sample_count=sample_count,
        frame_size=frame_size,
        workers=workers,
        mask_text_bands=True,
    )
    _progress(progress_callback, 35.0, "Extracting movie shot features")
    movie_features = _extract_features(
        movie_clips,
        sample_count=sample_count,
        frame_size=frame_size,
        workers=workers,
        mask_text_bands=False,
    )

    _progress(progress_callback, 68.0, "Building global shot similarity matrix")
    similarity = shot_localizer.cosine_matrix(ref_features, movie_features)
    _progress(progress_callback, 72.0, "Localizing shots with ORB retrieval and geometric verification")
    localize_kwargs: dict[str, Any] = {}
    localize_parameters = inspect.signature(shot_localizer.independent_localize).parameters
    if "geometry_candidate_count" in localize_parameters:
        localize_kwargs["geometry_candidate_count"] = geometry_candidate_count
    if "workers" in localize_parameters:
        localize_kwargs["workers"] = workers
    matches = shot_localizer.independent_localize(
        ref_features,
        movie_features,
        similarity,
        neighbor_radius,
        candidate_count,
        top_k,
        **localize_kwargs,
    )
    match_by_ref_index = {
        active_ref_clips[int(match["ref_index"])].index: match
        for match in matches
        if 0 <= int(match.get("ref_index", -1)) < len(active_ref_clips)
    }

    manual_count = 0
    ref_feature_by_id = {
        clip.shot_id: feature for clip, feature in zip(active_ref_clips, ref_features)
    }
    movie_feature_by_id = {
        clip.shot_id: (index, feature)
        for index, (clip, feature) in enumerate(zip(movie_clips, movie_features))
    }
    for ref_id, movie_id in (manual_overrides or {}).items():
        ref_feature = ref_feature_by_id.get(ref_id)
        movie_hit = movie_feature_by_id.get(movie_id)
        ref_clip = next((clip for clip in active_ref_clips if clip.shot_id == ref_id), None)
        if not ref_feature or not movie_hit or not ref_clip:
            continue
        movie_index, movie_feature = movie_hit
        match_by_ref_index[ref_clip.index] = _manual_match(ref_feature, movie_feature, movie_index)
        manual_count += 1

    timeline: list[dict[str, Any]] = []
    for ref_clip in ref_clips:
        if not _has_feature_source(ref_clip):
            timeline.append(
                _timeline_item(
                    ref_clip,
                    None,
                    movie_clips,
                    min_score=min_score,
                    min_geometry_inliers=min_geometry_inliers,
                    top_k=top_k,
                    reason="missing_reference_clip",
                )
            )
            continue
        match = match_by_ref_index.get(ref_clip.index)
        timeline.append(
            _timeline_item(
                ref_clip,
                match,
                movie_clips,
                min_score=min_score,
                min_geometry_inliers=min_geometry_inliers,
                top_k=top_k,
                reason=None if match else "localization_failed",
                manual_override=bool((match or {}).get("manual_override")),
            )
        )

    if diagnostics_dir:
        _progress(progress_callback, 94.0, "Writing alignment diagnostics")
        write_low_confidence_report(diagnostics_dir, timeline)
    if output_dir and export_pairs:
        _progress(progress_callback, 96.0, "Exporting review clip pairs")
        export_review_artifacts(
            timeline,
            ref_clips,
            movie_clips,
            Path(output_dir),
            low_score_threshold=low_score_threshold,
            min_geometry_inliers=min_geometry_inliers,
        )

    return {
        "ref_to_movie_timeline": timeline,
        "metadata": {
            "algorithm_version": ALGORITHM_VERSION,
            "localizer_module": "run_shot_match_localize.py",
            "source": "segmented_shot_clips_or_keyframes",
            "ref_clip_dir": str(ref_clip_dir),
            "movie_clip_dir": str(movie_clip_dir),
            "sample_count": int(sample_count),
            "frame_size": int(frame_size),
            "workers": int(workers),
            "neighbor_radius": int(neighbor_radius),
            "candidate_count": int(candidate_count),
            "geometry_candidate_count": int(geometry_candidate_count),
            "top_k": int(top_k),
            "min_score": float(min_score),
            "low_score_threshold": float(low_score_threshold),
            "min_geometry_inliers": int(min_geometry_inliers),
            "manual_override_count": manual_count,
            "missing_reference_clip_count": len(ref_clips) - len(active_ref_clips),
            "missing_movie_clip_count": len(all_movie_clips) - len(movie_clips),
            "active_ref_shot_count": len(active_ref_clips),
            "active_movie_shot_count": len(movie_clips),
            "reference_keyframe_fallback_count": sum(clip.path is None for clip in active_ref_clips),
            "movie_keyframe_fallback_count": sum(clip.path is None for clip in movie_clips),
            "export_pairs": bool(export_pairs),
            "summary": _summary(timeline),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shot matching and write the pipeline-compatible visual timeline.")
    parser.add_argument("--ref-analysis", required=True, help="Stage 1 ref_analysis.json")
    parser.add_argument("--movie-shots", required=True, help="Stage 3 movie_shots.json")
    parser.add_argument("--output-dir", default=str(default_output_dir("4_visual_alignment_engine")))
    parser.add_argument("--reference-dir", type=Path, help="Directory containing reference shot clips")
    parser.add_argument("--movie-dir", type=Path, help="Directory containing movie shot clips")
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--keyframes-per-shot", type=int, help="Compatibility alias for --sample-count")
    parser.add_argument("--frame-size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--neighbor-radius", "--neighbor-shot-window", dest="neighbor_radius", type=int, default=2)
    parser.add_argument("--candidate-count", "--recall-top-k", dest="candidate_count", type=int, default=30)
    parser.add_argument("--geometry-candidate-count", type=int, default=24)
    parser.add_argument("--top-k", "--top-n", dest="top_k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--low-score-threshold", type=float)
    parser.add_argument("--min-geometry-inliers", type=int, default=20)
    parser.add_argument("--manual-overrides")
    parser.add_argument("--diagnostics-dir")
    parser.add_argument("--no-export-pairs", action="store_true")

    # Retained so older pipeline invocations do not break after switching localizers.
    parser.add_argument("--feature-cache-dir")
    parser.add_argument("--feature-mode", default="shot_localize")
    parser.add_argument("--alignment-mode", default="independent_localize")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--refine-window", type=int)
    parser.add_argument("--context-radius", type=int)
    parser.add_argument("--repeat-penalty", type=float)
    parser.add_argument("--advance-bonus", type=float)
    parser.add_argument("--max-repeat", type=int)
    parser.add_argument("--rerank-top-n", "--rerank-top-k", dest="rerank_top_n", type=int)
    parser.add_argument("--temporal-radius-sec", type=float)
    parser.add_argument("--temporal-step-sec", type=float)
    parser.add_argument("--spatial-normalize", default="auto")
    parser.add_argument("--save-debug-boards", action="store_true")
    parser.add_argument("--disable-global-path", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref_analysis_path = Path(args.ref_analysis)
    movie_shots_path = Path(args.movie_shots)
    output_dir = Path(args.output_dir)
    diagnostics_dir = Path(args.diagnostics_dir) if args.diagnostics_dir else output_dir / "diagnostics"
    sample_count = args.keyframes_per_shot if args.keyframes_per_shot is not None else args.sample_count
    low_score_threshold = args.low_score_threshold if args.low_score_threshold is not None else args.min_score

    result = align_visual_timeline(
        _read_json_tolerant(ref_analysis_path),
        _read_json_tolerant(movie_shots_path),
        ref_clip_dir=args.reference_dir or ref_analysis_path.parent / "shot_clips",
        movie_clip_dir=args.movie_dir or movie_shots_path.parent / "shot_clips",
        output_dir=output_dir,
        sample_count=sample_count,
        frame_size=args.frame_size,
        workers=args.workers,
        neighbor_radius=args.neighbor_radius,
        candidate_count=args.candidate_count,
        geometry_candidate_count=args.geometry_candidate_count,
        min_score=args.min_score,
        low_score_threshold=low_score_threshold,
        min_geometry_inliers=args.min_geometry_inliers,
        top_k=args.top_k,
        manual_overrides=_load_manual_overrides(args.manual_overrides),
        diagnostics_dir=diagnostics_dir,
        export_pairs=not args.no_export_pairs,
        progress_callback=lambda percent, message: emit_progress("alignment", percent, message),
    )
    out = write_json(output_dir / "ref_to_movie_timeline.json", result)
    emit_progress("alignment", 100, "Visual alignment complete")
    print(out)


if __name__ == "__main__":
    main()
