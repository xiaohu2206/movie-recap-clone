from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from clone_narration_video.utils.video_tools import extract_frame
from clone_narration_video.utils.visual_features import (
    compare_homography_frames,
    compare_normalized_frames,
    load_normalized_frame,
)


def _id(item: dict[str, Any], key: str) -> str:
    return str(item.get(key) or item.get("shot_id") or "")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clip(value: float, low: float, high: float) -> float:
    if high < low:
        return low
    return max(low, min(high, value))


def _frame_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in item.get("sample_frames") or []:
        if isinstance(row, dict) and row.get("path"):
            rows.append(
                {
                    "path": str(row["path"]),
                    "time": _float(row.get("time"), _float(item.get("start"))),
                    "source": "sample",
                }
            )
    for row in item.get("keyframe_times") or []:
        if isinstance(row, dict) and row.get("path"):
            rows.append(
                {
                    "path": str(row["path"]),
                    "time": _float(row.get("time"), _float(item.get("start"))),
                    "source": row.get("role") or "keyframe",
                }
            )
    if not rows:
        keyframes = [str(path) for path in item.get("keyframes") or [] if path]
        start = _float(item.get("start"))
        end = _float(item.get("end"), start)
        for index, path in enumerate(keyframes):
            position = 0.5 if len(keyframes) == 1 else index / max(1, len(keyframes) - 1)
            rows.append({"path": path, "time": start + (end - start) * position, "source": "keyframe"})
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: _float(item.get("time"))):
        key = str(row.get("path") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _time_grid(start: float, end: float, step: float, max_frames: int = 64) -> list[float]:
    if end <= start:
        return [start]
    step = max(0.04, float(step))
    count = min(max_frames, max(1, int(round((end - start) / step)) + 1))
    if count == 1:
        return [(start + end) / 2.0]
    return [start + (end - start) * (i / float(count - 1)) for i in range(count)]


def _limit_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    if max_rows == 1:
        return [rows[len(rows) // 2]]
    step = (len(rows) - 1) / float(max_rows - 1)
    indexes = sorted({int(round(i * step)) for i in range(max_rows)})
    return [rows[index] for index in indexes]


def _cache_frame(video_path: str | Path, time_sec: float, cache_dir: str | Path, prefix: str) -> str | None:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{video_path}|{time_sec:.3f}|{prefix}".encode("utf-8")).hexdigest()[:16]
    out = cache / f"{prefix}_{digest}.jpg"
    if out.exists():
        return str(out)
    try:
        return str(extract_frame(video_path, time_sec, out))
    except Exception:
        return None


def _ensure_rows_from_video(
    item: dict[str, Any],
    *,
    video_path: str | Path | None,
    cache_dir: str | Path | None,
    step_sec: float,
    prefix: str,
    max_rows: int = 5,
) -> list[dict[str, Any]]:
    rows = _frame_rows(item)
    if len(rows) >= 3 or not video_path or not cache_dir:
        return _limit_rows(rows, max_rows)
    start = _float(item.get("start"))
    end = _float(item.get("end"), start)
    for time_sec in _time_grid(start, end, step_sec, max_frames=max_rows):
        path = _cache_frame(video_path, time_sec, cache_dir, prefix)
        if path:
            rows.append({"path": path, "time": time_sec, "source": "cache"})
    return _limit_rows(sorted(rows, key=lambda row: _float(row.get("time"))), max_rows)


def _candidate_rows(
    candidate: dict[str, Any],
    movie_shots: list[dict[str, Any]],
    *,
    movie_video_path: str | Path | None,
    cache_dir: str | Path | None,
    radius_sec: float,
    step_sec: float,
    neighbor_shot_window: int,
    max_rows: int = 12,
) -> list[dict[str, Any]]:
    movie_index = int(candidate.get("movie_index") or 0)
    start_index = max(0, movie_index - max(0, int(neighbor_shot_window)))
    end_index = min(len(movie_shots) - 1, movie_index + max(0, int(neighbor_shot_window)))
    rows: list[dict[str, Any]] = []
    for index in range(start_index, end_index + 1):
        rows.extend(_frame_rows(movie_shots[index]))

    if len(rows) < 3 and movie_video_path and cache_dir:
        start = max(0.0, _float(candidate.get("movie_start")) - max(0.0, radius_sec))
        end = _float(candidate.get("movie_end"), start) + max(0.0, radius_sec)
        prefix = str(candidate.get("movie_shot_id") or f"movie_{movie_index}")
        for time_sec in _time_grid(start, end, step_sec, max_frames=max_rows):
            path = _cache_frame(movie_video_path, time_sec, cache_dir, prefix)
            if path:
                rows.append({"path": path, "time": time_sec, "source": "cache"})

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: _float(item.get("time"))):
        key = str(row.get("path") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return _limit_rows(deduped, max_rows)


def _shot_at_time(movie_shots: list[dict[str, Any]], time_sec: float, fallback: dict[str, Any]) -> dict[str, Any]:
    for shot in movie_shots:
        if _float(shot.get("start")) - 0.02 <= time_sec <= _float(shot.get("end")) + 0.02:
            return shot
    return fallback


def _score_pair(
    ref_path: str,
    movie_path: str,
    *,
    mode: str,
    spatial_normalize: str,
    frame_cache: dict[str, np.ndarray | None] | None = None,
) -> dict[str, Any] | None:
    if frame_cache is not None:
        if ref_path not in frame_cache:
            frame_cache[ref_path] = load_normalized_frame(ref_path, spatial_normalize=spatial_normalize)
        if movie_path not in frame_cache:
            frame_cache[movie_path] = load_normalized_frame(movie_path, spatial_normalize=spatial_normalize)
        ref_img = frame_cache[ref_path]
        movie_img = frame_cache[movie_path]
    else:
        ref_img = load_normalized_frame(ref_path, spatial_normalize=spatial_normalize)
        movie_img = load_normalized_frame(movie_path, spatial_normalize=spatial_normalize)
    if ref_img is None or movie_img is None:
        return None
    temporal = compare_normalized_frames(ref_img, movie_img)
    spatial_score = temporal["score"]
    spatial_transform = "none"
    spatial_detail: dict[str, Any] = {}
    if mode == "spatial_temporal":
        spatial = compare_homography_frames(ref_img, movie_img)
        spatial_score = float(spatial.get("score") or 0.0)
        spatial_transform = str(spatial.get("transform") or "failed")
        spatial_detail = spatial
    return {
        "temporal_score": float(temporal["score"]),
        "spatial_score": float(spatial_score),
        "spatial_transform": spatial_transform,
        "temporal_detail": temporal,
        "spatial_detail": spatial_detail,
        "score": float(temporal["score"]) if mode != "spatial_temporal" else float(temporal["score"]) * 0.7 + spatial_score * 0.3,
    }


def _write_debug_board(
    path: str | Path,
    *,
    ref_path: str,
    movie_path: str,
    title: str,
    detail: dict[str, Any],
) -> None:
    ref = cv2.imread(ref_path)
    movie = cv2.imread(movie_path)
    if ref is None or movie is None:
        return
    height = 240

    def resize(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.resize(img, (max(1, int(round(w * height / max(1, h)))), height), interpolation=cv2.INTER_AREA)

    ref = resize(ref)
    movie = resize(movie)
    board = cv2.hconcat([ref, movie]) if ref.shape[0] == movie.shape[0] else np.hstack([ref, movie])
    lines = [
        title,
        f"temporal={detail.get('temporal_score', 0):.4f} spatial={detail.get('spatial_score', 0):.4f}",
        f"transform={detail.get('spatial_transform', 'none')} offset={detail.get('temporal_offset', 0):.3f}s",
    ]
    overlay = np.zeros((80, board.shape[1], 3), dtype=np.uint8)
    for idx, line in enumerate(lines):
        cv2.putText(overlay, line, (8, 22 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    out = np.vstack([overlay, board])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), out)


def refine_candidates_for_ref(
    ref_shot: dict[str, Any],
    candidates: list[dict[str, Any]],
    movie_shots: list[dict[str, Any]],
    *,
    alignment_mode: str,
    refine_top_k: int = 8,
    temporal_radius_sec: float = 4.0,
    temporal_step_sec: float = 0.25,
    neighbor_shot_window: int = 2,
    spatial_normalize: str = "auto",
    ref_video_path: str | Path | None = None,
    movie_video_path: str | Path | None = None,
    feature_cache_dir: str | Path | None = None,
    debug_dir: str | Path | None = None,
    max_ref_frames: int = 5,
    max_movie_frames: int = 12,
) -> list[dict[str, Any]]:
    if alignment_mode == "classic" or not candidates:
        return candidates
    mode = "temporal" if alignment_mode == "topiq_temporal" else alignment_mode
    cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
    ref_prefix = str(ref_shot.get("ref_shot_id") or ref_shot.get("shot_id") or "ref")
    ref_rows = _ensure_rows_from_video(
        ref_shot,
        video_path=ref_video_path,
        cache_dir=cache_dir,
        step_sec=temporal_step_sec,
        prefix=ref_prefix,
        max_rows=max_ref_frames,
    )
    if not ref_rows:
        return candidates

    refined: list[dict[str, Any]] = []
    debug_written = False
    frame_cache: dict[str, np.ndarray | None] = {}
    for rank, candidate in enumerate(candidates, start=1):
        if rank > max(1, int(refine_top_k)):
            refined.append(candidate)
            continue
        movie_rows = _candidate_rows(
            candidate,
            movie_shots,
            movie_video_path=movie_video_path,
            cache_dir=cache_dir,
            radius_sec=temporal_radius_sec,
            step_sec=temporal_step_sec,
            neighbor_shot_window=neighbor_shot_window,
            max_rows=max_movie_frames,
        )
        best: dict[str, Any] | None = None
        for ref_row in ref_rows:
            for movie_row in movie_rows:
                scored = _score_pair(
                    str(ref_row["path"]),
                    str(movie_row["path"]),
                    mode=mode,
                    spatial_normalize=spatial_normalize,
                    frame_cache=frame_cache,
                )
                if scored is None:
                    continue
                current = {
                    **scored,
                    "ref_path": str(ref_row["path"]),
                    "movie_path": str(movie_row["path"]),
                    "best_ref_time": _float(ref_row.get("time")),
                    "best_movie_time": _float(movie_row.get("time")),
                }
                if best is None or float(current["score"]) > float(best["score"]):
                    best = current
        if not best:
            refined.append(candidate)
            continue

        ref_start = _float(ref_shot.get("start"))
        ref_end = _float(ref_shot.get("end"), ref_start)
        ref_duration = max(0.04, ref_end - ref_start)
        ref_relative = _clip(float(best["best_ref_time"]) - ref_start, 0.0, ref_duration)
        raw_movie_start = float(best["best_movie_time"]) - ref_relative
        window_start = max(0.0, _float(candidate.get("movie_start")) - max(0.0, temporal_radius_sec))
        window_end = max(window_start + 0.04, _float(candidate.get("movie_end"), window_start) + max(0.0, temporal_radius_sec))
        movie_start = _clip(raw_movie_start, window_start, max(window_start, window_end - ref_duration))
        movie_end = min(window_end, movie_start + ref_duration)
        containing = _shot_at_time(movie_shots, float(best["best_movie_time"]), movie_shots[int(candidate.get("movie_index") or 0)])

        coarse = float(candidate.get("visual_score") or 0.0)
        temporal_score = float(best["temporal_score"])
        spatial_score = float(best["spatial_score"])
        fused = coarse * 0.45 + temporal_score * 0.40 + spatial_score * 0.15
        updated = dict(candidate)
        updated.update(
            {
                "movie_index": int(movie_shots.index(containing)) if containing in movie_shots else int(candidate.get("movie_index") or 0),
                "movie_shot_id": _id(containing, "movie_shot_id") or str(candidate.get("movie_shot_id") or ""),
                "movie_start": round(movie_start, 3),
                "movie_end": round(movie_end, 3),
                "visual_score": round(fused, 4),
                "coarse_visual_score": round(coarse, 4),
                "refinement_score": round(max(temporal_score, spatial_score), 4),
                "refinement": {
                    "enabled": True,
                    "mode": mode,
                    "best_ref_time": round(float(best["best_ref_time"]), 3),
                    "best_movie_time": round(float(best["best_movie_time"]), 3),
                    "temporal_offset": round(float(best["best_movie_time"]) - float(best["best_ref_time"]), 3),
                    "temporal_score": round(temporal_score, 4),
                    "spatial_score": round(spatial_score, 4),
                    "spatial_transform": best.get("spatial_transform") or "none",
                    "mask_used": spatial_normalize != "off",
                    "ref_frame_path": best["ref_path"],
                    "movie_frame_path": best["movie_path"],
                    "temporal_detail": best.get("temporal_detail") or {},
                    "spatial_detail": best.get("spatial_detail") or {},
                },
            }
        )
        refined.append(updated)

        if debug_dir and not debug_written:
            detail = dict(updated["refinement"])
            title = f"{ref_prefix} -> {updated.get('movie_shot_id')} score={updated.get('visual_score')}"
            _write_debug_board(Path(debug_dir) / f"{ref_prefix}.jpg", ref_path=best["ref_path"], movie_path=best["movie_path"], title=title, detail=detail)
            debug_written = True

    refined.sort(key=lambda row: float(row.get("visual_score") or 0.0), reverse=True)
    for visual_rank, row in enumerate(refined, start=1):
        row["visual_rank"] = visual_rank
    return refined
