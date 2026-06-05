from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .project_paths import MODEL_DIR
from .video_tools import extract_frame, video_info

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _shot_id(prefix: str, idx: int) -> str:
    width = 6 if prefix == "movie_shot" else 3
    return f"{prefix}_{idx:0{width}d}"


def _normalize_scenes(scenes: np.ndarray, frame_count: int, min_frames: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in scenes.tolist():
        s_i = max(0, int(s))
        e_i = min(max(0, frame_count - 1), int(e))
        if e_i < s_i:
            continue
        if out and s_i <= out[-1][1]:
            out[-1] = (out[-1][0], e_i)
        elif e_i - s_i + 1 >= min_frames or not out:
            out.append((s_i, e_i))
    return out or [(0, max(0, frame_count - 1))]


def _detect_with_transnet(
    video_path: str | Path,
    threshold: float,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[tuple[int, int]], int, dict[str, Any]]:
    from .transnetv2_torch import TransNetV2Torch

    model = TransNetV2Torch(str(MODEL_DIR))
    if progress_callback:
        progress_callback(1.0, "Loading TransNetV2 model")

    def on_model_progress(percent: float) -> None:
        if progress_callback:
            progress_callback(2.0 + min(100.0, max(0.0, percent)) * 0.78, "Detecting shot boundaries")

    frames, single, many = model.predict_video(str(video_path), progress_callback=on_model_progress)
    preds = many if many is not None and len(many) else single
    scenes = model.predictions_to_scenes(preds, threshold=threshold)
    if progress_callback:
        progress_callback(82.0, f"Detected {len(scenes)} raw shot boundaries")
    return _normalize_scenes(scenes, len(frames), 2), len(frames), model.get_backend_info()


def _detect_with_frame_diff(
    video_path: str | Path,
    threshold: float,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[tuple[int, int]], int, dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    scores: list[float] = []
    frames: list[int] = []
    prev_gray = None
    prev_hist = None
    prev_edge = None
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            edge = cv2.Canny(gray, 80, 160)
            if prev_gray is not None and prev_hist is not None and prev_edge is not None:
                gray_score = float(np.mean(cv2.absdiff(prev_gray, gray))) / 255.0
                hist_score = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                edge_score = float(np.mean(cv2.absdiff(prev_edge, edge))) / 255.0
                score = gray_score * 0.50 + hist_score * 0.40 + edge_score * 0.10
                scores.append(score)
                frames.append(idx)
            prev_gray = gray
            prev_hist = hist
            prev_edge = edge
            idx += 1
            if progress_callback and total_frames > 0 and (idx == 1 or idx % 300 == 0):
                progress_callback(min(82.0, (idx / total_frames) * 82.0), "Scanning frames with OpenCV")
    finally:
        cap.release()

    frame_count = max(1, idx)
    if not scores:
        return [(0, frame_count - 1)], frame_count, {"backend": "opencv_frame_diff", "device": "cpu", "cut_count": 0}

    arr = np.array(scores, dtype=np.float32)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    percentile = float(np.percentile(arr, max(90.0, min(99.7, 99.3 - (1.0 - threshold) * 5.0))))
    adaptive_threshold = max(0.045, median + max(2.2, threshold * 5.0) * max(mad, 0.002), percentile)

    cuts = [0]
    min_gap = max(8, int(round(fps * 0.35)))
    for i, score in enumerate(scores):
        left = scores[i - 1] if i > 0 else -1.0
        right = scores[i + 1] if i + 1 < len(scores) else -1.0
        frame_idx = frames[i]
        if score >= adaptive_threshold and score >= left and score >= right and frame_idx - cuts[-1] >= min_gap:
            cuts.append(frame_idx)

    bounds = []
    for i, start in enumerate(cuts):
        end = (cuts[i + 1] - 1) if i + 1 < len(cuts) else frame_count - 1
        bounds.append((start, end))
    return (
        bounds,
        frame_count,
        {
            "backend": "opencv_frame_diff",
            "device": "cpu",
            "cut_count": max(0, len(cuts) - 1),
            "min_gap_frames": min_gap,
            "adaptive_threshold": round(adaptive_threshold, 5),
            "score_median": round(median, 5),
            "score_p99": round(float(np.percentile(arr, 99.0)), 5),
        },
    )


def detect_shots(
    video_path: str | Path,
    *,
    shot_prefix: str,
    keyframe_dir: str | Path,
    threshold: float = 0.5,
    backend: str = "auto",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    info = video_info(video_path)
    fps = max(1e-6, float(info.get("fps") or 25.0))
    duration = float(info.get("duration") or 0.0)
    backend_info: dict[str, Any]
    if backend == "opencv":
        scenes, frame_count, backend_info = _detect_with_frame_diff(video_path, threshold, progress_callback)
    else:
        scenes, frame_count, backend_info = _detect_with_transnet(video_path, threshold, progress_callback)

    min_frames = max(2, int(round(fps * 0.15)))
    scenes = _normalize_scenes(np.array(scenes, dtype=np.int32), frame_count, min_frames)

    key_dir = Path(keyframe_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    total_scenes = len(scenes)
    if progress_callback:
        progress_callback(84.0, f"Exporting {total_scenes} keyframes")
    for idx, (start_f, end_f) in enumerate(scenes, start=1):
        start = max(0.0, start_f / fps)
        end = min(duration or ((end_f + 1) / fps), (end_f + 1) / fps)
        if end <= start:
            end = start + (1.0 / fps)
        mid = (start + end) / 2.0
        sid = _shot_id(shot_prefix, idx)
        key_path = extract_frame(video_path, mid, key_dir / f"{sid}_mid.jpg")
        shots.append(
            {
                f"{shot_prefix}_id": sid,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "keyframes": [str(key_path)],
                "start_frame": int(start_f),
                "end_frame": int(end_f),
            }
        )
        if progress_callback and (idx == 1 or idx == total_scenes or idx % 10 == 0):
            percent = 84.0 + (idx / max(1, total_scenes)) * 16.0
            progress_callback(percent, f"Exported keyframes {idx}/{total_scenes}")

    return {
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "frame_count": int(frame_count),
        "backend": backend_info,
        "shots": shots,
    }
