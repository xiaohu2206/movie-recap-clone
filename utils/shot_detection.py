from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .project_paths import MODEL_DIR
from .video_tools import extract_frames, video_info

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

DEFAULT_KEYFRAME_POSITIONS = (0.2, 0.4, 0.6, 0.8)


class ShotDetectionBackendUnavailable(RuntimeError):
    pass


def _acceleration_message(info: dict[str, Any]) -> str:
    device = str(info.get("device") or "cpu").lower()
    torch_version = str(info.get("torch_version") or "unknown")
    if device.startswith("cuda"):
        device_name = str(info.get("device_name") or "CUDA GPU")
        cuda_version = str(info.get("cuda_version") or "unknown")
        return f"TransNetV2 使用 GPU 加速: {device_name} (Torch {torch_version}, CUDA {cuda_version})"
    fallback_reason = str(info.get("device_fallback_reason") or "").strip()
    suffix = f"，GPU 回退原因: {fallback_reason}" if fallback_reason else ""
    return f"TransNetV2 使用 CPU 推理 (Torch {torch_version}){suffix}"


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


def parse_keyframe_positions(value: str | list[float] | tuple[float, ...] | None) -> list[float]:
    if value is None:
        return list(DEFAULT_KEYFRAME_POSITIONS)
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
        positions = [float(part) for part in raw if part]
    else:
        positions = [float(part) for part in value]
    cleaned = sorted({min(0.95, max(0.05, item)) for item in positions})
    return cleaned or list(DEFAULT_KEYFRAME_POSITIONS)


def _time_at_position(start: float, end: float, position: float) -> float:
    duration = max(1e-6, end - start)
    return start + duration * min(0.95, max(0.05, float(position)))


def _keyframe_role(index: int) -> str:
    return f"fifth_{index + 1}"


def _sample_times(start: float, end: float, sample_fps: float, max_frames: int) -> list[float]:
    if sample_fps <= 0 or max_frames <= 0:
        return []
    duration = max(0.0, end - start)
    if duration <= 0:
        return []
    count = min(max_frames, max(1, int(round(duration * sample_fps))))
    if count == 1:
        return [(start + end) / 2.0]
    margin = min(duration * 0.08, 0.25)
    usable_start = start + margin
    usable_end = max(usable_start, end - margin)
    step = (usable_end - usable_start) / float(count - 1)
    return [usable_start + step * i for i in range(count)]


def _detect_with_transnet(
    video_path: str | Path,
    threshold: float,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[tuple[int, int]], int, dict[str, Any]]:
    try:
        from .transnetv2_torch import TransNetV2Torch
    except Exception as exc:
        raise ShotDetectionBackendUnavailable(f"TransNetV2 unavailable: {exc}") from exc

    model = TransNetV2Torch(str(MODEL_DIR))
    model_info = model.get_backend_info()
    if progress_callback:
        progress_callback(1.0, _acceleration_message(model_info))
    logger.info("[shot_detection] %s", _acceleration_message(model_info))

    def on_model_progress(percent: float) -> None:
        if progress_callback:
            progress_callback(2.0 + min(100.0, max(0.0, percent)) * 0.78, "Detecting shot boundaries")

    frames, single, many = model.predict_video(str(video_path), progress_callback=on_model_progress)
    final_model_info = model.get_backend_info()
    if str(final_model_info.get("device")) != str(model_info.get("device")):
        fallback_message = f"TransNetV2 运行时已切换: {_acceleration_message(final_model_info)}"
        if progress_callback:
            progress_callback(81.0, fallback_message)
        logger.warning("[shot_detection] %s", fallback_message)
    preds = many if many is not None and len(many) else single
    scenes = model.predictions_to_scenes(preds, threshold=threshold)
    if progress_callback:
        progress_callback(82.0, f"Detected {len(scenes)} raw shot boundaries")
    return _normalize_scenes(scenes, len(frames), 2), len(frames), final_model_info


def _detect_with_frame_diff(
    video_path: str | Path,
    threshold: float,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[tuple[int, int]], int, dict[str, Any]]:
    if progress_callback:
        progress_callback(1.0, "OpenCV 镜头检测使用 CPU")
    logger.info("[shot_detection] OpenCV 镜头检测使用 CPU")
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
    keyframe_positions: str | list[float] | tuple[float, ...] | None = None,
    sample_fps: float = 0.0,
    max_sample_frames_per_shot: int = 0,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    info = video_info(video_path)
    fps = max(1e-6, float(info.get("fps") or 25.0))
    duration = float(info.get("duration") or 0.0)
    backend_info: dict[str, Any]
    if backend == "opencv":
        scenes, frame_count, backend_info = _detect_with_frame_diff(video_path, threshold, progress_callback)
        backend_info["requested_backend"] = "opencv"
    elif backend in {"auto", "transnet"}:
        try:
            scenes, frame_count, backend_info = _detect_with_transnet(video_path, threshold, progress_callback)
        except ShotDetectionBackendUnavailable as exc:
            if backend == "transnet":
                raise
            logger.warning("[shot_detection] TransNetV2 unavailable; falling back to OpenCV: %s", exc)
            if progress_callback:
                progress_callback(1.0, "TransNetV2 unavailable, falling back to OpenCV")
            scenes, frame_count, backend_info = _detect_with_frame_diff(video_path, threshold, progress_callback)
            backend_info["fallback_from"] = "transnet"
            backend_info["fallback_reason"] = str(exc)[:500]
        backend_info["requested_backend"] = backend
    else:
        raise ValueError(f"Unsupported shot detection backend: {backend}")

    min_frames = max(2, int(round(fps * 0.15)))
    scenes = _normalize_scenes(np.array(scenes, dtype=np.int32), frame_count, min_frames)
    detection_seconds = time.perf_counter() - total_started

    key_dir = Path(keyframe_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = key_dir / "samples"
    positions = parse_keyframe_positions(keyframe_positions)
    sample_fps = max(0.0, float(sample_fps))
    max_sample_frames_per_shot = max(0, int(max_sample_frames_per_shot))
    if sample_fps > 0 and max_sample_frames_per_shot > 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    frame_requests: list[tuple[float, Path]] = []
    total_scenes = len(scenes)
    if progress_callback:
        progress_callback(84.0, f"Exporting {total_scenes} keyframes")
    for idx, (start_f, end_f) in enumerate(scenes, start=1):
        start = max(0.0, start_f / fps)
        end = min(duration or ((end_f + 1) / fps), (end_f + 1) / fps)
        if end <= start:
            end = start + (1.0 / fps)
        sid = _shot_id(shot_prefix, idx)
        keyframes: list[str] = []
        keyframe_times: list[dict[str, Any]] = []
        for keyframe_index, position in enumerate(positions):
            role = _keyframe_role(keyframe_index)
            time_sec = _time_at_position(start, end, position)
            key_path = key_dir / f"{sid}_{role}_{int(round(position * 100)):02d}.jpg"
            frame_requests.append((time_sec, key_path))
            keyframes.append(str(key_path))
            keyframe_times.append(
                {
                    "path": str(key_path),
                    "time": round(time_sec, 3),
                    "frame": int(round(time_sec * fps)),
                    "role": role,
                    "position": round(position, 4),
                }
            )

        sample_frames: list[dict[str, Any]] = []
        if sample_fps > 0 and max_sample_frames_per_shot > 0:
            for sample_idx, time_sec in enumerate(_sample_times(start, end, sample_fps, max_sample_frames_per_shot), start=1):
                sample_path = sample_dir / f"{sid}_sample_{sample_idx:03d}.jpg"
                frame_requests.append((time_sec, sample_path))
                sample_frames.append(
                    {
                        "path": str(sample_path),
                        "time": round(time_sec, 3),
                        "frame": int(round(time_sec * fps)),
                    }
                )
        shots.append(
            {
                f"{shot_prefix}_id": sid,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "keyframes": keyframes,
                "keyframe_times": keyframe_times,
                "sample_frames": sample_frames,
                "start_frame": int(start_f),
                "end_frame": int(end_f),
            }
        )
    keyframe_started = time.perf_counter()

    def on_frame_progress(processed: int, total: int) -> None:
        if progress_callback and (processed == 1 or processed == total or processed % 30 == 0):
            percent = 84.0 + (processed / max(1, total)) * 16.0
            progress_callback(percent, f"Exported frames {processed}/{total}")

    extract_frames(video_path, frame_requests, fps=fps, progress_callback=on_frame_progress)
    keyframe_seconds = time.perf_counter() - keyframe_started
    timings = dict(backend_info.get("timings") or {})
    timings.update(
        {
            "detection_seconds": round(detection_seconds, 3),
            "keyframe_seconds": round(keyframe_seconds, 3),
            "total_seconds": round(time.perf_counter() - total_started, 3),
        }
    )
    backend_info["timings"] = timings

    return {
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "frame_count": int(frame_count),
        "backend": backend_info,
        "shots": shots,
    }
