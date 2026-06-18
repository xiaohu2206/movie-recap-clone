from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2

from .ffmpeg_utils import (
    get_ffmpeg_cuda_prefix,
    probe_duration,
    probe_fps,
    resolve_ffmpeg_bin,
    run_ffmpeg,
)


def video_info(path: str | Path) -> dict[str, Any]:
    return {"duration": probe_duration(path), "fps": probe_fps(path)}


def _write_frame_image(out_path: str | Path, frame: Any) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, frame)
    if not ok:
        raise RuntimeError(f"Failed to encode frame: {out}")
    out.write_bytes(encoded.tobytes())
    if not out.exists() or out.stat().st_size <= 0:
        raise RuntimeError(f"Failed to write frame: {out}")
    return out


def extract_frame(video_path: str | Path, time_sec: float, out_path: str | Path) -> Path:
    out = Path(out_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
                ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"抽帧失败: {video_path} @ {time_sec:.3f}s")
        _write_frame_image(out, frame)
    finally:
        cap.release()
    return out


def extract_frames(
    video_path: str | Path,
    requests: Iterable[tuple[float, str | Path]],
    *,
    fps: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Extract sorted frame requests while keeping a single decoder open."""
    items = [(max(0.0, float(time_sec)), Path(out_path)) for time_sec, out_path in requests]
    if not items:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    actual_fps = max(1e-6, float(fps or cap.get(cv2.CAP_PROP_FPS) or 25.0))
    indexed = sorted(
        ((int(round(time_sec * actual_fps)), order, out_path) for order, (time_sec, out_path) in enumerate(items)),
        key=lambda item: (item[0], item[1]),
    )
    results: list[Path | None] = [None] * len(items)
    max_sequential_gap = max(1, int(round(actual_fps * 2.0)))
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    cached_frame_index = -1
    cached_frame = None

    try:
        for processed, (target_frame, order, out_path) in enumerate(indexed, start=1):
            frame = cached_frame if target_frame == cached_frame_index else None
            if frame is None:
                if target_frame < current_frame or target_frame - current_frame > max_sequential_gap:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or target_frame)

                ok = False
                while current_frame <= target_frame:
                    ok, candidate = cap.read()
                    if not ok or candidate is None:
                        break
                    frame = candidate
                    cached_frame_index = current_frame
                    cached_frame = candidate
                    current_frame += 1

                if not ok or frame is None:
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    if frame_count > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
                        ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"Failed to extract frame: {video_path} @ frame {target_frame}")

            _write_frame_image(out_path, frame)
            results[order] = out_path
            if progress_callback:
                progress_callback(processed, len(indexed))
    finally:
        cap.release()

    return [path for path in results if path is not None]


def cut_clip(video_path: str | Path, start: float, end: float, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.04, float(end) - float(start))
    cmd = [
        resolve_ffmpeg_bin(),
        "-y",
        *get_ffmpeg_cuda_prefix(),
        "-ss",
        f"{max(0.0, float(start)):.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        str(out),
    ]
    try:
        run_ffmpeg(cmd)
    except RuntimeError:
        run_ffmpeg([x for x in cmd if x not in {"-hwaccel", "cuda"}])
    return out


def export_shot_clips(
    video_path: str | Path,
    shots: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    id_key: str,
    ratio: float = 1.0,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ratio = min(1.0, max(0.0, float(ratio)))
    if not shots:
        return []
    count = len(shots) if ratio >= 1.0 else max(1, math.ceil(len(shots) * ratio))
    selected = shots[:count]
    paths: list[str] = []
    for idx, shot in enumerate(selected, start=1):
        sid = str(shot.get(id_key) or f"shot_{idx:03d}")
        clip_path = cut_clip(video_path, shot["start"], shot["end"], out / f"{sid}.mp4")
        paths.append(str(clip_path))
        if progress_callback:
            progress_callback(idx, len(selected), str(clip_path))
    return paths

