from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

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


def extract_frame(video_path: str | Path, time_sec: float, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
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
        cv2.imwrite(str(out), frame)
    finally:
        cap.release()
    return out


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

