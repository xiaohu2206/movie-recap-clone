from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .ffmpeg_utils import probe_duration, probe_fps


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

