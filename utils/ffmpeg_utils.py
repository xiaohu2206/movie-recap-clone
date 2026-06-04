from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


WIN_NO_WINDOW: int = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_HWACCEL_VALUES = {"auto", "cpu", "cuda"}
_CUDA_CACHE: Optional[bool] = None


def _bin_name(base: str) -> str:
    return f"{base}.exe" if os.name == "nt" else base


def _resolve_bin(base: str) -> Optional[str]:
    name = _bin_name(base)

    env_path = os.environ.get("FFMPEG_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            if base == "ffprobe":
                if "ffprobe" in p.name.lower():
                    return str(p)
                sibling = p.parent / name
                if sibling.exists():
                    return str(sibling)
            else:
                return str(p)

    env_dir = os.environ.get("FFMPEG_DIR") or os.environ.get("FFMPEG_HOME")
    if env_dir:
        p = Path(env_dir) / name
        if p.exists():
            return str(p)

    hit = shutil.which(base)
    if hit:
        return hit

    try:
        import imageio_ffmpeg

        ff = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if base == "ffmpeg" and ff.exists():
            return str(ff)
        probe = ff.parent / name
        if base == "ffprobe" and probe.exists():
            return str(probe)
    except Exception:
        pass

    roots = []
    try:
        roots.append(Path(sys.executable).resolve().parent)
    except Exception:
        pass
    roots.extend([Path.cwd(), Path(__file__).resolve().parents[2]])

    for root in roots:
        for candidate in (root / name, root / "resources" / name, root / "src-tauri" / "resources" / name):
            if candidate.exists():
                return str(candidate)

    if os.name == "nt":
        for candidate in (
            Path("C:/Program Files/ffmpeg/bin") / name,
            Path("C:/ffmpeg/bin") / name,
            Path.home() / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / name,
            Path("C:/ProgramData/chocolatey/bin") / name,
        ):
            if candidate.exists():
                return str(candidate)
    return None


def resolve_ffmpeg_bin() -> str:
    hit = _resolve_bin("ffmpeg")
    if not hit:
        raise FileNotFoundError("未找到 ffmpeg，请安装 ffmpeg 或设置 FFMPEG_PATH/FFMPEG_DIR")
    return hit


def resolve_ffprobe_bin() -> str:
    hit = _resolve_bin("ffprobe")
    if not hit:
        raise FileNotFoundError("未找到 ffprobe，请安装 ffmpeg 或设置 FFMPEG_PATH/FFMPEG_DIR")
    return hit


def run_ffmpeg(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        kwargs["creationflags"] = WIN_NO_WINDOW
    proc = subprocess.run(args, check=False, **kwargs)
    if check and proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "ffmpeg failed").strip())
    return proc


def get_ffmpeg_hwaccel_mode() -> str:
    raw = str(os.environ.get("CLONE_FFMPEG_HWACCEL") or os.environ.get("OMNI_FFMPEG_HWACCEL") or "auto").strip().lower()
    return raw if raw in _HWACCEL_VALUES else "auto"


def ffmpeg_supports_cuda_hwaccel() -> bool:
    global _CUDA_CACHE
    if _CUDA_CACHE is not None:
        return _CUDA_CACHE
    try:
        proc = run_ffmpeg([resolve_ffmpeg_bin(), "-hide_banner", "-hwaccels"], check=False)
        text = f"{proc.stdout}\n{proc.stderr}".lower()
        _CUDA_CACHE = proc.returncode == 0 and any(x in text for x in ("cuda", "nvdec", "cuvid"))
    except Exception:
        _CUDA_CACHE = False
    return _CUDA_CACHE


def get_ffmpeg_cuda_prefix() -> Tuple[str, ...]:
    mode = get_ffmpeg_hwaccel_mode()
    if mode == "cpu":
        return ()
    if mode == "cuda" or ffmpeg_supports_cuda_hwaccel():
        return ("-hwaccel", "cuda")
    return ()


def ffprobe_json(path: str | Path) -> dict:
    cmd = [
        resolve_ffprobe_bin(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = run_ffmpeg(cmd)
    return json.loads(proc.stdout or "{}")


def probe_duration(path: str | Path) -> float:
    data = ffprobe_json(path)
    try:
        return float((data.get("format") or {}).get("duration") or 0.0)
    except Exception:
        return 0.0


def probe_fps(path: str | Path) -> float:
    data = ffprobe_json(path)
    for stream in data.get("streams") or []:
        if stream.get("codec_type") != "video":
            continue
        rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")
        if "/" in rate:
            a, b = rate.split("/", 1)
            try:
                den = float(b)
                return float(a) / den if den else 25.0
            except Exception:
                pass
        try:
            value = float(rate)
            if value > 0:
                return value
        except Exception:
            pass
    return 25.0


def extract_audio_mp3(video_path: str | Path, audio_path: str | Path) -> Path:
    out = Path(audio_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        resolve_ffmpeg_bin(),
        "-y",
        *get_ffmpeg_cuda_prefix(),
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "3",
        str(out),
    ]
    try:
        run_ffmpeg(cmd)
    except RuntimeError:
        cmd = [x for x in cmd if x not in {"-hwaccel", "cuda"}]
        run_ffmpeg(cmd)
    return out

