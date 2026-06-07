from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.ffmpeg_utils import probe_duration, ffprobe_json, resolve_ffmpeg_bin, run_ffmpeg
from clone_narration_video.utils.generate_video_audio_modes import (
    audio_mode_for_item,
    is_original_audio_item,
    original_audio_result,
)
from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.progress import emit_progress
from clone_narration_video.utils.project_paths import default_output_dir, relpath, resolve_existing_path
from clone_narration_video.utils.tts.edge.edge_tts_service import edge_tts_service


def _round(value: float) -> float:
    return round(float(value), 3)


def _s_to_us(value: float) -> int:
    return int(round(max(0.0, float(value)) * 1_000_000))


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _safe_name(value: str, fallback: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\r\n\t]+', "_", (value or "").strip()).strip(" .")
    return safe or fallback


def _default_jianying_draft_root() -> Path | None:
    candidates: list[Path] = []
    if sys.platform.startswith("win"):
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            candidates.extend(
                [
                    Path(local_app_data) / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft",
                    Path(local_app_data) / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft",
                ]
            )
    elif sys.platform == "darwin":
        home = Path.home()
        candidates.extend(
            [
                home / "Movies" / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft",
                home / "Movies" / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft",
            ]
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _quote_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", r"'\''")


def _timeline_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("final_timeline") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("final_timeline.json 中没有 final_timeline 数据")
    return [row for row in rows if isinstance(row, dict)]


def _source_path(raw: str) -> Path:
    p = resolve_existing_path(raw)
    if not p.exists():
        raise FileNotFoundError(f"视频素材不存在: {raw}")
    return p.resolve()


def _probe_video_meta(video_path: Path) -> dict[str, Any]:
    data = ffprobe_json(video_path)
    stream = next((s for s in data.get("streams") or [] if s.get("codec_type") == "video"), None)
    if not stream:
        raise RuntimeError(f"无法读取视频流: {video_path}")
    width = int(stream.get("width") or 1920)
    height = int(stream.get("height") or 1080)
    rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1")
    fps = 30.0
    if "/" in rate:
        a, b = rate.split("/", 1)
        try:
            fps = float(a) / float(b)
        except Exception:
            fps = 30.0
    else:
        try:
            fps = float(rate)
        except Exception:
            fps = 30.0
    return {
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else 30.0,
        "duration": probe_duration(video_path),
    }


_NVENC_RUNTIME_CACHE: bool | None = None


def _ffmpeg_has_encoder(name: str) -> bool:
    try:
        proc = run_ffmpeg([resolve_ffmpeg_bin(), "-hide_banner", "-encoders"], check=False)
        text = f"{proc.stdout}\n{proc.stderr}".lower()
        return name.lower() in text
    except Exception:
        return False


def _h264_nvenc_runtime_available() -> bool:
    global _NVENC_RUNTIME_CACHE
    if _NVENC_RUNTIME_CACHE is not None:
        return _NVENC_RUNTIME_CACHE
    if not _ffmpeg_has_encoder("h264_nvenc"):
        _NVENC_RUNTIME_CACHE = False
        return False
    try:
        proc = run_ffmpeg(
            [
                resolve_ffmpeg_bin(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:d=0.1",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            check=False,
        )
        _NVENC_RUNTIME_CACHE = proc.returncode == 0
    except Exception:
        _NVENC_RUNTIME_CACHE = False
    return bool(_NVENC_RUNTIME_CACHE)


def _select_video_encoder(encoder: str) -> str:
    selected = (encoder or "auto").strip().lower()
    if selected == "auto":
        return "h264_nvenc" if _h264_nvenc_runtime_available() else "libx264"
    return selected


def _video_codec_args(encoder: str) -> list[str]:
    selected = _select_video_encoder(encoder)
    if selected == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]


def _audio_codec_args() -> list[str]:
    return ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def _audio_pts_filter() -> str:
    return "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0"


def _item_duration_from_clips(item: dict[str, Any]) -> float:
    total = 0.0
    for clip in item.get("video_clips") or []:
        try:
            start = float(clip.get("movie_start") or 0.0)
            end = float(clip.get("movie_end") or 0.0)
            total += max(0.0, end - start)
        except Exception:
            pass
    return _round(total)


def _retime_clips_to_audio(item: dict[str, Any], target_duration: float) -> list[dict[str, Any]]:
    clips = [c for c in (item.get("video_clips") or []) if isinstance(c, dict)]
    if not clips:
        return []
    current = _item_duration_from_clips(item)
    if current <= 0 or target_duration <= 0:
        return clips

    adjusted: list[dict[str, Any]] = []
    if target_duration <= current:
        ratio = target_duration / current
        used = 0.0
        for idx, clip in enumerate(clips):
            start = float(clip.get("movie_start") or 0.0)
            end = float(clip.get("movie_end") or 0.0)
            src_dur = max(0.0, end - start)
            take = max(0.0, target_duration - used) if idx == len(clips) - 1 else src_dur * ratio
            if take <= 0.03:
                continue
            row = dict(clip)
            row["movie_start"] = _round(start)
            row["movie_end"] = _round(start + take)
            row["duration"] = _round(take)
            adjusted.append(row)
            used += take
        return adjusted

    adjusted = [dict(c) for c in clips]
    extra = target_duration - current
    last = adjusted[-1]
    last["movie_end"] = _round(float(last.get("movie_end") or 0.0) + extra)
    last["duration"] = _round(float(last.get("duration") or 0.0) + extra)
    return adjusted


async def _synthesize_one(
    item: dict[str, Any],
    audio_dir: Path,
    *,
    voice_id: str,
    speed_ratio: float | None,
    proxy: str | None,
    reuse: bool,
) -> dict[str, Any]:
    if is_original_audio_item(item):
        return original_audio_result(item)

    item_id = str(item.get("item_id") or item.get("segment_id") or uuid.uuid4().hex[:8])
    narration = str(item.get("narration") or "").strip()
    out_path = audio_dir / f"{_safe_name(item_id, 'item')}.mp3"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if not narration:
        duration = max(float(item.get("tts_duration") or 0.0), _item_duration_from_clips(item), 0.3)
        run_ffmpeg(
            [
                resolve_ffmpeg_bin(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=mono",
                "-t",
                f"{duration:.3f}",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "3",
                str(out_path),
            ]
        )
        return {"path": str(out_path), "duration": probe_duration(out_path), "silent": True}

    if reuse and out_path.exists() and out_path.stat().st_size > 0:
        return {"path": str(out_path), "duration": probe_duration(out_path), "reused": True}

    result = await edge_tts_service.synthesize(
        narration,
        voice_id=voice_id,
        speed_ratio=speed_ratio,
        out_path=out_path,
        proxy_override=proxy,
    )
    if not result.get("success"):
        raise RuntimeError(f"Edge TTS 合成失败 {item_id}: {result.get('error') or result.get('message')}")
    duration = float(result.get("duration") or 0.0) or probe_duration(out_path)
    if duration <= 0:
        raise RuntimeError(f"无法读取配音时长: {out_path}")
    return {"path": str(out_path), "duration": _round(duration), "voice_id": voice_id}


async def synthesize_timeline_audio(
    items: list[dict[str, Any]],
    output_dir: Path,
    *,
    voice_id: str,
    speed_ratio: float | None,
    proxy: str | None,
    concurrency: int,
    reuse: bool,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    audio_dir = output_dir / "audio"
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    total = len(items)
    completed = 0
    lock = asyncio.Lock()

    async def run(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        nonlocal completed
        async with sem:
            key = str(item.get("item_id") or item.get("segment_id") or uuid.uuid4().hex[:8])
            result = await _synthesize_one(
                item,
                audio_dir,
                voice_id=voice_id,
                speed_ratio=speed_ratio,
                proxy=proxy,
                reuse=reuse,
            )
            label = "original" if result.get("original_audio") else "tts"
            print(f"[{label}] {key} {result['duration']:.3f}s")
            async with lock:
                completed += 1
                if progress_callback:
                    progress_callback((completed / max(1, total)) * 100.0, f"Synthesized audio {completed}/{total}")
            return key, result

    pairs = await asyncio.gather(*(run(item) for item in items))
    return dict(pairs)


def _cut_video_clip(
    source: Path,
    start: float,
    duration: float,
    out_path: Path,
    *,
    encoder: str,
    keep_audio: bool = False,
    audio_volume: float = 1.0,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        resolve_ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-t",
        f"{max(0.03, duration):.3f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if keep_audio:
        cmd += [
            "-map",
            "0:a?",
            "-vf",
            "setsar=1,format=yuv420p",
            *_video_codec_args(encoder),
        ]
        if audio_volume != 1.0:
            cmd += ["-af", f"volume={max(0.0, float(audio_volume)):.3f},{_audio_pts_filter()}"]
        else:
            cmd += ["-af", _audio_pts_filter()]
        cmd += [*_audio_codec_args(), "-movflags", "+faststart"]
    else:
        cmd += [
            "-an",
            "-vf",
            "setsar=1,format=yuv420p",
            *_video_codec_args(encoder),
            "-movflags",
            "+faststart",
        ]
    cmd.append(str(out_path))
    run_ffmpeg(cmd)
    return out_path


def _concat_videos(paths: list[Path], out_path: Path, *, reencode: bool = False, encoder: str = "auto") -> Path:
    if not paths:
        raise ValueError("没有可拼接的视频片段")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1 and not reencode:
        shutil.copy2(paths[0], out_path)
        return out_path

    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text(
        "\n".join(f"file '{_quote_concat_path(p)}'" for p in paths),
        encoding="utf-8",
    )
    cmd = [
        resolve_ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-fflags",
        "+genpts",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
    ]
    if reencode:
        cmd += [
            "-vf",
            "setsar=1,format=yuv420p",
            *_video_codec_args(encoder),
            "-af",
            _audio_pts_filter(),
            *_audio_codec_args(),
            "-movflags",
            "+faststart",
        ]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(out_path))
    run_ffmpeg(cmd)
    return out_path


def _mux_item_with_audio(visual_path: Path, audio_path: Path, out_path: Path, duration: float, *, encoder: str) -> Path:
    visual_duration = probe_duration(visual_path)
    pad = max(0.0, duration - visual_duration)
    vf = f"tpad=stop_mode=clone:stop_duration={pad:.3f},trim=duration={duration:.3f},setpts=PTS-STARTPTS,setsar=1,format=yuv420p"
    run_ffmpeg(
        [
            resolve_ffmpeg_bin(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(visual_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[0:v]{vf}[v];[1:a]atrim=0:{duration:.3f},{_audio_pts_filter()}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            *_video_codec_args(encoder),
            *_audio_codec_args(),
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    )
    return out_path


def render_video(
    items: list[dict[str, Any]],
    audio_results: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    output_name: str,
    encoder: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    tmp_dir = output_dir / "tmp_video"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    item_paths: list[Path] = []
    selected_encoder = _select_video_encoder(encoder)
    print(f"[video] encoder={selected_encoder}")

    total_items = len(items)
    for item_index, item in enumerate(items, start=1):
        key = str(item.get("item_id") or item.get("segment_id") or f"item_{item_index:03d}")
        audio_mode = audio_mode_for_item(item)
        if audio_mode == "original":
            audio = audio_results.get(key) or original_audio_result(item)
            audio_duration = _item_duration_from_clips(item)
            clips = [c for c in (item.get("video_clips") or []) if isinstance(c, dict)]
        else:
            audio = audio_results[key]
            audio_path = Path(str(audio["path"]))
            audio_duration = float(audio["duration"])
            clips = _retime_clips_to_audio(item, audio_duration)
        if not clips:
            raise ValueError(f"{key} 没有可用 video_clips")

        clip_paths: list[Path] = []
        for clip_index, clip in enumerate(clips, start=1):
            source = _source_path(str(clip.get("source") or ""))
            start = float(clip.get("movie_start") or 0.0)
            end = float(clip.get("movie_end") or 0.0)
            duration = max(0.03, end - start)
            clip_path = tmp_dir / f"{item_index:04d}_{clip_index:03d}.mp4"
            _cut_video_clip(
                source,
                start,
                duration,
                clip_path,
                encoder=selected_encoder,
                keep_audio=audio_mode == "original",
            )
            clip_paths.append(clip_path)

        item_path = tmp_dir / f"{item_index:04d}_narrated.mp4"
        if audio_mode == "original":
            _concat_videos(clip_paths, item_path, reencode=True, encoder=selected_encoder)
        else:
            visual_path = tmp_dir / f"{item_index:04d}_visual.mp4"
            _concat_videos(clip_paths, visual_path)
            _mux_item_with_audio(visual_path, audio_path, item_path, audio_duration, encoder=selected_encoder)
        item_paths.append(item_path)
        print(f"[video] {key} mode={audio_mode} clips={len(clip_paths)} {audio_duration:.3f}s")
        if progress_callback:
            progress_callback((item_index / max(1, total_items)) * 85.0, f"Rendered video segments {item_index}/{total_items}")

    output_path = output_dir / output_name
    if progress_callback:
        progress_callback(90.0, "Merging final video")
    _concat_videos(item_paths, output_path, reencode=True, encoder=selected_encoder)
    if progress_callback:
        progress_callback(100.0, "Video render complete")
    return {
        "output_video": relpath(output_path),
        "segments_count": len(item_paths),
        "duration": _round(probe_duration(output_path)),
    }


def _copy_unique(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst = dst_dir / f"{src.stem}_{uuid.uuid4().hex[:6]}{src.suffix}"
    shutil.copy2(src, dst)
    return dst


def _make_video_material(material_id: str, path: Path) -> dict[str, Any]:
    meta = _probe_video_meta(path)
    return {
        "audio_fade": None,
        "category_id": "",
        "category_name": "local",
        "check_flag": 63487,
        "crop": {
            "upper_left_x": 0,
            "upper_left_y": 0,
            "upper_right_x": 1,
            "upper_right_y": 0,
            "lower_left_x": 0,
            "lower_left_y": 1,
            "lower_right_x": 1,
            "lower_right_y": 1,
        },
        "crop_ratio": "free",
        "crop_scale": 1,
        "duration": _s_to_us(meta["duration"]),
        "height": meta["height"],
        "id": material_id,
        "local_material_id": "",
        "material_id": material_id,
        "material_name": path.name,
        "media_path": "",
        "path": str(path),
        "remote_url": "",
        "type": "video",
        "width": meta["width"],
    }


def _make_audio_material(material_id: str, path: Path, duration_us: int) -> dict[str, Any]:
    return {
        "app_id": 0,
        "category_id": "",
        "category_name": "local",
        "check_flag": 3,
        "copyright_limit_type": "none",
        "duration": duration_us,
        "effect_id": "",
        "formula_id": "",
        "id": material_id,
        "local_material_id": material_id,
        "music_id": material_id,
        "name": path.name,
        "path": str(path),
        "source_platform": 0,
        "type": "extract_music",
        "wave_points": [],
    }


def _empty_materials(video_materials: list[dict[str, Any]], audio_materials: list[dict[str, Any]], speed_materials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "videos": video_materials,
        "speeds": speed_materials,
        "ai_translates": [],
        "audio_balances": [],
        "audio_effects": [],
        "audio_fades": [],
        "audio_track_indexes": [],
        "audios": audio_materials,
        "beats": [],
        "canvases": [],
        "chromas": [],
        "color_curves": [],
        "digital_humans": [],
        "drafts": [],
        "effects": [],
        "flowers": [],
        "green_screens": [],
        "handwrites": [],
        "hsl": [],
        "images": [],
        "log_color_wheels": [],
        "loudnesses": [],
        "manual_deformations": [],
        "material_animations": [],
        "material_colors": [],
        "multi_language_refs": [],
        "placeholders": [],
        "plugin_effects": [],
        "primary_color_wheels": [],
        "realtime_denoises": [],
        "shapes": [],
        "smart_crops": [],
        "smart_relights": [],
        "sound_channel_mappings": [],
        "stickers": [],
        "tail_leaders": [],
        "text_templates": [],
        "texts": [],
        "time_marks": [],
        "transitions": [],
        "video_effects": [],
        "video_trackings": [],
        "vocal_beautifys": [],
        "vocal_separations": [],
        "masks": [],
    }


def _write_extra_draft_files(draft_dir: Path, meta: dict[str, Any]) -> None:
    (draft_dir / "draft_agency_config.json").write_text(
        json.dumps({"is_auto_agency_enabled": False, "is_auto_agency_popup": False, "is_single_agency_mode": False, "marterials": None, "use_converter": False, "video_resolution": meta["height"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (draft_dir / "draft_biz_config.json").write_text(
        json.dumps({"ai_packaging_infos": [], "commercial_music_category_ids": [], "pc_feature_flag": 0, "recognize_tasks": [], "safe_area_type": 0, "template_item_infos": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    (draft_dir / "attachment_pc_common.json").write_text("{}", encoding="utf-8")
    (draft_dir / "attachment_editing.json").write_text(
        json.dumps({"editing_draft": {"is_use_text_to_audio": True, "version": "1.0.0"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    common_dir = draft_dir / "common_attachment"
    common_dir.mkdir(parents=True, exist_ok=True)
    (common_dir / "aigc_aigc_generate.json").write_text(
        json.dumps({"aigc_aigc_generate": {"aigc_generate_segment_list": [], "version": "1.0.0"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (common_dir / "attachment_script_video.json").write_text(
        json.dumps({"script_video": {"attachment_valid": False, "parts": [], "sync_subtitle": False, "version": "1.0.0"}}, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_jianying_draft(
    items: list[dict[str, Any]],
    audio_results: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    draft_name: str,
    target_draft_root: Path | None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = target_draft_root or _default_jianying_draft_root()
    if base_dir is None:
        raise ValueError("未找到本机剪映/CapCut 草稿目录，请通过 --jianying-draft-dir 指定草稿根目录")
    draft_dir = base_dir / f"{_safe_name(draft_name, 'CloneNarration')}_{ts}_{uuid.uuid4().hex[:6]}"
    assets_video_dir = draft_dir / "assets" / "video"
    assets_audio_dir = draft_dir / "assets" / "audio"
    assets_video_dir.mkdir(parents=True, exist_ok=True)
    assets_audio_dir.mkdir(parents=True, exist_ok=True)

    source_map: dict[str, tuple[str, Path]] = {}
    audio_map: dict[str, tuple[str, Path, int]] = {}
    video_materials: list[dict[str, Any]] = []
    audio_materials: list[dict[str, Any]] = []
    speed_materials: list[dict[str, Any]] = []
    video_segments: list[dict[str, Any]] = []
    audio_segments: list[dict[str, Any]] = []
    timeline_cursor_us = 0
    first_video: Path | None = None
    total_items = len(items)

    for item_index, item in enumerate(items, start=1):
        key = str(item.get("item_id") or item.get("segment_id") or f"item_{item_index:03d}")
        audio_mode = audio_mode_for_item(item)
        if audio_mode == "original":
            audio = original_audio_result(item)
            clips = [c for c in (item.get("video_clips") or []) if isinstance(c, dict)]
        else:
            audio = audio_results[key]
            clips = _retime_clips_to_audio(item, float(audio["duration"]))
        item_start_us = timeline_cursor_us
        item_duration_us = 0

        for clip in clips:
            source_src = _source_path(str(clip.get("source") or ""))
            source_key = str(source_src)
            if source_key not in source_map:
                copied = _copy_unique(source_src, assets_video_dir)
                material_id = uuid.uuid4().hex
                source_map[source_key] = (material_id, copied)
                video_materials.append(_make_video_material(material_id, copied))
                if first_video is None:
                    first_video = copied
            material_id, copied_path = source_map[source_key]
            source_duration = probe_duration(copied_path)
            start = max(0.0, float(clip.get("movie_start") or 0.0))
            duration = max(0.03, float(clip.get("movie_end") or 0.0) - start)
            if source_duration > 0:
                duration = min(duration, max(0.03, source_duration - start))
            duration_us = _s_to_us(duration)
            speed_id = uuid.uuid4().hex
            speed_materials.append({"curve_speed": None, "id": speed_id, "mode": 0, "speed": 1, "type": "speed"})
            video_segments.append(
                {
                    "enable_adjust": True,
                    "enable_color_correct_adjust": False,
                    "enable_color_curves": True,
                    "enable_color_match_adjust": False,
                    "enable_color_wheels": True,
                    "enable_lut": True,
                    "enable_smart_color_adjust": False,
                    "last_nonzero_volume": 1,
                    "reverse": False,
                    "track_attribute": 0,
                    "track_render_index": 0,
                    "visible": True,
                    "id": uuid.uuid4().hex,
                    "material_id": material_id,
                    "target_timerange": {"start": timeline_cursor_us, "duration": duration_us},
                    "common_keyframes": [],
                    "keyframe_refs": [],
                    "source_timerange": {"start": _s_to_us(start), "duration": duration_us},
                    "speed": 1,
                    "volume": 1 if audio_mode == "original" else 0,
                    "extra_material_refs": [speed_id],
                    "clip": {"alpha": 1, "flip": {"horizontal": False, "vertical": False}, "rotation": 0, "scale": {"x": 1, "y": 1}, "transform": {"x": 0, "y": 0}},
                    "uniform_scale": {"on": True, "value": 1},
                    "hdr_settings": {"intensity": 1, "mode": 1, "nits": 1000},
                    "render_index": 0,
                }
            )
            timeline_cursor_us += duration_us
            item_duration_us += duration_us

        if audio_mode == "original":
            if progress_callback and (item_index == 1 or item_index == total_items or item_index % 10 == 0):
                progress_callback((item_index / max(1, total_items)) * 70.0, f"Built draft timeline {item_index}/{total_items}")
            continue

        audio_src = Path(str(audio["path"]))
        audio_dur = float(audio["duration"])
        copied_audio = _copy_unique(audio_src, assets_audio_dir)
        audio_key = str(copied_audio)
        audio_duration_us = _s_to_us(audio_dur)
        if audio_key not in audio_map:
            audio_mat_id = uuid.uuid4().hex
            audio_map[audio_key] = (audio_mat_id, copied_audio, audio_duration_us)
            audio_materials.append(_make_audio_material(audio_mat_id, copied_audio, audio_duration_us))
        audio_mat_id, _, _ = audio_map[audio_key]
        audio_speed_id = uuid.uuid4().hex
        speed_materials.append({"curve_speed": None, "id": audio_speed_id, "mode": 0, "speed": 1, "type": "speed"})
        audio_segments.append(
            {
                "enable_adjust": True,
                "last_nonzero_volume": 1,
                "reverse": False,
                "track_attribute": 0,
                "track_render_index": 0,
                "visible": True,
                "id": uuid.uuid4().hex,
                "material_id": audio_mat_id,
                "target_timerange": {"start": item_start_us, "duration": min(audio_duration_us, max(audio_duration_us, item_duration_us))},
                "common_keyframes": [],
                "keyframe_refs": [],
                "source_timerange": {"start": 0, "duration": audio_duration_us},
                "speed": 1,
                "volume": 1,
                "extra_material_refs": [audio_speed_id],
                "is_tone_modify": False,
                "clip": None,
                "hdr_settings": None,
                "render_index": 0,
            }
        )

        if progress_callback and (item_index == 1 or item_index == total_items or item_index % 10 == 0):
            progress_callback((item_index / max(1, total_items)) * 70.0, f"Built draft timeline {item_index}/{total_items}")

    if not first_video:
        raise ValueError("没有可用于草稿的视频素材")
    if progress_callback:
        progress_callback(78.0, "Writing draft metadata")
    video_meta = _probe_video_meta(first_video)
    now_us = _now_us()
    track_video_id = uuid.uuid4().hex
    track_audio_id = uuid.uuid4().hex
    draft_info = {
        "canvas_config": {"width": video_meta["width"], "height": video_meta["height"], "ratio": "original"},
        "color_space": 0,
        "config": {"material_save_mode": 0, "video_mute": False, "record_audio_last_index": 1},
        "cover": {"cover_draft_id": "", "cover_template": "", "sub_type": "local", "type": "image", "web_cover_info": ""},
        "create_time": now_us,
        "duration": timeline_cursor_us,
        "fps": int(round(video_meta["fps"] or 30)),
        "free_render_index_mode_on": False,
        "id": str(uuid.uuid4()).upper(),
        "keyframe_graph_list": [],
        "keyframes": {k: [] for k in ["adjusts", "audios", "effects", "filters", "handwrites", "stickers", "texts", "videos"]},
        "last_modified_platform": {"app_id": 359289, "app_source": "cc", "app_version": "6.5.0", "os": "windows", "os_version": "10"},
        "materials": _empty_materials(video_materials, audio_materials, speed_materials),
        "name": draft_name,
        "new_version": "110.0.0",
        "relationships": [],
        "render_index_track_mode_on": True,
        "source": "default",
        "static_cover_image_path": "",
        "tracks": [
            {"attribute": 0, "flag": 0, "id": track_video_id, "is_default_name": False, "name": "main", "segments": video_segments, "type": "video"},
            {"attribute": 0, "flag": 0, "id": track_audio_id, "is_default_name": False, "name": "audio", "segments": audio_segments, "type": "audio"},
        ],
        "update_time": now_us,
        "version": 360000,
        "platform": {"app_id": 359289, "app_source": "cc", "app_version": "6.5.0", "os": "windows", "os_version": "10"},
    }
    (draft_dir / "draft_info.json").write_text(json.dumps(draft_info, ensure_ascii=False, indent=2), encoding="utf-8")

    draft_meta = {
        "draft_fold_path": str(draft_dir),
        "draft_id": str(uuid.uuid4()).upper(),
        "draft_name": draft_name,
        "draft_root_path": str(draft_dir.parent),
        "tm_draft_create": now_us,
        "tm_draft_modified": now_us,
        "tm_duration": timeline_cursor_us,
        "draft_materials": [{"type": t, "value": []} for t in [0, 1, 2, 3, 6, 7, 8]],
    }
    (draft_dir / "draft_meta_info.json").write_text(json.dumps(draft_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_extra_draft_files(draft_dir, video_meta)

    try:
        run_ffmpeg(
            [
                resolve_ffmpeg_bin(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                "0",
                "-i",
                str(first_video),
                "-frames:v",
                "1",
                str(draft_dir / "draft_cover.jpg"),
            ],
            check=False,
        )
    except Exception:
        pass

    if progress_callback:
        progress_callback(100.0, "Jianying draft complete")

    return {
        "draft_dir": relpath(draft_dir),
        "draft_root": relpath(draft_dir.parent),
        "segments_count": len(items),
        "duration": _round(timeline_cursor_us / 1_000_000),
    }


def write_manifest(output_dir: Path, payload: dict[str, Any]) -> Path:
    return write_json(output_dir / "generate_video_result.json", payload)


def _serialize_audio_results(audio_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key, value in audio_results.items():
        path = str(value.get("path") or "")
        row = {
            "path": relpath(path) if path else "",
            "duration": value.get("duration"),
        }
        for flag in ("original_audio", "silent", "reused"):
            if flag in value:
                row[flag] = value[flag]
        if "voice_id" in value:
            row["voice_id"] = value["voice_id"]
        rows[key] = row
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第 8 步：生成剪映草稿或直接合成视频")
    parser.add_argument("--timeline", default=r".\outputs\7_timeline_composer\final_timeline.json")
    parser.add_argument("--output-dir", default=str(default_output_dir("8_generate_video")))
    parser.add_argument("--mode", choices=["draft", "video", "both"], default="draft")
    parser.add_argument("--voice-id", default=os.getenv("CLONE_EDGE_VOICE") or "zh-CN-XiaoxiaoNeural")
    parser.add_argument("--tts-speed", type=float, default=1.0)
    parser.add_argument("--edge-proxy", default=os.getenv("EDGE_TTS_PROXY") or None)
    parser.add_argument("--tts-concurrency", type=int, default=4)
    parser.add_argument("--reuse-tts", action="store_true")
    parser.add_argument("--draft-name", default="CloneNarration")
    parser.add_argument("--jianying-draft-dir", help="可选：直接输出到剪映草稿根目录")
    parser.add_argument("--video-output-name", default="clone_narration_output.mp4")
    parser.add_argument("--video-encoder", choices=["auto", "libx264", "h264_nvenc"], default=os.getenv("CLONE_VIDEO_ENCODER") or "auto")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = read_json(resolve_existing_path(args.timeline))
    items = _timeline_items(data)

    audio_results = await synthesize_timeline_audio(
        items,
        output_dir,
        voice_id=args.voice_id,
        speed_ratio=args.tts_speed,
        proxy=args.edge_proxy,
        concurrency=args.tts_concurrency,
        reuse=bool(args.reuse_tts),
        progress_callback=lambda percent, message: emit_progress("render", percent * 0.35, message),
    )

    result: dict[str, Any] = {
        "mode": args.mode,
        "timeline": str(args.timeline),
        "audio_dir": relpath(output_dir / "audio"),
        "audio_results": _serialize_audio_results(audio_results),
    }
    if args.mode in {"draft", "both"}:
        draft_root = Path(args.jianying_draft_dir).resolve() if args.jianying_draft_dir else None
        result["jianying_draft"] = generate_jianying_draft(
            items,
            audio_results,
            output_dir,
            draft_name=args.draft_name,
            target_draft_root=draft_root,
            progress_callback=lambda percent, message: emit_progress(
                "render",
                35.0 + percent * (30.0 if args.mode == "both" else 65.0) / 100.0,
                message,
            ),
        )
    if args.mode in {"video", "both"}:
        result["rendered_video"] = render_video(
            items,
            audio_results,
            output_dir,
            output_name=args.video_output_name,
            encoder=args.video_encoder,
            progress_callback=lambda percent, message: emit_progress(
                "render",
                (65.0 if args.mode == "both" else 35.0) + percent * (35.0 if args.mode == "both" else 65.0) / 100.0,
                message,
            ),
        )

    manifest = write_manifest(output_dir, result)
    emit_progress("render", 100, "Video generation complete")
    print(manifest)


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
