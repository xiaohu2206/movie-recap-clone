from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .ffmpeg_utils import extract_audio_mp3
from .asr.asr_utils import utterances_to_srt


_TS = r"(\d{2}:\d{2}:\d{2}[,.]\d{3})"


def parse_srt_time(value: str) -> float:
    raw = value.strip().replace(".", ",")
    h, m, rest = raw.split(":")
    s, ms = rest.split(",", 1)
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms[:3].ljust(3, "0")) / 1000.0


def format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path: str | Path) -> list[dict[str, Any]]:
    content = Path(path).read_text(encoding="utf-8-sig")
    text = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    compressed = re.compile(rf"^\[{_TS}\s*-\s*{_TS}\]\s*(.+)$")
    entries = []
    for idx, line in enumerate([x.strip() for x in text.splitlines() if x.strip()], start=1):
        m = compressed.match(line)
        if m:
            start, end, body = m.groups()
            entries.append({"index": idx, "start": parse_srt_time(start), "end": parse_srt_time(end), "text": body.strip()})
    if entries:
        return entries

    blocks = re.split(r"\n\s*\n", text)
    for fallback_idx, block in enumerate(blocks, start=1):
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        timing_i = next((i for i, x in enumerate(lines) if "-->" in x), None)
        if timing_i is None:
            continue
        start_raw, end_raw = [x.strip().split()[0] for x in lines[timing_i].split("-->", 1)]
        body = " ".join(lines[timing_i + 1 :]).strip()
        if body:
            entries.append(
                {
                    "index": fallback_idx,
                    "start": parse_srt_time(start_raw),
                    "end": parse_srt_time(end_raw),
                    "text": body,
                }
            )
    return entries


def write_srt(path: str | Path, entries: list[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for idx, item in enumerate(entries, start=1):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        lines.extend(
            [
                str(idx),
                f"{format_srt_time(float(item.get('start') or 0.0))} --> {format_srt_time(float(item.get('end') or 0.0))}",
                text,
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def copy_or_create_srt(source: str | Path | None, target: str | Path) -> Path:
    out = Path(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    if source:
        shutil.copyfile(str(source), str(out))
    else:
        out.write_text("", encoding="utf-8")
    return out


def extract_srt_with_bcut(video_path: str | Path, out_srt: str | Path, work_dir: str | Path) -> Path:
    from .asr.asr_bcut import BcutASR

    audio_path = Path(work_dir) / "asr_audio.mp3"
    extract_audio_mp3(video_path, audio_path)
    asr = BcutASR(str(audio_path), use_cache=True)
    data = asr.run()
    utterances = data.get("utterances") if isinstance(data, dict) else None
    if not isinstance(utterances, list) or not utterances:
        raise RuntimeError("Bcut ASR 未返回有效 utterances")
    out = Path(out_srt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(utterances_to_srt(utterances), encoding="utf-8")
    return out

