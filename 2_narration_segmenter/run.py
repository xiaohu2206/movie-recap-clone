from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.project_paths import default_output_dir
from clone_narration_video.utils.subtitle_tools import parse_srt


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _shot_id(shot: dict[str, Any]) -> str:
    return str(shot.get("ref_shot_id") or shot.get("shot_id") or "")


def _subtitle_shots(sub: dict[str, Any], shots: list[dict[str, Any]]) -> list[str]:
    ids = []
    ss = float(sub["start"])
    se = float(sub["end"])
    for shot in shots:
        sid = _shot_id(shot)
        if sid and _overlap(ss, se, float(shot.get("start") or 0.0), float(shot.get("end") or 0.0)) > 0:
            ids.append(sid)
    return ids


def segment_narration(
    subtitle_srt: str | Path,
    ref_shots: list[dict[str, Any]],
    *,
    pause_threshold: float = 1.2,
    max_shots: int = 6,
) -> dict[str, Any]:
    subtitles = parse_srt(subtitle_srt)
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_shots: list[str] = []

    def flush() -> None:
        nonlocal current, current_shots
        if not current:
            return
        start = min(float(x["start"]) for x in current)
        end = max(float(x["end"]) for x in current)
        seg_id = f"seg_{len(segments) + 1:03d}"
        segments.append(
            {
                "segment_id": seg_id,
                "ref_start": round(start, 3),
                "ref_end": round(end, 3),
                "text": "".join(str(x.get("text") or "").strip() for x in current),
                "ref_shot_ids": current_shots,
                "segment_type": "narration",
            }
        )
        current = []
        current_shots = []

    prev_end: float | None = None
    for sub in subtitles:
        sub_shots = _subtitle_shots(sub, ref_shots)
        combined = list(dict.fromkeys([*current_shots, *sub_shots]))
        pause = (float(sub["start"]) - prev_end) if prev_end is not None else 0.0
        if current and (pause > pause_threshold or len(combined) > max_shots):
            flush()
            combined = sub_shots
        current.append(sub)
        current_shots = list(dict.fromkeys([*current_shots, *combined]))
        prev_end = float(sub["end"])
    flush()

    return {"narration_segments": segments, "subtitle_count": len(subtitles), "ref_shot_count": len(ref_shots)}


def main() -> None:
    parser = argparse.ArgumentParser(description="解说段落切分模块")
    parser.add_argument("--input", help="参考解析 JSON，默认使用其中 subtitle_srt/ref_shots")
    parser.add_argument("--subtitle-srt", help="字幕 SRT")
    parser.add_argument("--ref-analysis", help="参考解析 JSON")
    parser.add_argument("--output-dir", default=str(default_output_dir("2_narration_segmenter")))
    parser.add_argument("--pause-threshold", type=float, default=1.2)
    parser.add_argument("--max-shots", type=int, default=6)
    args = parser.parse_args()

    data = read_json(args.input or args.ref_analysis) if (args.input or args.ref_analysis) else {}
    subtitle_srt = args.subtitle_srt or data.get("subtitle_srt")
    ref_shots = data.get("ref_shots") or []
    if not subtitle_srt:
        raise SystemExit("缺少 subtitle_srt")
    if not isinstance(ref_shots, list):
        raise SystemExit("缺少 ref_shots")

    result = segment_narration(
        subtitle_srt,
        ref_shots,
        pause_threshold=args.pause_threshold,
        max_shots=args.max_shots,
    )
    out = write_json(Path(args.output_dir) / "narration_segments.json", result)
    print(out)


if __name__ == "__main__":
    main()

