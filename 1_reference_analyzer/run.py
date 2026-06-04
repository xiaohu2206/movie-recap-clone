from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.project_paths import default_output_dir
from clone_narration_video.utils.shot_detection import detect_shots
from clone_narration_video.utils.subtitle_tools import copy_or_create_srt, extract_srt_with_bcut


def analyze_reference_video(
    ref_video_path: str | Path,
    *,
    output_dir: str | Path,
    subtitle_srt: str | Path | None = None,
    asr_provider: str = "bcut",
    threshold: float = 0.5,
    backend: str = "auto",
    ref_video_id: str | None = None,
) -> dict[str, Any]:
    video = Path(ref_video_path)
    if not video.exists():
        raise FileNotFoundError(f"参考视频不存在: {video}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_id = ref_video_id or f"ref_{uuid.uuid4().hex[:8]}"
    srt_out = out_dir / "ref_subtitle.srt"
    if subtitle_srt:
        copy_or_create_srt(subtitle_srt, srt_out)
    elif asr_provider == "none":
        copy_or_create_srt(None, srt_out)
    elif asr_provider == "bcut":
        extract_srt_with_bcut(video, srt_out, out_dir)
    else:
        raise ValueError(f"不支持的 ASR provider: {asr_provider}")

    shot_result = detect_shots(
        video,
        shot_prefix="ref_shot",
        keyframe_dir=out_dir / "keyframes",
        threshold=threshold,
        backend=backend,
    )
    result = {
        "ref_video_id": ref_id,
        "ref_video_path": str(video),
        "duration": shot_result["duration"],
        "fps": shot_result["fps"],
        "subtitle_srt": str(srt_out),
        "ref_shots": shot_result["shots"],
        "shot_backend": shot_result["backend"],
    }
    write_json(out_dir / "ref_analysis.json", result)
    return result


def _load_input(args: argparse.Namespace) -> dict[str, Any]:
    data = read_json(args.input) if args.input else {}
    if args.ref_video_path:
        data["ref_video_path"] = args.ref_video_path
    if args.subtitle_srt:
        data["subtitle_srt"] = args.subtitle_srt
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="参考视频解析模块")
    parser.add_argument("--input", help="输入 JSON，包含 ref_video_path，可选 subtitle_srt")
    parser.add_argument("--ref-video-path", help="参考解说视频路径")
    parser.add_argument("--subtitle-srt", help="已有字幕 SRT；传入后跳过 ASR")
    parser.add_argument("--output-dir", default=str(default_output_dir("1_reference_analyzer")))
    parser.add_argument("--asr-provider", choices=["bcut", "none"], default="bcut")
    parser.add_argument("--threshold", type=float, default=0.5, help="TransNetV2 镜头切分阈值")
    parser.add_argument("--backend", choices=["auto", "transnet", "opencv"], default="auto")
    args = parser.parse_args()

    data = _load_input(args)
    ref_video_path = data.get("ref_video_path")
    if not ref_video_path:
        raise SystemExit("缺少 ref_video_path")

    result = analyze_reference_video(
        ref_video_path,
        output_dir=args.output_dir,
        subtitle_srt=data.get("subtitle_srt"),
        asr_provider=args.asr_provider,
        threshold=args.threshold,
        backend=args.backend,
        ref_video_id=data.get("ref_video_id"),
    )
    print(write_json(Path(args.output_dir) / "ref_analysis.json", result))


if __name__ == "__main__":
    main()

