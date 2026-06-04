from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(args: list[str]) -> None:
    proc = subprocess.run([sys.executable, *args], cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _stage(step: int, name: str, output_path: Path) -> None:
    print(f"[pipeline] {step}_{name} -> {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="clone_narration_video pipeline")
    parser.add_argument("--ref-video-path", required=True)
    parser.add_argument("--movie-path", required=True)
    parser.add_argument("--subtitle-srt", help="existing subtitle for reference video; omit to run ASR")
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--asr-provider", choices=["bcut", "none"], default="bcut")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--backend", choices=["auto", "transnet", "opencv"], default="auto")
    parser.add_argument("--ai-provider", choices=["custom_openai"], default="custom_openai")
    parser.add_argument("--ai-api-key", default="")
    parser.add_argument("--ai-base-url", default="")
    parser.add_argument("--ai-model", default="")
    parser.add_argument("--ai-temperature", type=float, default=0.7)
    parser.add_argument("--chars-per-second", type=float, default=4.2)
    parser.add_argument("--render-mode", choices=["none", "draft", "video", "both"], default="none")
    parser.add_argument("--edge-voice-id", default="zh-CN-XiaoxiaoNeural")
    parser.add_argument("--edge-tts-speed", type=float, default=1.0)
    parser.add_argument("--jianying-draft-dir", default="")
    parser.add_argument("--video-output-name", default="clone_narration_output.mp4")
    parser.add_argument("--video-encoder", choices=["auto", "libx264", "h264_nvenc"], default="auto")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    ref_dir = output_root / "1_reference_analyzer"
    seg_dir = output_root / "2_narration_segmenter"
    movie_dir = output_root / "3_movie_shot_parser"
    align_dir = output_root / "4_visual_alignment_engine"
    bind_dir = output_root / "5_script_visual_binder"
    rewrite_dir = output_root / "6_rewrite_engine"
    final_dir = output_root / "7_timeline_composer"
    generate_dir = output_root / "8_generate_video"

    ref_cmd = [
        str(ROOT / "1_reference_analyzer" / "run.py"),
        "--ref-video-path",
        args.ref_video_path,
        "--output-dir",
        str(ref_dir),
        "--asr-provider",
        args.asr_provider,
        "--threshold",
        str(args.threshold),
        "--backend",
        args.backend,
    ]
    if args.subtitle_srt:
        ref_cmd += ["--subtitle-srt", args.subtitle_srt]
    _stage(1, "reference_analyzer", ref_dir / "ref_analysis.json")
    _run(ref_cmd)

    _stage(2, "narration_segmenter", seg_dir / "narration_segments.json")
    _run(
        [
            str(ROOT / "2_narration_segmenter" / "run.py"),
            "--input",
            str(ref_dir / "ref_analysis.json"),
            "--output-dir",
            str(seg_dir),
        ]
    )
    _stage(3, "movie_shot_parser", movie_dir / "movie_shots.json")
    _run(
        [
            str(ROOT / "3_movie_shot_parser" / "run.py"),
            "--movie-path",
            args.movie_path,
            "--output-dir",
            str(movie_dir),
            "--threshold",
            str(args.threshold),
            "--backend",
            args.backend,
        ]
    )
    _stage(4, "visual_alignment_engine", align_dir / "ref_to_movie_timeline.json")
    _run(
        [
            str(ROOT / "4_visual_alignment_engine" / "run.py"),
            "--ref-analysis",
            str(ref_dir / "ref_analysis.json"),
            "--movie-shots",
            str(movie_dir / "movie_shots.json"),
            "--output-dir",
            str(align_dir),
        ]
    )
    _stage(5, "script_visual_binder", bind_dir / "script_mapping.json")
    _run(
        [
            str(ROOT / "5_script_visual_binder" / "run.py"),
            "--narration-segments",
            str(seg_dir / "narration_segments.json"),
            "--timeline",
            str(align_dir / "ref_to_movie_timeline.json"),
            "--output-dir",
            str(bind_dir),
        ]
    )

    rewrite_cmd = [
        str(ROOT / "6_rewrite_engine" / "run.py"),
        "--script-mapping",
        str(bind_dir / "script_mapping.json"),
        "--output-dir",
        str(rewrite_dir),
        "--provider",
        args.ai_provider,
        "--temperature",
        str(args.ai_temperature),
    ]
    if args.ai_api_key:
        rewrite_cmd += ["--api-key", args.ai_api_key]
    if args.ai_base_url:
        rewrite_cmd += ["--base-url", args.ai_base_url]
    if args.ai_model:
        rewrite_cmd += ["--model", args.ai_model]
    _stage(6, "rewrite_engine", rewrite_dir / "rewritten_script.json")
    _run(rewrite_cmd)

    _stage(7, "timeline_composer", final_dir / "final_timeline.json")
    _run(
        [
            str(ROOT / "7_timeline_composer" / "run.py"),
            "--rewritten-script",
            str(rewrite_dir / "rewritten_script.json"),
            "--script-mapping",
            str(bind_dir / "script_mapping.json"),
            "--movie-shots",
            str(movie_dir / "movie_shots.json"),
            "--movie-source",
            args.movie_path,
            "--output-dir",
            str(final_dir),
            "--chars-per-second",
            str(args.chars_per_second),
        ]
    )

    if args.render_mode != "none":
        _stage(8, "generate_video", generate_dir / "generate_video_result.json")
        generate_cmd = [
            str(ROOT / "8_generate_video" / "run.py"),
            "--timeline",
            str(final_dir / "final_timeline.json"),
            "--output-dir",
            str(generate_dir),
            "--mode",
            args.render_mode,
            "--voice-id",
            args.edge_voice_id,
            "--tts-speed",
            str(args.edge_tts_speed),
            "--video-output-name",
            args.video_output_name,
            "--video-encoder",
            args.video_encoder,
        ]
        if args.jianying_draft_dir:
            generate_cmd += ["--jianying-draft-dir", args.jianying_draft_dir]
        _run(generate_cmd)
        print(generate_dir / "generate_video_result.json")
    else:
        print(final_dir / "final_timeline.json")


if __name__ == "__main__":
    main()
