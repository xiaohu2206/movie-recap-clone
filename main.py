from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parent


def _stage_python() -> Path:
    for rel in (Path("python") / "python.exe", Path(".venv") / "Scripts" / "python.exe"):
        candidate = ROOT / rel
        if candidate.exists():
            return candidate
    return Path(sys.executable)


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _mount_log(log_file: str) -> tuple[TextIO, TextIO, TextIO] | None:
    if not log_file:
        return None
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, file)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, file)  # type: ignore[assignment]
    print(f"[pipeline] log mounted -> {path}", flush=True)
    return file, original_stdout, original_stderr


def _pipe_stream(stream: TextIO | None, target: TextIO) -> None:
    if stream is None:
        return
    for line in iter(stream.readline, ""):
        target.write(line)
        target.flush()
    stream.close()


def _run(args: list[str]) -> None:
    proc = subprocess.Popen(
        [str(_stage_python()), *args],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    stdout_thread = threading.Thread(target=_pipe_stream, args=(proc.stdout, sys.stdout), daemon=True)
    stderr_thread = threading.Thread(target=_pipe_stream, args=(proc.stderr, sys.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    proc.wait()
    stdout_thread.join()
    stderr_thread.join()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _stage(step: int, name: str, output_path: Path) -> None:
    print(f"[pipeline] {step}_{name} -> {output_path}", flush=True)


def _stage_skipped(step: int, name: str, output_path: Path) -> None:
    print(f"[pipeline] {step}_{name} skipped -> {output_path}", flush=True)


def _has_valid_output(output_path: Path) -> bool:
    if not output_path.exists() or not output_path.is_file():
        return False
    if output_path.suffix.lower() != ".json":
        return output_path.stat().st_size > 0
    try:
        with output_path.open("r", encoding="utf-8") as file:
            json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _run_stage(step: int, name: str, output_path: Path, command: list[str], resume: bool) -> None:
    if resume and _has_valid_output(output_path):
        _stage_skipped(step, name, output_path)
        return
    _stage(step, name, output_path)
    _run(command)


def _clean_stage_outputs(output_root: Path) -> None:
    for name in (
        "1_reference_analyzer",
        "2_narration_segmenter",
        "3_movie_shot_parser",
        "4_visual_alignment_engine",
        "5_script_visual_binder",
        "6_rewrite_engine",
        "7_timeline_composer",
        "8_generate_video",
    ):
        target = output_root / name
        if target.exists():
            shutil.rmtree(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="clone_narration_video pipeline")
    parser.add_argument("--ref-video-path", required=True)
    parser.add_argument("--movie-path", required=True)
    parser.add_argument("--subtitle-srt", help="existing subtitle for reference video; omit to run ASR")
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--log-file", default="", help="append pipeline stdout/stderr to this file")
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
    parser.add_argument("--resume", action="store_true", help="skip stages whose output already exists")
    parser.add_argument("--restart", action="store_true", help="clear stage outputs before running")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    log_file = args.log_file or str(output_root / "logs" / "pipeline.log")
    log_mount = _mount_log(log_file)
    if args.restart:
        _clean_stage_outputs(output_root)

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
    _run_stage(1, "reference_analyzer", ref_dir / "ref_analysis.json", ref_cmd, args.resume)

    _run_stage(
        2,
        "narration_segmenter",
        seg_dir / "narration_segments.json",
        [
            str(ROOT / "2_narration_segmenter" / "run.py"),
            "--input",
            str(ref_dir / "ref_analysis.json"),
            "--output-dir",
            str(seg_dir),
        ],
        args.resume,
    )
    _run_stage(
        3,
        "movie_shot_parser",
        movie_dir / "movie_shots.json",
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
        ],
        args.resume,
    )
    _run_stage(
        4,
        "visual_alignment_engine",
        align_dir / "ref_to_movie_timeline.json",
        [
            str(ROOT / "4_visual_alignment_engine" / "run.py"),
            "--ref-analysis",
            str(ref_dir / "ref_analysis.json"),
            "--movie-shots",
            str(movie_dir / "movie_shots.json"),
            "--output-dir",
            str(align_dir),
        ],
        args.resume,
    )
    _run_stage(
        5,
        "script_visual_binder",
        bind_dir / "script_mapping.json",
        [
            str(ROOT / "5_script_visual_binder" / "run.py"),
            "--narration-segments",
            str(seg_dir / "narration_segments.json"),
            "--timeline",
            str(align_dir / "ref_to_movie_timeline.json"),
            "--output-dir",
            str(bind_dir),
        ],
        args.resume,
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
    _run_stage(6, "rewrite_engine", rewrite_dir / "rewritten_script.json", rewrite_cmd, args.resume)

    _run_stage(
        7,
        "timeline_composer",
        final_dir / "final_timeline.json",
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
        ],
        args.resume,
    )

    if args.render_mode != "none":
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
        if args.resume:
            generate_cmd += ["--reuse-tts"]
        _run_stage(8, "generate_video", generate_dir / "generate_video_result.json", generate_cmd, args.resume)
        print(generate_dir / "generate_video_result.json")
    else:
        print(final_dir / "final_timeline.json")

    if log_mount:
        log_handle, original_stdout, original_stderr = log_mount
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
