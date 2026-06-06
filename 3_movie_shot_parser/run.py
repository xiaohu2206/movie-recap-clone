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
from clone_narration_video.utils.progress import emit_progress
from clone_narration_video.utils.project_paths import default_output_dir
from clone_narration_video.utils.shot_detection import detect_shots
from clone_narration_video.utils.video_tools import export_shot_clips

# 大写配置-输出分割后的镜头
EXPORT_SHOT_CLIPS = True          # 默认不开启；开启后输出分割后的镜头视频片段
SHOT_CLIPS_DIRNAME = "shot_clips"  # 独立文件夹，用于放置分割后的镜头
SHOT_CLIPS_RATIO = 0.2             # 默认只输出前 20% 的镜头


def parse_movie_shots(
    movie_path: str | Path,
    *,
    output_dir: str | Path,
    threshold: float = 0.5,
    backend: str = "auto",
    keyframe_positions: str | list[float] | None = None,
    sample_fps: float = 0.0,
    max_sample_frames_per_shot: int = 0,
    movie_id: str | None = None,
) -> dict[str, Any]:
    video = Path(movie_path)
    if not video.exists():
        raise FileNotFoundError(f"原电影不存在: {video}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emit_progress("shots", 1, "Starting movie shot detection")
    shot_result = detect_shots(
        video,
        shot_prefix="movie_shot",
        keyframe_dir=out_dir / "keyframes",
        threshold=threshold,
        backend=backend,
        keyframe_positions=keyframe_positions,
        sample_fps=sample_fps,
        max_sample_frames_per_shot=max_sample_frames_per_shot,
        progress_callback=lambda percent, message: emit_progress("shots", percent, message),
    )
    emit_progress("shots", 100, "Movie shot detection complete")
    if EXPORT_SHOT_CLIPS:
        clip_paths = export_shot_clips(
            video,
            shot_result["shots"],
            out_dir / SHOT_CLIPS_DIRNAME,
            id_key="movie_shot_id",
            ratio=SHOT_CLIPS_RATIO,
            progress_callback=lambda i, total, path: emit_progress(
                "shots", 100, f"Exported shot clip {i}/{total}"
            ),
        )
        emit_progress("shots", 100, f"Exported {len(clip_paths)} shot clips")
    result = {
        "movie_id": movie_id or f"movie_{uuid.uuid4().hex[:8]}",
        "movie_path": str(video),
        "duration": shot_result["duration"],
        "fps": shot_result["fps"],
        "movie_shots": shot_result["shots"],
        "shot_backend": shot_result["backend"],
    }
    write_json(out_dir / "movie_shots.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="原电影镜头拆分模块")
    parser.add_argument("--input", help="输入 JSON，包含 movie_path")
    parser.add_argument("--movie-path", help="原电影路径")
    parser.add_argument("--output-dir", default=str(default_output_dir("3_movie_shot_parser")))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--backend", choices=["auto", "transnet", "opencv"], default="auto")
    parser.add_argument("--keyframe-positions", default="0.12,0.5,0.88")
    parser.add_argument("--sample-fps", type=float, default=0.0)
    parser.add_argument("--max-sample-frames-per-shot", type=int, default=0)
    args = parser.parse_args()

    data = read_json(args.input) if args.input else {}
    movie_path = args.movie_path or data.get("movie_path")
    if not movie_path:
        raise SystemExit("缺少 movie_path")
    result = parse_movie_shots(
        movie_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        backend=args.backend,
        keyframe_positions=args.keyframe_positions,
        sample_fps=args.sample_fps,
        max_sample_frames_per_shot=args.max_sample_frames_per_shot,
        movie_id=data.get("movie_id"),
    )
    print(write_json(Path(args.output_dir) / "movie_shots.json", result))


if __name__ == "__main__":
    main()

