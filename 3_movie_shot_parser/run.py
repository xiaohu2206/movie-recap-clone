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


def parse_movie_shots(
    movie_path: str | Path,
    *,
    output_dir: str | Path,
    threshold: float = 0.5,
    backend: str = "auto",
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
        progress_callback=lambda percent, message: emit_progress("shots", percent, message),
    )
    emit_progress("shots", 100, "Movie shot detection complete")
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
        movie_id=data.get("movie_id"),
    )
    print(write_json(Path(args.output_dir) / "movie_shots.json", result))


if __name__ == "__main__":
    main()

