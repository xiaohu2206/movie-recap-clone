#!/usr/bin/env python3
"""检查 ref_audio_rebuild_timeline.json：相邻镜头时间重叠（<=5s）与 source_ref_shot_id 重复。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_clips(path: Path) -> list[dict]:
    """将 final_timeline 中的 video_clips 按时间线顺序展开为扁平列表。"""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    clips: list[dict] = []
    for item in data.get("final_timeline", []):
        for clip in item.get("video_clips", []):
            clips.append(
                {
                    "item_id": item.get("item_id"),
                    "clip_id": clip.get("clip_id"),
                    "source": clip.get("source", ""),
                    "movie_start": float(clip.get("movie_start", 0.0)),
                    "movie_end": float(clip.get("movie_end", 0.0)),
                    "source_ref_shot_id": clip.get("source_ref_shot_id"),
                    "is_fallback": bool(clip.get("is_fallback", False)),
                }
            )
    return clips


def check_time_overlap(clips: list[dict], max_overlap: float = 5.0, epsilon: float = 1e-6) -> list[dict]:
    """前镜头 movie_end 超过后镜头 movie_start，且超出在 max_overlap 秒内。

    仅比较来自电影源的非 fallback 镜头（fallback 使用 ref.mp4，时间轴不同）。
    """
    movie_clips = [c for c in clips if not c["is_fallback"]]
    issues = []
    for i in range(len(movie_clips) - 1):
        prev, nxt = movie_clips[i], movie_clips[i + 1]
        overlap = prev["movie_end"] - nxt["movie_start"]
        if epsilon < overlap <= max_overlap:
            issues.append(
                {
                    "prev": prev,
                    "next": nxt,
                    "overlap": overlap,
                }
            )
    return issues


def check_duplicate_ref_shot(clips: list[dict]) -> list[dict]:
    """相邻镜头 source_ref_shot_id 相同（重复镜头）。"""
    issues = []
    for i in range(len(clips) - 1):
        prev, nxt = clips[i], clips[i + 1]
        if prev["source_ref_shot_id"] and prev["source_ref_shot_id"] == nxt["source_ref_shot_id"]:
            issues.append({"prev": prev, "next": nxt})
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 ref_audio_rebuild_timeline.json 相邻镜头异常")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="outputs/4.1_ref_audio_rebuild_composer/ref_audio_rebuild_timeline.json",
        help="ref_audio_rebuild_timeline.json 路径",
    )
    parser.add_argument("--max-overlap", type=float, default=5.0, help="时间重叠上限（秒，默认 5）")
    args = parser.parse_args()

    path = Path(args.json_path)
    if not path.exists():
        raise SystemExit(f"文件不存在: {path}")

    clips = load_clips(path)
    print(f"已加载: {path}  镜头总数: {len(clips)}\n")

    overlap_issues = check_time_overlap(clips, args.max_overlap)
    print(f"【检查1】相邻镜头时间重叠（0 < 超出 <= {args.max_overlap:g}s）：共 {len(overlap_issues)} 处")
    for it in overlap_issues:
        p, n = it["prev"], it["next"]
        print(
            f"  - {p['item_id']}/{p['clip_id']}(end={p['movie_end']:.3f}) -> "
            f"{n['item_id']}/{n['clip_id']}(start={n['movie_start']:.3f})  超出 {it['overlap']:.3f}s"
        )

    print()
    dup_issues = check_duplicate_ref_shot(clips)
    print(f"【检查2】相邻镜头 source_ref_shot_id 重复：共 {len(dup_issues)} 处")
    for it in dup_issues:
        p, n = it["prev"], it["next"]
        print(f"  - {p['item_id']}/{p['clip_id']} 与 {n['item_id']}/{n['clip_id']}  重复: {p['source_ref_shot_id']}")


if __name__ == "__main__":
    main()
