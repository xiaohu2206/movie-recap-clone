#!/usr/bin/env python3
"""检查 shot_breakdown.json：相邻镜头时间重叠（<=5s）与 movie_shot_ids 重复。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_shots(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    shots: list[dict] = []
    for shot in data.get("shots", []):
        shots.append(
            {
                "item_id": shot.get("item_id"),
                "clip_id": shot.get("clip_id"),
                "movie_start": float(shot.get("movie_start", 0.0)),
                "movie_end": float(shot.get("movie_end", 0.0)),
                "movie_shot_ids": list(shot.get("movie_shot_ids") or []),
            }
        )
    return shots


def check_time_overlap(shots: list[dict], max_overlap: float = 5.0, epsilon: float = 1e-6) -> list[dict]:
    """前镜头 movie_end 超过后镜头 movie_start，且超出在 max_overlap 秒内。"""
    issues = []
    for i in range(len(shots) - 1):
        prev, nxt = shots[i], shots[i + 1]
        overlap = prev["movie_end"] - nxt["movie_start"]
        if epsilon < overlap <= max_overlap:
            issues.append({"prev": prev, "next": nxt, "overlap": overlap})
    return issues


def check_duplicate_movie_shot_ids(shots: list[dict]) -> list[dict]:
    """相邻镜头 movie_shot_ids 有交集（重复镜头）。"""
    issues = []
    for i in range(len(shots) - 1):
        prev, nxt = shots[i], shots[i + 1]
        duplicated = sorted(set(prev["movie_shot_ids"]) & set(nxt["movie_shot_ids"]))
        if duplicated:
            issues.append({"prev": prev, "next": nxt, "duplicated": duplicated})
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 shot_breakdown.json 相邻镜头异常")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="outputs/8_generate_video/shot_breakdown.json",
        help="shot_breakdown.json 路径",
    )
    parser.add_argument("--max-overlap", type=float, default=5.0, help="时间重叠上限（秒，默认 5）")
    args = parser.parse_args()

    path = Path(args.json_path)
    if not path.exists():
        raise SystemExit(f"文件不存在: {path}")

    shots = load_shots(path)
    print(f"已加载: {path}  镜头总数: {len(shots)}\n")

    overlap_issues = check_time_overlap(shots, args.max_overlap)
    print(f"【检查1】相邻镜头时间重叠（0 < 超出 <= {args.max_overlap:g}s）：共 {len(overlap_issues)} 处")
    for it in overlap_issues:
        p, n = it["prev"], it["next"]
        print(
            f"  - {p['item_id']}/{p['clip_id']}(end={p['movie_end']:.3f}) -> "
            f"{n['item_id']}/{n['clip_id']}(start={n['movie_start']:.3f})  超出 {it['overlap']:.3f}s"
        )

    print()
    dup_issues = check_duplicate_movie_shot_ids(shots)
    print(f"【检查2】相邻镜头 movie_shot_ids 重复：共 {len(dup_issues)} 处")
    for it in dup_issues:
        p, n = it["prev"], it["next"]
        print(f"  - {p['item_id']}/{p['clip_id']} 与 {n['item_id']}/{n['clip_id']}  重复: {it['duplicated']}")


if __name__ == "__main__":
    main()
