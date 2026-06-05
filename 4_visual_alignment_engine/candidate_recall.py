from __future__ import annotations

from typing import Any

from clone_narration_video.utils.visual_features import compare_shot_features


def _id(item: dict[str, Any], key: str) -> str:
    return str(item.get(key) or item.get("shot_id") or "")


def build_candidates(
    ref_feature: dict[str, Any],
    movie_features: list[dict[str, Any]],
    movie_shots: list[dict[str, Any]],
    *,
    recall_top_k: int = 80,
    rerank_top_k: int = 20,
) -> list[dict[str, Any]]:
    recall_rows = []
    for movie_index, movie_feature in enumerate(movie_features):
        score = compare_shot_features(ref_feature, movie_feature, include_orb=False)
        recall_rows.append(
            {
                "movie_index": movie_index,
                "recall_score": score["lightweight_score"],
            }
        )

    recall_rows.sort(key=lambda row: row["recall_score"], reverse=True)
    recalled = recall_rows[: max(1, int(recall_top_k))]

    candidates: list[dict[str, Any]] = []
    for row in recalled:
        movie_index = int(row["movie_index"])
        movie = movie_shots[movie_index]
        visual = compare_shot_features(ref_feature, movie_features[movie_index], include_orb=True)
        candidates.append(
            {
                "movie_index": movie_index,
                "movie_shot_id": _id(movie, "movie_shot_id"),
                "movie_start": float(movie.get("start") or 0.0),
                "movie_end": float(movie.get("end") or 0.0),
                "visual_score": float(visual["score"]),
                "recall_score": float(row["recall_score"]),
                "detail": visual,
            }
        )

    candidates.sort(key=lambda row: row["visual_score"], reverse=True)
    candidates = candidates[: max(1, int(rerank_top_k))]
    for rank, candidate in enumerate(candidates, start=1):
        candidate["visual_rank"] = rank
    return candidates
