from __future__ import annotations

from pathlib import Path
from typing import Any

from clone_narration_video.utils.json_io import write_json


ACCEPTED_MATCH_STATUSES = {"matched", "matched_low_confidence", "inferred_by_neighbors"}


def score_gap(chosen: dict[str, Any] | None, candidates: list[dict[str, Any]]) -> float:
    if not chosen or not candidates:
        return 0.0
    chosen_id = str(chosen.get("movie_shot_id") or "")
    alternatives = [row for row in candidates if str(row.get("movie_shot_id") or "") != chosen_id]
    if not alternatives:
        return round(float(chosen.get("visual_score") or 0.0), 4)
    second = max(alternatives, key=lambda row: float(row.get("visual_score") or 0.0))
    return round(float(chosen.get("visual_score") or 0.0) - float(second.get("visual_score") or 0.0), 4)


def confidence(
    visual_score: float,
    gap: float,
    *,
    path_continuous: bool,
) -> str:
    if visual_score >= 0.82 and gap >= 0.08 and path_continuous:
        return "high"
    if visual_score >= 0.65 or (visual_score >= 0.45 and path_continuous):
        return "medium"
    return "low"


def status_for_match(candidate: dict[str, Any] | None, *, min_score: float, confidence_value: str) -> str:
    if not candidate:
        return "unmatched"
    visual_score = float(candidate.get("visual_score") or 0.0)
    if visual_score < min_score:
        return "needs_review"
    if confidence_value == "low":
        return "matched_low_confidence"
    return "matched"


def accepted_reason(
    candidate: dict[str, Any] | None,
    *,
    min_score: float,
    confidence_value: str,
    gap: float,
    manual_override: bool = False,
) -> str:
    if manual_override:
        return "manual_override"
    if not candidate:
        return "no_candidate"
    visual_score = float(candidate.get("visual_score") or 0.0)
    diagnostics = candidate.get("diagnostics") or {}
    if visual_score < min_score:
        return "visual_score_below_min_score"
    if confidence_value == "high":
        return "visual_score_high_and_path_continuous"
    if bool(diagnostics.get("boosted_by_continuity")):
        return "path_continuity_promoted_candidate"
    if gap < 0.03:
        return "candidate_scores_too_close"
    if float(diagnostics.get("path_penalty") or 0.0) > 0.0:
        return "accepted_with_path_penalty"
    return "visual_score_accepted"


def format_candidates(
    candidates: list[dict[str, Any]],
    *,
    top_n: int,
    selected_movie_shot_id: str | None,
) -> list[dict[str, Any]]:
    rows = []
    ordered = sorted(candidates, key=lambda row: float(row.get("visual_score") or 0.0), reverse=True)
    for index, candidate in enumerate(ordered[: max(0, int(top_n))], start=1):
        detail = candidate.get("detail") or {}
        rows.append(
            {
                "movie_shot_id": candidate.get("movie_shot_id"),
                "movie_start": candidate.get("movie_start"),
                "movie_end": candidate.get("movie_end"),
                "visual_score": round(float(candidate.get("visual_score") or 0.0), 4),
                "recall_score": round(float(candidate.get("recall_score") or 0.0), 4),
                "final_score": round(float(candidate.get("final_score") or 0.0), 4),
                "coarse_visual_score": round(float(candidate.get("coarse_visual_score") or candidate.get("visual_score") or 0.0), 4),
                "refinement_score": round(float(candidate.get("refinement_score") or 0.0), 4),
                "visual_rank": int(candidate.get("visual_rank") or index),
                "final_rank": candidate.get("final_rank"),
                "selected": str(candidate.get("movie_shot_id") or "") == str(selected_movie_shot_id or ""),
                "refinement": candidate.get("refinement") or {},
                "detail": {
                    "score": detail.get("score", 0.0),
                    "hash": detail.get("hash", 0.0),
                    "hist": detail.get("hist", 0.0),
                    "orb": detail.get("orb", 0.0),
                    "best_pair": detail.get("best_pair"),
                    "top_pairs": detail.get("top_pairs") or [],
                },
            }
        )
    return rows


def build_timeline_item(
    ref: dict[str, Any],
    candidate: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    *,
    top_n: int,
    min_score: float,
    match_type: str,
    manual_override: bool = False,
) -> dict[str, Any]:
    gap = score_gap(candidate, candidates)
    path_info = (candidate or {}).get("diagnostics") or {}
    confidence_value = confidence(
        float((candidate or {}).get("visual_score") or 0.0),
        gap,
        path_continuous=bool(path_info.get("path_continuous", False)),
    )
    status = status_for_match(candidate, min_score=min_score, confidence_value=confidence_value)
    if manual_override and candidate:
        status = "matched"
    reason = accepted_reason(
        candidate,
        min_score=min_score,
        confidence_value=confidence_value,
        gap=gap,
        manual_override=manual_override,
    )
    selected_id = str((candidate or {}).get("movie_shot_id") or "")
    refinement_info = (candidate or {}).get("refinement") or {"enabled": False}
    path_diag = path_info.get("path") or {
        "anchor": False,
        "segment_index": None,
        "jump_allowed": False,
        "skip_state": candidate is None,
    }
    return {
        "ref_shot_id": str(ref.get("ref_shot_id") or ref.get("shot_id") or ""),
        "ref_start": float(ref.get("start") or 0.0),
        "ref_end": float(ref.get("end") or 0.0),
        "movie_start": candidate.get("movie_start") if candidate else None,
        "movie_end": candidate.get("movie_end") if candidate else None,
        "movie_shot_ids": [selected_id] if candidate and selected_id else [],
        "match_score": round(float((candidate or {}).get("visual_score") or 0.0), 4),
        "final_score": round(float((candidate or {}).get("final_score") or 0.0), 4),
        "match_type": "manual_override" if manual_override else match_type,
        "confidence": confidence_value,
        "status": status,
        "diagnostics": {
            "visual_rank": (candidate or {}).get("visual_rank"),
            "final_rank": (candidate or {}).get("final_rank"),
            "score_gap": gap,
            "time_delta": path_info.get("time_delta", 0.0),
            "path_penalty": path_info.get("path_penalty", 0.0),
            "continuity_score": path_info.get("continuity_score", 0.0),
            "transition_score": path_info.get("transition_score", 0.0),
            "path_continuous": bool(path_info.get("path_continuous", False)),
            "boosted_by_continuity": bool(path_info.get("boosted_by_continuity", False)),
            "accepted_reason": reason,
            "refinement": refinement_info,
            "path": path_diag,
        },
        "candidates": format_candidates(candidates, top_n=top_n, selected_movie_shot_id=selected_id),
    }


def write_low_confidence_report(output_dir: str | Path, timeline: list[dict[str, Any]]) -> Path:
    rows = [
        {
            "ref_shot_id": item.get("ref_shot_id"),
            "status": item.get("status"),
            "confidence": item.get("confidence"),
            "match_score": item.get("match_score"),
            "movie_shot_ids": item.get("movie_shot_ids") or [],
            "diagnostics": item.get("diagnostics") or {},
        }
        for item in timeline
        if item.get("confidence") == "low" or item.get("status") not in {"matched"}
    ]
    return write_json(Path(output_dir) / "low_confidence_report.json", {"items": rows})
