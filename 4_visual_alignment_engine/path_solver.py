from __future__ import annotations

import math
from typing import Any


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def transition_metrics(
    prev: dict[str, Any] | None,
    current: dict[str, Any],
    ref_gap: float,
    *,
    backward_tolerance: float = 0.5,
) -> dict[str, float | bool | str]:
    if prev is None:
        return {
            "transition_score": 0.5,
            "continuity_score": 0.5,
            "time_delta": 0.0,
            "path_penalty": 0.0,
            "backward_penalty": 0.0,
            "jump_penalty": 0.0,
            "repeat_penalty": 0.0,
            "path_continuous": True,
            "boosted_by_continuity": False,
        }

    expected_start = float(prev.get("movie_end") or 0.0) + max(0.0, ref_gap)
    current_start = float(current.get("movie_start") or 0.0)
    time_delta = current_start - expected_start
    abs_delta = abs(time_delta)
    continuity_score = math.exp(-abs_delta / 8.0)

    backward_penalty = 0.0
    if current_start < float(prev.get("movie_start") or 0.0) - backward_tolerance:
        backward_penalty = min(0.65, (float(prev.get("movie_start") or 0.0) - current_start) / 25.0)

    jump_penalty = 0.0
    jump_threshold = 60.0 if bool(prev.get("path_anchor")) or bool(current.get("path_anchor")) else 20.0
    if abs_delta > jump_threshold:
        jump_penalty = min(0.28, (abs_delta - jump_threshold) / 100.0)
        if float(current.get("visual_score") or 0.0) >= 0.82:
            jump_penalty *= 0.35
        if bool(prev.get("path_anchor")) or bool(current.get("path_anchor")):
            jump_penalty *= 0.15

    repeat_penalty = 0.0
    if str(prev.get("movie_shot_id") or "") and str(prev.get("movie_shot_id")) == str(current.get("movie_shot_id")):
        repeat_penalty = 0.18

    path_penalty = backward_penalty + jump_penalty + repeat_penalty
    transition_score = _clip01(continuity_score - path_penalty)
    return {
        "transition_score": round(float(transition_score), 4),
        "continuity_score": round(float(continuity_score), 4),
        "time_delta": round(float(time_delta), 4),
        "path_penalty": round(float(path_penalty), 4),
        "backward_penalty": round(float(backward_penalty), 4),
        "jump_penalty": round(float(jump_penalty), 4),
        "repeat_penalty": round(float(repeat_penalty), 4),
        "path_continuous": backward_penalty == 0.0 and abs_delta <= (30.0 if bool(prev.get("path_anchor")) or bool(current.get("path_anchor")) else 12.0),
        "boosted_by_continuity": continuity_score >= float(current.get("visual_score") or 0.0) + 0.08,
    }


def _node_score(
    candidate: dict[str, Any],
    metrics: dict[str, float | bool | str],
    *,
    visual_weight: float,
    transition_weight: float,
) -> float:
    visual_score = float(candidate.get("visual_score") or 0.0)
    transition_score = float(metrics.get("transition_score") or 0.0)
    penalty = float(metrics.get("path_penalty") or 0.0)
    return _clip01(visual_score * visual_weight + transition_score * transition_weight - penalty * 0.25)


def solve_global_path(
    ref_shots: list[dict[str, Any]],
    candidates_by_ref: list[list[dict[str, Any]]],
    *,
    visual_weight: float = 0.72,
    transition_weight: float = 0.28,
) -> list[dict[str, Any] | None]:
    if not candidates_by_ref:
        return []

    dp: list[list[dict[str, Any]]] = []
    for ref_index, candidates in enumerate(candidates_by_ref):
        row: list[dict[str, Any]] = []
        if not candidates:
            dp.append(row)
            continue

        ref_gap = 0.0
        if ref_index > 0:
            ref_gap = float(ref_shots[ref_index].get("start") or 0.0) - float(ref_shots[ref_index - 1].get("end") or 0.0)

        for candidate_index, candidate in enumerate(candidates):
            if ref_index == 0 or not dp[ref_index - 1]:
                metrics = transition_metrics(None, candidate, ref_gap)
                node_score = _node_score(
                    candidate,
                    metrics,
                    visual_weight=visual_weight,
                    transition_weight=transition_weight,
                )
                row.append(
                    {
                        "score": node_score,
                        "node_score": node_score,
                        "prev_index": None,
                        "metrics": metrics,
                    }
                )
                continue

            best_prev: dict[str, Any] | None = None
            for prev_index, prev_state in enumerate(dp[ref_index - 1]):
                prev_candidate = candidates_by_ref[ref_index - 1][prev_index]
                metrics = transition_metrics(prev_candidate, candidate, ref_gap)
                node_score = _node_score(
                    candidate,
                    metrics,
                    visual_weight=visual_weight,
                    transition_weight=transition_weight,
                )
                total_score = float(prev_state["score"]) + node_score
                if best_prev is None or total_score > float(best_prev["score"]):
                    best_prev = {
                        "score": total_score,
                        "node_score": node_score,
                        "prev_index": prev_index,
                        "metrics": metrics,
                    }
            row.append(best_prev or {"score": 0.0, "node_score": 0.0, "prev_index": None, "metrics": {}})

        for rank, state in enumerate(sorted(row, key=lambda item: item["node_score"], reverse=True), start=1):
            state["final_rank"] = rank
        for candidate_index, state in enumerate(row):
            candidates[candidate_index]["final_score"] = round(float(state.get("node_score") or 0.0), 4)
            candidates[candidate_index]["path_score"] = round(float(state.get("score") or 0.0), 4)
            candidates[candidate_index]["final_rank"] = int(state.get("final_rank") or candidate_index + 1)
            candidates[candidate_index]["diagnostics"] = state.get("metrics") or {}
        dp.append(row)

    path_indexes: list[int | None] = [None] * len(candidates_by_ref)
    last_row_index = None
    for index in range(len(dp) - 1, -1, -1):
        if dp[index]:
            last_row_index = index
            break
    if last_row_index is None:
        return [None] * len(candidates_by_ref)

    cursor = max(range(len(dp[last_row_index])), key=lambda idx: dp[last_row_index][idx]["score"])
    for ref_index in range(last_row_index, -1, -1):
        path_indexes[ref_index] = cursor
        prev_index = dp[ref_index][cursor].get("prev_index")
        if prev_index is None:
            break
        cursor = int(prev_index)

    chosen: list[dict[str, Any] | None] = []
    for ref_index, candidate_index in enumerate(path_indexes):
        if candidate_index is None or not candidates_by_ref[ref_index]:
            chosen.append(None)
            continue
        candidate = dict(candidates_by_ref[ref_index][candidate_index])
        state = dp[ref_index][candidate_index]
        candidate["final_score"] = round(float(state.get("node_score") or 0.0), 4)
        candidate["path_score"] = round(float(state.get("score") or 0.0), 4)
        candidate["diagnostics"] = state.get("metrics") or {}
        candidate["final_rank"] = int(state.get("final_rank") or 1)
        chosen.append(candidate)
    return chosen


def _score_gap(candidates: list[dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    ordered = sorted(candidates, key=lambda row: float(row.get("visual_score") or 0.0), reverse=True)
    if len(ordered) == 1:
        return float(ordered[0].get("visual_score") or 0.0)
    return float(ordered[0].get("visual_score") or 0.0) - float(ordered[1].get("visual_score") or 0.0)


def _is_anchor(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    return (
        float(candidate.get("visual_score") or 0.0) >= 0.86
        and _score_gap(candidates) >= 0.10
        and float(candidate.get("refinement_score") or candidate.get("visual_score") or 0.0) >= 0.78
    )


def solve_segmented_global_path(
    ref_shots: list[dict[str, Any]],
    candidates_by_ref: list[list[dict[str, Any]]],
    *,
    min_visual_score: float = 0.35,
) -> list[dict[str, Any] | None]:
    working: list[list[dict[str, Any]]] = []
    segment_index = 0
    for ref_index, candidates in enumerate(candidates_by_ref):
        if not candidates:
            working.append([])
            continue
        ordered = sorted(candidates, key=lambda row: float(row.get("visual_score") or 0.0), reverse=True)
        best = dict(ordered[0])
        if _is_anchor(best, ordered):
            best["path_anchor"] = True
            best_diag = dict(best.get("diagnostics") or {})
            best_diag["path"] = {
                "anchor": True,
                "segment_index": segment_index,
                "jump_allowed": True,
                "skip_state": False,
            }
            best["diagnostics"] = best_diag
            working.append([best])
            segment_index += 1
            continue
        if float(best.get("visual_score") or 0.0) < float(min_visual_score) * 0.85:
            working.append([])
            continue
        working.append(ordered)

    chosen: list[dict[str, Any] | None] = [None] * len(working)
    cursor = 0
    while cursor < len(working):
        if not working[cursor]:
            cursor += 1
            continue
        start = cursor
        while cursor < len(working) and working[cursor]:
            cursor += 1
        segment_path = solve_global_path(ref_shots[start:cursor], working[start:cursor])
        for offset, candidate in enumerate(segment_path):
            chosen[start + offset] = candidate
    current_segment = 0
    for index, candidate in enumerate(chosen):
        if candidate is None:
            continue
        diagnostics = dict(candidate.get("diagnostics") or {})
        path_diag = dict(diagnostics.get("path") or {})
        anchor = bool(candidate.get("path_anchor") or path_diag.get("anchor"))
        if anchor and index > 0:
            current_segment += 1
        path_diag.update(
            {
                "anchor": anchor,
                "segment_index": current_segment,
                "jump_allowed": anchor,
                "skip_state": False,
            }
        )
        diagnostics["path"] = path_diag
        candidate["diagnostics"] = diagnostics
    return chosen


def solve_greedy_path(
    ref_shots: list[dict[str, Any]],
    candidates_by_ref: list[list[dict[str, Any]]],
) -> list[dict[str, Any] | None]:
    chosen: list[dict[str, Any] | None] = []
    prev: dict[str, Any] | None = None
    for ref_index, candidates in enumerate(candidates_by_ref):
        if not candidates:
            chosen.append(None)
            continue

        ref_gap = 0.0
        if ref_index > 0:
            ref_gap = float(ref_shots[ref_index].get("start") or 0.0) - float(ref_shots[ref_index - 1].get("end") or 0.0)

        scored = []
        for candidate in candidates:
            metrics = transition_metrics(prev, candidate, ref_gap)
            final = _node_score(candidate, metrics, visual_weight=0.7, transition_weight=0.3)
            candidate["final_score"] = round(float(final), 4)
            candidate["diagnostics"] = metrics
            scored.append((final, metrics, candidate))
        scored.sort(key=lambda row: row[0], reverse=True)
        for rank, row in enumerate(scored, start=1):
            row[2]["final_rank"] = rank
        best_final, metrics, best_candidate = scored[0]
        candidate = dict(best_candidate)
        candidate["final_score"] = round(float(best_final), 4)
        candidate["path_score"] = round(sum(float(x.get("path_score") or x.get("final_score") or 0.0) for x in chosen if x) + best_final, 4)
        candidate["diagnostics"] = metrics
        candidate["final_rank"] = 1
        chosen.append(candidate)
        prev = candidate
    return chosen
