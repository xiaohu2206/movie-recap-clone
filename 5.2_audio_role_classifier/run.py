from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

import classifier_llm
from clone_narration_video.utils.ai import AIModelConfig, CustomOpenAIProvider
from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.progress import emit_progress
from clone_narration_video.utils.project_paths import default_output_dir


STRONG_SUBTITLE_COVERAGE = 0.45
WEAK_SUBTITLE_COVERAGE = 0.20
CONNECTED_GAP_SECONDS = 1.0
SHORT_CLIP_SECONDS = 1.2
ORIGINAL_AUDIO_BUDGET_RATIO = 0.30
MIN_ORIGINAL_BLOCK_SECONDS = 3.0


def _clip_index(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("clip_index"))
    except Exception:
        return fallback


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _row_start(row: dict[str, Any]) -> float:
    return _float(row.get("start"))


def _row_end(row: dict[str, Any]) -> float:
    return _float(row.get("end"))


def _duration(row: dict[str, Any]) -> float:
    return max(0.0, _row_end(row) - _row_start(row))


def _subtitle_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for sub in row.get("movie_subtitles") or []:
        text = str(sub.get("text") if isinstance(sub, dict) else sub or "").strip()
        if text:
            texts.append(text)
    return texts


def _subtitle_intervals(row: dict[str, Any]) -> list[tuple[float, float]]:
    start = _row_start(row)
    end = _row_end(row)
    if end <= start:
        return []

    subtitles = row.get("movie_subtitles") or []
    if not isinstance(subtitles, list):
        return []

    intervals: list[tuple[float, float]] = []
    for sub in subtitles:
        if not isinstance(sub, dict):
            continue
        sub_start = _float(sub.get("start"), -1.0)
        sub_end = _float(sub.get("end"), -1.0)
        overlap_start = max(start, sub_start)
        overlap_end = min(end, sub_end)
        if overlap_end > overlap_start:
            intervals.append((overlap_start, overlap_end))
    return intervals


def _merged_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _subtitle_stats(row: dict[str, Any]) -> tuple[float, int, float]:
    duration = _duration(row)
    intervals = _subtitle_intervals(row)
    overlap = _merged_duration(intervals)
    coverage = overlap / duration if duration > 0 else 0.0
    return coverage, len(intervals), overlap


def _is_connected(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _row_start(right) <= _row_end(left) + CONNECTED_GAP_SECONDS


def _initial_clip_signals(segment: dict[str, Any], ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_role = str(segment.get("text_role") or "").lower()
    signals: list[dict[str, Any]] = []
    total = len(ranges)
    for fallback, row in enumerate(ranges):
        coverage, subtitle_count, subtitle_overlap = _subtitle_stats(row)
        score = 0
        if text_role == "ending":
            score -= 4
        elif text_role == "hook":
            score -= 3 if fallback < max(1, int(total * 0.6)) else 1
        if coverage >= STRONG_SUBTITLE_COVERAGE:
            score += 3
        elif coverage >= WEAK_SUBTITLE_COVERAGE:
            score += 2
        elif subtitle_overlap > 0:
            score += 1
        else:
            score -= 3
        if subtitle_count >= 2:
            score += 1
        if _duration(row) < SHORT_CLIP_SECONDS:
            score -= 1
        signals.append(
            {
                "clip_index": _clip_index(row, fallback),
                "coverage": round(coverage, 4),
                "subtitle_count": subtitle_count,
                "subtitle_overlap": round(subtitle_overlap, 3),
                "score": score,
                "audio_role": "narration",
            }
        )

    for pos, signal in enumerate(signals):
        row = ranges[pos]
        connected_subtitled_neighbor = False
        if pos > 0:
            connected_subtitled_neighbor = (
                signals[pos - 1]["coverage"] >= WEAK_SUBTITLE_COVERAGE
                and signal["coverage"] >= WEAK_SUBTITLE_COVERAGE
                and _is_connected(ranges[pos - 1], row)
            )
        if pos + 1 < len(signals):
            connected_subtitled_neighbor = connected_subtitled_neighbor or (
                signals[pos + 1]["coverage"] >= WEAK_SUBTITLE_COVERAGE
                and signal["coverage"] >= WEAK_SUBTITLE_COVERAGE
                and _is_connected(row, ranges[pos + 1])
            )
        if connected_subtitled_neighbor:
            signal["score"] += 1
        if signal["score"] >= 3:
            signal["audio_role"] = "original_dialogue"
    return signals


def _role_groups(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for pos, signal in enumerate(signals):
        role = str(signal.get("audio_role") or "narration")
        if groups and groups[-1]["audio_role"] == role:
            groups[-1]["end_pos"] = pos
            groups[-1]["clip_indexes"].append(int(signal["clip_index"]))
            groups[-1]["score"] += float(signal.get("score") or 0)
        else:
            groups.append(
                {
                    "audio_role": role,
                    "start_pos": pos,
                    "end_pos": pos,
                    "clip_indexes": [int(signal["clip_index"])],
                    "score": float(signal.get("score") or 0),
                }
            )
    return groups


def _group_span_duration(group: dict[str, Any], ranges: list[dict[str, Any]]) -> float:
    rows = ranges[int(group["start_pos"]) : int(group["end_pos"]) + 1]
    if not rows:
        return 0.0
    return max(0.0, _row_end(rows[-1]) - _row_start(rows[0]))


def _smooth_clip_roles(signals: list[dict[str, Any]], ranges: list[dict[str, Any]]) -> None:
    for group in _role_groups(signals):
        if group["audio_role"] != "original_dialogue":
            continue
        if _group_span_duration(group, ranges) < MIN_ORIGINAL_BLOCK_SECONDS:
            for pos in range(int(group["start_pos"]), int(group["end_pos"]) + 1):
                signals[pos]["audio_role"] = "narration"

    groups = _role_groups(signals)
    for idx, group in enumerate(groups):
        if group["audio_role"] != "narration" or idx == 0 or idx + 1 >= len(groups):
            continue
        if groups[idx - 1]["audio_role"] != "original_dialogue" or groups[idx + 1]["audio_role"] != "original_dialogue":
            continue
        if len(group["clip_indexes"]) != 1 or _group_span_duration(group, ranges) > SHORT_CLIP_SECONDS:
            continue
        pos = int(group["start_pos"])
        if _is_connected(ranges[pos - 1], ranges[pos]) and _is_connected(ranges[pos], ranges[pos + 1]):
            signals[pos]["audio_role"] = "original_dialogue"


def _candidate_score(signals: list[dict[str, Any]], original_positions: set[int]) -> float:
    score = 0.0
    for pos, signal in enumerate(signals):
        desire = float(signal.get("score") or 0) - 2.5
        score += desire if pos in original_positions else -desire
    return score


def _project_to_best_legacy_pattern(signals: list[dict[str, Any]]) -> None:
    total = len(signals)
    candidates: list[set[int]] = [set(), set(range(total))]
    for split_pos in range(1, total):
        candidates.append(set(range(split_pos, total)))
        candidates.append(set(range(0, split_pos)))
    best = max(candidates, key=lambda positions: _candidate_score(signals, positions))
    for pos, signal in enumerate(signals):
        signal["audio_role"] = "original_dialogue" if pos in best else "narration"


def _signals_to_decision(signals: list[dict[str, Any]]) -> dict[str, Any]:
    original_indexes = {
        int(signal["clip_index"])
        for signal in signals
        if signal.get("audio_role") == "original_dialogue"
    }
    all_indexes = [int(signal["clip_index"]) for signal in signals]
    if not all_indexes or not original_indexes:
        pattern = "all_narration"
        split = None
    elif len(original_indexes) == len(all_indexes):
        pattern = "all_original_audio"
        split = None
    else:
        first_original_pos = next(pos for pos, signal in enumerate(signals) if signal.get("audio_role") == "original_dialogue")
        first_narration_after_original = next(
            (
                pos
                for pos, signal in enumerate(signals[first_original_pos:], start=first_original_pos)
                if signal.get("audio_role") != "original_dialogue"
            ),
            None,
        )
        if first_original_pos == 0:
            pattern = "original_audio_then_narration"
            split = int(signals[first_narration_after_original]["clip_index"]) if first_narration_after_original is not None else None
        else:
            pattern = "narration_then_original_audio"
            split = int(signals[first_original_pos]["clip_index"])
    return {
        "audio_pattern": pattern,
        "split_clip_index": split,
        "source": "timing_rule",
        "clip_signals": signals,
    }


def _timing_signals_for_segment(segment: dict[str, Any]) -> list[dict[str, Any]]:
    ranges = [
        {**row, "clip_index": _clip_index(row, fallback)}
        for fallback, row in enumerate(segment.get("movie_time_ranges") or [])
        if isinstance(row, dict)
    ]
    signals = _initial_clip_signals(segment, ranges)
    _smooth_clip_roles(signals, ranges)
    return signals


def _classify_segment(segment: dict[str, Any]) -> dict[str, Any]:
    signals = _timing_signals_for_segment(segment)
    _project_to_best_legacy_pattern(signals)
    return _signals_to_decision(signals)


def _groups_for_decision(segment: dict[str, Any], decision: dict[str, Any]) -> tuple[set[int], set[int]]:
    clip_signals = decision.get("clip_signals") or []
    if clip_signals and str(decision.get("source") or "").startswith("timing"):
        original_indexes = {
            int(signal.get("clip_index"))
            for signal in clip_signals
            if signal.get("audio_role") == "original_dialogue"
        }
        indexes = {_clip_index(row, idx) for idx, row in enumerate(segment.get("movie_time_ranges") or [])}
        return indexes - original_indexes, original_indexes

    ranges = segment.get("movie_time_ranges") or []
    indexes = [_clip_index(row, idx) for idx, row in enumerate(ranges)]
    pattern = str(decision.get("audio_pattern") or "all_narration")
    split = decision.get("split_clip_index")
    if pattern == "all_original_audio":
        return set(), set(indexes)
    if pattern == "narration_then_original_audio":
        split_value = int(split)
        return {idx for idx in indexes if idx < split_value}, {idx for idx in indexes if idx >= split_value}
    if pattern == "original_audio_then_narration":
        split_value = int(split)
        return {idx for idx in indexes if idx >= split_value}, {idx for idx in indexes if idx < split_value}
    return set(indexes), set()


def _slice_text_by_ratio(text: str, start_ratio: float, end_ratio: float) -> str:
    raw = text or ""
    if not raw:
        return ""
    length = len(raw)
    start = max(0, min(length, int(round(length * start_ratio))))
    end = max(start, min(length, int(round(length * end_ratio))))
    return raw[start:end].strip()


def _narration_text(segment: dict[str, Any], narration_indexes: set[int]) -> str:
    old_text = str(segment.get("old_text") or segment.get("text") or "")
    ranges = segment.get("movie_time_ranges") or []
    total_duration = sum(_duration(row) for row in ranges)
    if not old_text or not ranges or not narration_indexes:
        return ""
    if len(narration_indexes) == len(ranges) or total_duration <= 0:
        return old_text

    cursor = 0.0
    parts: list[str] = []
    for idx, row in sorted((_clip_index(row, fallback), row) for fallback, row in enumerate(ranges)):
        start_ratio = cursor / total_duration
        cursor += _duration(row)
        end_ratio = cursor / total_duration
        if idx in narration_indexes:
            parts.append(_slice_text_by_ratio(old_text, start_ratio, end_ratio))
    return "".join(parts).strip() or old_text


def _audio_blocks(ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for row in ranges:
        role = str(row.get("audio_role") or "narration")
        block_type = "original_audio" if role == "original_dialogue" else "narration"
        idx = _clip_index(row, len(blocks))
        if blocks and blocks[-1]["type"] == block_type:
            blocks[-1]["clip_indexes"].append(idx)
            blocks[-1]["end"] = _row_end(row)
        else:
            blocks.append(
                {
                    "type": block_type,
                    "clip_indexes": [idx],
                    "start": _row_start(row),
                    "end": _row_end(row),
                }
            )
    return blocks


def _narration_decision(source: str = "budget_rule") -> dict[str, Any]:
    return {
        "audio_pattern": "all_narration",
        "split_clip_index": None,
        "source": source,
    }


def _original_time_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=_row_start):
        if not groups or _row_start(row) > _row_end(groups[-1][-1]) + CONNECTED_GAP_SECONDS:
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def _decision_budget_stats(segment: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    item = dict(segment)
    ranges = []
    for fallback, row in enumerate(segment.get("movie_time_ranges") or []):
        if not isinstance(row, dict):
            continue
        ranges.append({**row, "clip_index": _clip_index(row, fallback)})
    item["movie_time_ranges"] = ranges

    _, original_indexes = _groups_for_decision(item, decision)
    original_rows = [row for row in ranges if _clip_index(row, 0) in original_indexes]
    time_groups = _original_time_groups(original_rows)
    group_durations = [
        max(0.0, _row_end(group[-1]) - _row_start(group[0]))
        for group in time_groups
        if group
    ]
    return {
        "clip_count": len(original_indexes),
        "duration": sum(group_durations),
        "group_durations": group_durations,
    }


def _decision_priority(decision: dict[str, Any], stats: dict[str, Any]) -> float:
    try:
        return float(decision.get("priority"))
    except Exception:
        return 50.0 + min(25.0, float(stats.get("duration") or 0.0))


def _apply_budget_constraints(
    script_mapping: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    total_duration = sum(_duration(row) for segment in script_mapping for row in segment.get("movie_time_ranges") or [] if isinstance(row, dict))
    total_clips = sum(1 for segment in script_mapping for row in segment.get("movie_time_ranges") or [] if isinstance(row, dict))
    max_original_duration = total_duration * ORIGINAL_AUDIO_BUDGET_RATIO
    max_original_clips = math.ceil(total_clips * ORIGINAL_AUDIO_BUDGET_RATIO) if total_clips else 0

    constrained: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    rejected_short = 0

    for order, segment in enumerate(script_mapping):
        seg_id = str(segment.get("segment_id") or "")
        decision = dict(decisions.get(seg_id) or _narration_decision("missing_decision"))
        stats = _decision_budget_stats(segment, decision)
        if stats["clip_count"] <= 0:
            constrained[seg_id] = decision
            continue
        if not stats["group_durations"] or min(stats["group_durations"]) < MIN_ORIGINAL_BLOCK_SECONDS:
            rejected_short += 1
            constrained[seg_id] = _narration_decision("budget_min_original_block")
            constrained[seg_id]["budget_rejected"] = {
                "reason": "original block shorter than minimum",
                "min_original_block_seconds": MIN_ORIGINAL_BLOCK_SECONDS,
                "group_durations": stats["group_durations"],
            }
            continue
        candidates.append(
            {
                "order": order,
                "segment_id": seg_id,
                "decision": decision,
                "duration": float(stats["duration"]),
                "clip_count": int(stats["clip_count"]),
                "priority": _decision_priority(decision, stats),
            }
        )
        constrained[seg_id] = _narration_decision("budget_pending")

    used_duration = 0.0
    used_clips = 0
    kept_segments: list[str] = []
    rejected_budget = 0
    for candidate in sorted(candidates, key=lambda item: (-item["priority"], item["order"])):
        fits_duration = used_duration + candidate["duration"] <= max_original_duration + 1e-6
        fits_clips = used_clips + candidate["clip_count"] <= max_original_clips
        if fits_duration and fits_clips:
            decision = dict(candidate["decision"])
            decision["budget_kept"] = {
                "priority": candidate["priority"],
                "duration": round(candidate["duration"], 3),
                "clip_count": candidate["clip_count"],
            }
            constrained[candidate["segment_id"]] = decision
            used_duration += candidate["duration"]
            used_clips += candidate["clip_count"]
            kept_segments.append(candidate["segment_id"])
        else:
            rejected_budget += 1
            constrained[candidate["segment_id"]] = _narration_decision("budget_over_limit")
            constrained[candidate["segment_id"]]["budget_rejected"] = {
                "reason": "original budget exceeded",
                "priority": candidate["priority"],
                "duration": round(candidate["duration"], 3),
                "clip_count": candidate["clip_count"],
            }

    return constrained, {
        "target_ratio": ORIGINAL_AUDIO_BUDGET_RATIO,
        "min_original_block_seconds": MIN_ORIGINAL_BLOCK_SECONDS,
        "total_duration": round(total_duration, 3),
        "max_original_duration": round(max_original_duration, 3),
        "used_original_duration": round(used_duration, 3),
        "total_clips": total_clips,
        "max_original_clips": max_original_clips,
        "used_original_clips": used_clips,
        "kept_segments": kept_segments,
        "rejected_short": rejected_short,
        "rejected_budget": rejected_budget,
    }


def _apply_decision(segment: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    item = dict(segment)
    ranges = []
    for fallback, row in enumerate(segment.get("movie_time_ranges") or []):
        new_row = dict(row)
        new_row["clip_index"] = _clip_index(new_row, fallback)
        ranges.append(new_row)
    item["movie_time_ranges"] = ranges

    narration_indexes, original_indexes = _groups_for_decision(item, decision)
    for row in ranges:
        idx = _clip_index(row, 0)
        if idx in original_indexes:
            row["audio_role"] = "original_dialogue"
            row["audio_action"] = "keep_original_audio"
        else:
            row["audio_role"] = "narration"
            row["audio_action"] = "narration"

    original_subtitles: list[str] = []
    for row in ranges:
        if _clip_index(row, 0) in original_indexes:
            original_subtitles.extend(_subtitle_texts(row))

    item["audio_pattern"] = decision.get("audio_pattern") or "all_narration"
    item["split_clip_index"] = decision.get("split_clip_index")
    item["narration_part"] = {
        "clip_indexes": sorted(narration_indexes),
        "old_text": _narration_text(item, narration_indexes),
    }
    item["original_audio_part"] = {
        "clip_indexes": sorted(original_indexes),
        "subtitles": original_subtitles,
    }
    item["audio_blocks"] = _audio_blocks(ranges)
    item["audio_decision"] = {
        "source": decision.get("source") or "timing_rule",
        "clip_signals": decision.get("clip_signals") or [],
        "strategy": {
            "subtitle_text_used_by_code": False,
            "subtitle_text_available_to_llm": decision.get("source") == "llm",
            "strong_subtitle_coverage": STRONG_SUBTITLE_COVERAGE,
            "weak_subtitle_coverage": WEAK_SUBTITLE_COVERAGE,
            "min_original_block_seconds": MIN_ORIGINAL_BLOCK_SECONDS,
            "original_audio_budget_ratio": ORIGINAL_AUDIO_BUDGET_RATIO,
            "legacy_projection": "best_of_four_audio_patterns",
            "llm_instruction": "semantic judgement without string matching",
        },
    }
    if decision.get("budget_kept"):
        item["audio_decision"]["budget_kept"] = decision.get("budget_kept")
    if decision.get("budget_rejected"):
        item["audio_decision"]["budget_rejected"] = decision.get("budget_rejected")
    return item


async def classify_audio_roles(
    script_mapping: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    batch_size: int,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    timing_decisions: dict[str, dict[str, Any]] = {}
    llm_input: list[dict[str, Any]] = []
    for segment in script_mapping:
        seg_id = str(segment.get("segment_id") or "")
        timing_signals = _timing_signals_for_segment(segment)
        timing_segment = dict(segment)
        timing_segment["_timing_signals"] = timing_signals
        llm_input.append(timing_segment)
        timing_decisions[seg_id] = _classify_segment(segment)

    if progress_callback:
        progress_callback(15.0, "Prepared timing signals")

    decisions: dict[str, dict[str, Any]] = {}
    llm_calls = 0
    fallback = 0
    used_llm = bool((api_key or "").strip())
    if used_llm:
        cfg = AIModelConfig(
            provider="custom_openai",
            api_key=api_key,
            base_url=base_url,
            model_name=model,
            temperature=temperature,
        )
        provider = CustomOpenAIProvider(cfg)
        try:
            decisions = await classifier_llm.classify_segments(llm_input, provider=provider, batch_size=batch_size)
            llm_calls = (len(llm_input) + max(1, batch_size) - 1) // max(1, batch_size)
        finally:
            await provider.close()
        if progress_callback:
            progress_callback(70.0, "Applied LLM audio decisions")
    else:
        decisions = dict(timing_decisions)
        if progress_callback:
            progress_callback(70.0, "Using timing fallback because API key is missing")

    for segment in script_mapping:
        seg_id = str(segment.get("segment_id") or "")
        if (decisions.get(seg_id) or {}).get("source") == "fallback":
            fallback += 1
            fallback_decision = dict(timing_decisions.get(seg_id) or _classify_segment(segment))
            fallback_decision["source"] = "timing_fallback_after_llm"
            decisions[seg_id] = fallback_decision

    budgeted_decisions, budget_summary = _apply_budget_constraints(script_mapping, decisions)

    output_rows: list[dict[str, Any]] = []
    total = len(script_mapping)
    for idx, segment in enumerate(script_mapping, start=1):
        seg_id = str(segment.get("segment_id") or "")
        timing_decision = timing_decisions.get(seg_id) or _classify_segment(segment)
        decision = budgeted_decisions.get(seg_id) or decisions.get(seg_id) or timing_decision
        decision["clip_signals"] = timing_decision.get("clip_signals") or []
        output_rows.append(_apply_decision(segment, decision))
        if progress_callback and (idx == 1 or idx == total or idx % 10 == 0):
            progress_callback(70.0 + (idx / max(1, total)) * 30.0, f"Classified audio roles {idx}/{total}")

    return {
        "script_mapping": output_rows,
        "audio_backend": {
            "provider": "custom_openai" if used_llm else "timing_rule",
            "model": model if used_llm else "",
            "rule_hit": 0 if used_llm else len(script_mapping),
            "llm_segments": len(script_mapping) if used_llm else 0,
            "llm_calls": llm_calls,
            "fallback": fallback,
            "subtitle_text_used_by_code": False,
            "subtitle_text_available_to_llm": used_llm,
            "budget": budget_summary,
        },
    }


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="原声判定模块")
    parser.add_argument("--script-mapping", required=True, help="第 5.1 步输出的 script_mapping_subtitled.json")
    parser.add_argument("--output-dir", default=str(default_output_dir("5.2_audio_role_classifier")))
    parser.add_argument("--provider", choices=["custom_openai"], default=os.getenv("CLONE_AI_PROVIDER", "custom_openai"))
    parser.add_argument("--api-key", default=os.getenv("CLONE_AI_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    parser.add_argument("--base-url", default=os.getenv("CLONE_AI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    parser.add_argument("--model", default=os.getenv("CLONE_AI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("CLONE_AI_TEMPERATURE", "0.2")))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    data = read_json(args.script_mapping)
    if "script_mapping" not in data:
        raise SystemExit("script_mapping.json 缺少 script_mapping 数组")
    script_mapping = data.get("script_mapping")
    if not isinstance(script_mapping, list):
        raise SystemExit("script_mapping.json 缺少 script_mapping 数组")

    result = asyncio.run(
        classify_audio_roles(
            script_mapping,
            api_key=(args.api_key or "").strip(),
            base_url=(args.base_url or "").strip(),
            model=(args.model or "").strip(),
            temperature=float(args.temperature),
            batch_size=int(args.batch_size),
            progress_callback=lambda percent, message: emit_progress("audio_role", percent, message),
        )
    )
    out = write_json(Path(args.output_dir) / "script_mapping_with_audio.json", result)
    print(out)


if __name__ == "__main__":
    main()
