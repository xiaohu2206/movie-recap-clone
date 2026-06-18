from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.project_paths import default_output_dir
from clone_narration_video.utils.shot_breakdown import build_shot_breakdown

MIN_ORIGINAL_PLAY_DURATION = 3.0
ORIGINAL_MERGE_GAP = 0.35
MOVIE_ROLLBACK_TOLERANCE = 0.05
MIN_CLIP_DURATION = 0.03


def _round(value: float) -> float:
    return round(float(value), 3)


def _duration(start: Any, end: Any) -> float:
    try:
        return max(0.0, float(end) - float(start))
    except Exception:
        return 0.0


def _text_units(text: str) -> int:
    cleaned = re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]【】]", "", text or "")
    return max(1, len(cleaned))


def estimate_tts_duration(text: str, chars_per_second: float, min_duration: float) -> float:
    cps = max(0.1, chars_per_second)
    return _round(max(min_duration, _text_units(text) / cps))


def _movie_shot_rows(movie_shots_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not movie_shots_data:
        return []
    rows = movie_shots_data.get("movie_shots") or []
    if not isinstance(rows, list):
        return []
    return sorted(rows, key=lambda x: float(x.get("start") or 0.0))


def _clip(
    clip_index: int,
    start: float,
    end: float,
    source: str,
    *,
    source_ref_shot_id: str | None = None,
    movie_shot_ids: list[str] | None = None,
    allocation: str,
) -> dict[str, Any]:
    return {
        "clip_id": f"clip_{clip_index:03d}",
        "movie_start": _round(start),
        "movie_end": _round(end),
        "duration": _round(_duration(start, end)),
        "source": source,
        "source_ref_shot_id": source_ref_shot_id,
        "movie_shot_ids": movie_shot_ids or [],
        "allocation": allocation,
    }


def _base_ranges(item: dict[str, Any]) -> list[dict[str, Any]]:
    ranges = []
    for row in item.get("movie_time_ranges") or []:
        dur = _duration(row.get("start"), row.get("end"))
        if dur <= 0:
            continue
        ranges.append(row)
    return ranges


def _range_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("range_id") or f"range_{index:03d}")


def _range_lookup(item: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    by_id = {}
    for idx, row in enumerate(_base_ranges(item), start=1):
        item_row = dict(row)
        item_row["range_id"] = _range_id(item_row, idx)
        rows.append(item_row)
        by_id[str(item_row["range_id"])] = item_row
    return rows, by_id


def _audio_action(row: dict[str, Any]) -> str:
    return str(row.get("audio_action") or "rewrite_and_voiceover")


def _audio_mode_from_action(action: str) -> str:
    if action == "play_original_audio":
        return "original"
    if action == "play_original_audio_low_volume":
        return "mixed"
    return "voiceover"


def _clip_audio_enabled(audio_mode: str) -> bool:
    return audio_mode in {"original", "mixed"}


def _range_ids_for_unit(unit: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[str]:
    ids = [str(x) for x in unit.get("related_range_ids") or [] if str(x) in by_id]
    return list(dict.fromkeys(ids))


def _unit_audio_mode(unit: dict[str, Any], ranges: list[dict[str, Any]]) -> str:
    if bool(unit.get("keep_original_audio")):
        return "original"
    actions = {_audio_action(row) for row in ranges}
    if "play_original_audio" in actions:
        return "original"
    if "play_original_audio_low_volume" in actions:
        return "mixed"
    return "voiceover"


def _unit_text(unit: dict[str, Any], key: str) -> str:
    return str(unit.get(key) or "").strip()


def _unit_voiceover_text(unit: dict[str, Any]) -> str:
    return _unit_text(unit, "new_text") or _unit_text(unit, "old_text")


def _join_text(parts: list[str]) -> str:
    return "".join(part.strip() for part in parts if part and part.strip())


def _ranges_for_ids(ids: list[str], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [by_id[range_id] for range_id in ids if range_id in by_id]


def _clips_from_ranges(
    ranges: list[dict[str, Any]],
    *,
    source: str,
    audio_mode: str,
    allocation: str,
) -> list[dict[str, Any]]:
    clips = []
    for row in ranges:
        clip = _clip(
            len(clips) + 1,
            float(row.get("start") or 0.0),
            float(row.get("end") or 0.0),
            source,
            source_ref_shot_id=row.get("source_ref_shot_id"),
            movie_shot_ids=[str(x) for x in row.get("movie_shot_ids") or []],
            allocation=allocation,
        )
        clip["keep_original_audio"] = _clip_audio_enabled(audio_mode)
        if audio_mode == "mixed":
            clip["original_audio_volume"] = 0.35
        clips.append(clip)
    return clips


def _mark_clip_audio(clips: list[dict[str, Any]], audio_mode: str) -> list[dict[str, Any]]:
    result = []
    for clip in clips:
        row = dict(clip)
        row["keep_original_audio"] = _clip_audio_enabled(audio_mode)
        if audio_mode == "mixed":
            row["original_audio_volume"] = 0.35
        result.append(row)
    return result


def _range_audio_decision(audio_mode: str, ranges: list[dict[str, Any]]) -> dict[str, Any]:
    roles = [str(row.get("audio_role") or "") for row in ranges if row.get("audio_role")]
    actions = [_audio_action(row) for row in ranges]
    source_range_ids = [str(row.get("range_id") or "") for row in ranges if row.get("range_id")]
    default_role = {
        "voiceover": "narration_overlay",
        "original": "original_dialogue",
        "mixed": "mixed_narration_and_dialogue",
    }.get(audio_mode, "narration_overlay")
    default_action = {
        "voiceover": "rewrite_and_voiceover",
        "original": "play_original_audio",
        "mixed": "play_original_audio_low_volume",
    }.get(audio_mode, "rewrite_and_voiceover")
    return {
        "audio_role": roles[0] if len(set(roles)) == 1 else default_role,
        "audio_action": actions[0] if len(set(actions)) == 1 else default_action,
        "source_range_ids": source_range_ids,
    }


def _ref_source(item: dict[str, Any], ranges: list[dict[str, Any]]) -> dict[str, Any]:
    ref_range = item.get("ref_time_range") or {}
    ref_shot_ids = [str(row.get("source_ref_shot_id")) for row in ranges if row.get("source_ref_shot_id")]
    return {
        "ref_start": ref_range.get("start"),
        "ref_end": ref_range.get("end"),
        "ref_shot_ids": list(dict.fromkeys(ref_shot_ids)),
    }


def _has_audio_decisions(item: dict[str, Any]) -> bool:
    if isinstance(item.get("rewritten_units"), list) and item.get("rewritten_units"):
        return True
    return any(isinstance(row, dict) and row.get("audio_action") for row in item.get("movie_time_ranges") or [])


def _chunk_by_units(item: dict[str, Any], all_ranges: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    used_range_ids: set[str] = set()
    units = [unit for unit in item.get("rewritten_units") or [] if isinstance(unit, dict)]

    for unit in units:
        range_ids = _range_ids_for_unit(unit, by_id)
        ranges = _ranges_for_ids(range_ids, by_id)
        mode = _unit_audio_mode(unit, ranges)
        if not ranges and mode == "original":
            continue
        narration_text = "" if mode == "original" else _unit_text(unit, "new_text")
        if mode in {"voiceover", "mixed"} and not narration_text and not ranges:
            continue
        if chunks and chunks[-1]["audio_mode"] == mode:
            chunks[-1]["units"].append(unit)
            chunks[-1]["range_ids"].extend(range_id for range_id in range_ids if range_id not in chunks[-1]["range_ids"])
        else:
            chunks.append({"audio_mode": mode, "units": [unit], "range_ids": range_ids})
        used_range_ids.update(range_ids)

    for row in all_ranges:
        range_id = str(row.get("range_id") or "")
        if not range_id or range_id in used_range_ids:
            continue
        mode = _audio_mode_from_action(_audio_action(row))
        if chunks and chunks[-1]["audio_mode"] == mode and not chunks[-1]["units"]:
            chunks[-1]["range_ids"].append(range_id)
        else:
            chunks.append({"audio_mode": mode, "units": [], "range_ids": [range_id]})
    return chunks


def _chunk_by_ranges(all_ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for row in all_ranges:
        mode = _audio_mode_from_action(_audio_action(row))
        range_id = str(row.get("range_id") or "")
        if chunks and chunks[-1]["audio_mode"] == mode:
            chunks[-1]["range_ids"].append(range_id)
        else:
            chunks.append({"audio_mode": mode, "units": [], "range_ids": [range_id]})
    return chunks


def _row_start(row: dict[str, Any]) -> float:
    return float(row.get("start") or 0.0)


def _row_end(row: dict[str, Any]) -> float:
    return float(row.get("end") or 0.0)


def _row_shot_ids(row: dict[str, Any]) -> list[str]:
    return [str(x) for x in row.get("movie_shot_ids") or [] if str(x)]


def _merge_movie_subtitles(target: dict[str, Any], row: dict[str, Any]) -> None:
    merged = list(target.get("movie_subtitles") or [])
    seen = {
        (float(x.get("start") or 0.0), float(x.get("end") or 0.0), str(x.get("text") or ""))
        for x in merged
        if isinstance(x, dict)
    }
    for sub in row.get("movie_subtitles") or []:
        if not isinstance(sub, dict):
            continue
        key = (float(sub.get("start") or 0.0), float(sub.get("end") or 0.0), str(sub.get("text") or ""))
        if key not in seen:
            merged.append(sub)
            seen.add(key)
    if merged:
        target["movie_subtitles"] = merged


def _coalesce_nearby_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in rows:
        shot_ids = _row_shot_ids(row)
        start = _row_start(row)
        end = _row_end(row)
        if end <= start:
            continue
        if merged:
            last = merged[-1]
            last_ids = _row_shot_ids(last)
            shared_shot = bool(shot_ids and set(shot_ids).intersection(last_ids))
            near_or_overlap = start <= _row_end(last) + ORIGINAL_MERGE_GAP
            if shared_shot and near_or_overlap:
                last["start"] = _round(min(_row_start(last), start))
                last["end"] = _round(max(_row_end(last), end))
                last["movie_shot_ids"] = list(dict.fromkeys([*last_ids, *shot_ids]))
                _merge_movie_subtitles(last, row)
                continue
        merged.append(dict(row))
    return merged


def _visual_keys(row: dict[str, Any]) -> set[str]:
    shot_ids = _row_shot_ids(row)
    if shot_ids:
        return {f"shot:{shot_id}" for shot_id in shot_ids}
    return {f"time:{_row_start(row):.3f}-{_row_end(row):.3f}"}


def _clip_visual_keys(clip: dict[str, Any]) -> set[str]:
    shot_ids = [str(x) for x in clip.get("movie_shot_ids") or [] if str(x)]
    if shot_ids:
        return {f"shot:{shot_id}" for shot_id in shot_ids}
    return {f"time:{float(clip.get('movie_start') or 0.0):.3f}-{float(clip.get('movie_end') or 0.0):.3f}"}


def _prepare_rows_for_timeline(
    rows: list[dict[str, Any]],
    *,
    used_visual_keys: set[str],
    min_movie_start: float | None,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    local_keys: set[str] = set()
    cursor = min_movie_start
    for row in _coalesce_nearby_rows(rows):
        keys = _visual_keys(row)
        if keys.intersection(used_visual_keys) or keys.intersection(local_keys):
            continue
        start = _row_start(row)
        end = _row_end(row)
        if cursor is not None and start < cursor - MOVIE_ROLLBACK_TOLERANCE:
            if end <= cursor + MIN_CLIP_DURATION:
                continue
            row = dict(row)
            row["start"] = _round(cursor)
            start = _row_start(row)
        if end - start <= MIN_CLIP_DURATION:
            continue
        prepared.append(row)
        local_keys.update(keys)
        cursor = max(cursor if cursor is not None else end, end)
    return prepared


def _clip_index(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("clip_index"))
    except Exception:
        return fallback


def _ranges_for_indexes(item: dict[str, Any], indexes: set[int]) -> list[dict[str, Any]]:
    selected = []
    for fallback, row in enumerate(item.get("movie_time_ranges") or []):
        if _clip_index(row, fallback) in indexes:
            selected.append(row)
    return selected


def _range_duration(rows: list[dict[str, Any]]) -> float:
    return _round(sum(_duration(row.get("start"), row.get("end")) for row in rows))


def _subtitle_texts(rows: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        for sub in row.get("movie_subtitles") or []:
            text = str(sub.get("text") if isinstance(sub, dict) else sub or "").strip()
            if text:
                texts.append(text)
    return texts


def _sub_item(item: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    clone = dict(item)
    clone["movie_time_ranges"] = rows
    return clone


def _group_original_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        start = _row_start(row)
        end = _row_end(row)
        if end <= start:
            continue
        if groups and start <= _row_end(groups[-1][-1]) + ORIGINAL_MERGE_GAP:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def _group_span_duration(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return _round(_row_end(rows[-1]) - _row_start(rows[0]))


def _group_movie_shot_ids(rows: list[dict[str, Any]]) -> list[str]:
    shot_ids: list[str] = []
    for row in rows:
        shot_ids.extend(_row_shot_ids(row))
    return list(dict.fromkeys(shot_ids))


def _group_ref_shot_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(row.get("source_ref_shot_id"))
        for row in rows
        if row.get("source_ref_shot_id")
    ]


def _original_group_narration(base_item: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    text = " ".join(_subtitle_texts(rows)).strip()
    if text:
        return text
    original_part = base_item.get("original_audio_part") or {}
    subtitles = [str(x).strip() for x in original_part.get("subtitles") or [] if str(x).strip()]
    if subtitles:
        return " ".join(subtitles)
    return str(base_item.get("new_text") or base_item.get("old_text") or "").strip()


def allocate_video_clips(
    item: dict[str, Any],
    *,
    tts_duration: float,
    source: str,
    movie_shots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    ranges = _base_ranges(item)
    if not ranges:
        return [], "missing_visual_ranges"

    available = sum(_duration(row.get("start"), row.get("end")) for row in ranges)
    clips: list[dict[str, Any]] = []

    if tts_duration <= available:
        ratio = tts_duration / available if available else 0.0
        allocated = 0.0
        for idx, row in enumerate(ranges, start=1):
            start = float(row.get("start") or 0.0)
            original_duration = _duration(row.get("start"), row.get("end"))
            if idx == len(ranges):
                take = max(0.0, tts_duration - allocated)
            else:
                take = original_duration * ratio
            if take <= 0.03:
                continue
            clips.append(
                _clip(
                    len(clips) + 1,
                    start,
                    start + take,
                    source,
                    source_ref_shot_id=row.get("source_ref_shot_id"),
                    movie_shot_ids=[str(x) for x in row.get("movie_shot_ids") or []],
                    allocation="trimmed" if ratio < 0.98 else "original",
                )
            )
            allocated += take
        return clips, "trimmed_to_tts" if tts_duration < available * 0.98 else "kept_original"

    for row in ranges:
        clips.append(
            _clip(
                len(clips) + 1,
                float(row.get("start") or 0.0),
                float(row.get("end") or 0.0),
                source,
                source_ref_shot_id=row.get("source_ref_shot_id"),
                movie_shot_ids=[str(x) for x in row.get("movie_shot_ids") or []],
                allocation="original",
            )
        )

    remaining = _round(tts_duration - available)
    if remaining > 0.03 and clips:
        clips[-1]["movie_end"] = _round(float(clips[-1]["movie_end"]) + remaining)
        clips[-1]["duration"] = _round(float(clips[-1]["duration"]) + remaining)
        clips[-1]["allocation"] = "synthetic_extension"
        return clips, "extended_last_clip"
    return clips, "kept_original"


def _confidence(item: dict[str, Any], clips: list[dict[str, Any]], allocation_status: str) -> str:
    if not clips:
        return "low"
    if allocation_status in {"missing_visual_ranges", "extended_last_clip"}:
        return "low"
    range_conf = [str(x.get("confidence") or "low") for x in item.get("movie_time_ranges") or []]
    if any(x == "low" for x in range_conf):
        return "medium"
    if str(item.get("rewrite_status")) != "ai_rewritten":
        return "medium"
    return "high"


def _timeline_item(
    *,
    item_index: int,
    item: dict[str, Any],
    ranges: list[dict[str, Any]],
    narration: str,
    audio_mode: str,
    cursor: float,
    source: str,
    movie_shots: list[dict[str, Any]],
    chars_per_second: float,
    min_duration: float,
) -> tuple[dict[str, Any], float]:
    if audio_mode == "original":
        clips = _clips_from_ranges(ranges, source=source, audio_mode=audio_mode, allocation="original_audio")
        duration = _round(sum(float(clip.get("duration") or 0.0) for clip in clips))
        allocation_status = "original_audio"
        tts_duration = duration
    else:
        tts_duration = estimate_tts_duration(narration, chars_per_second, min_duration) if narration else 0.0
        if audio_mode == "mixed" and tts_duration <= 0:
            tts_duration = _round(sum(_duration(row.get("start"), row.get("end")) for row in ranges))
        chunk_item = dict(item)
        chunk_item["movie_time_ranges"] = ranges
        clips, allocation_status = allocate_video_clips(
            chunk_item,
            tts_duration=tts_duration,
            source=source,
            movie_shots=movie_shots,
        )
        clips = _mark_clip_audio(clips, audio_mode)
        duration = tts_duration

    end = _round(cursor + duration)
    return {
        "item_id": f"item_{item_index:03d}",
        "segment_id": item.get("segment_id"),
        "audio_mode": audio_mode,
        "audio_type": "original_audio" if audio_mode == "original" else "narration",
        "OST": 1 if audio_mode in {"original", "mixed"} else 0,
        "keep_original_audio": audio_mode == "original",
        "narration": "" if audio_mode == "original" else narration,
        "tts_duration": _round(tts_duration),
        "timeline_start": _round(cursor),
        "timeline_end": end,
        "video_clips": clips,
        "ref_source": _ref_source(item, ranges),
        "audio_decision": _range_audio_decision(audio_mode, ranges),
        "confidence": _confidence({**item, "movie_time_ranges": ranges}, clips, allocation_status),
        "allocation_status": allocation_status,
        "audio_pattern": item.get("audio_pattern") or ("all_original_audio" if audio_mode == "original" else "all_narration"),
    }, end


def _compose_audio_aware_items(
    item: dict[str, Any],
    *,
    start_index: int,
    cursor: float,
    source: str,
    movie_shots: list[dict[str, Any]],
    chars_per_second: float,
    min_duration: float,
) -> tuple[list[dict[str, Any]], float]:
    all_ranges, by_id = _range_lookup(item)
    units = [unit for unit in item.get("rewritten_units") or [] if isinstance(unit, dict)]
    chunks = _chunk_by_units(item, all_ranges, by_id) if units else _chunk_by_ranges(all_ranges)
    if not chunks:
        return [], cursor

    rows: list[dict[str, Any]] = []
    next_index = start_index
    for chunk in chunks:
        audio_mode = str(chunk.get("audio_mode") or "voiceover")
        ranges = _ranges_for_ids([str(x) for x in chunk.get("range_ids") or []], by_id)
        chunk_units = [unit for unit in chunk.get("units") or [] if isinstance(unit, dict)]
        if audio_mode == "original" and not ranges:
            continue
        narration = _join_text([_unit_voiceover_text(unit) for unit in chunk_units if not unit.get("keep_original_audio")])
        if not narration and audio_mode == "voiceover" and not units:
            narration = str(item.get("new_text") or item.get("old_text") or "")
        if audio_mode in {"voiceover", "mixed"} and not narration and not ranges:
            continue
        row, cursor = _timeline_item(
            item_index=next_index,
            item=item,
            ranges=ranges,
            narration=narration,
            audio_mode=audio_mode,
            cursor=cursor,
            source=source,
            movie_shots=movie_shots,
            chars_per_second=chars_per_second,
            min_duration=min_duration,
        )
        rows.append(row)
        next_index += 1
    return rows, cursor


def compose_timeline(
    rewritten_script: list[dict[str, Any]],
    *,
    source: str,
    movie_shots_data: dict[str, Any] | None,
    chars_per_second: float,
    min_duration: float,
) -> dict[str, Any]:
    movie_shots = _movie_shot_rows(movie_shots_data)
    final_timeline = []
    cursor = 0.0
    used_visual_keys: set[str] = set()

    def usable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _prepare_rows_for_timeline(
            rows,
            used_visual_keys=used_visual_keys,
            min_movie_start=None,
        )

    def remember_clips(clips: list[dict[str, Any]]) -> None:
        for clip in clips:
            used_visual_keys.update(_clip_visual_keys(clip))

    def append_narration_item(base_item: dict[str, Any], idx: int, suffix: str, narration: str) -> None:
        nonlocal cursor
        base_item = _sub_item(base_item, usable_rows(_base_ranges(base_item)))
        tts_duration = estimate_tts_duration(narration, chars_per_second, min_duration)
        clips, allocation_status = allocate_video_clips(
            base_item,
            tts_duration=tts_duration,
            source=source,
            movie_shots=movie_shots,
        )
        ref_range = base_item.get("ref_time_range") or {}
        ref_shot_ids = [
            str(row.get("source_ref_shot_id"))
            for row in base_item.get("movie_time_ranges") or []
            if row.get("source_ref_shot_id")
        ]
        item_id = f"item_{idx:03d}{suffix}"
        final_timeline.append(
            {
                "item_id": item_id,
                "segment_id": base_item.get("segment_id"),
                "audio_mode": "voiceover",
                "audio_type": "narration",
                "OST": 0,
                "narration": narration,
                "tts_duration": tts_duration,
                "timeline_start": _round(cursor),
                "timeline_end": _round(cursor + tts_duration),
                "video_clips": _mark_clip_audio(clips, "voiceover"),
                "ref_source": {
                    "ref_start": ref_range.get("start"),
                    "ref_end": ref_range.get("end"),
                    "ref_shot_ids": ref_shot_ids,
                },
                "audio_decision": _range_audio_decision("voiceover", base_item.get("movie_time_ranges") or []),
                "confidence": _confidence(base_item, clips, allocation_status),
                "allocation_status": allocation_status,
                "audio_pattern": base_item.get("audio_pattern") or "all_narration",
            }
        )
        remember_clips(clips)
        cursor = _round(cursor + tts_duration)

    def append_original_item(base_item: dict[str, Any], idx: int, suffix: str, rows: list[dict[str, Any]]) -> None:
        nonlocal cursor
        rows = usable_rows(rows)
        groups = _group_original_rows(rows)
        for group_index, group_rows in enumerate(groups, start=1):
            group_suffix = suffix if len(groups) == 1 else f"{suffix}_{group_index:02d}"
            duration = _group_span_duration(group_rows)
            clip = _clip(
                1,
                _row_start(group_rows[0]),
                _row_end(group_rows[-1]),
                source,
                source_ref_shot_id=group_rows[0].get("source_ref_shot_id"),
                movie_shot_ids=_group_movie_shot_ids(group_rows),
                allocation="original_merged",
            )
            clip["keep_original_audio"] = True
            clips = [clip]
            ref_range = base_item.get("ref_time_range") or {}
            final_timeline.append(
                {
                    "item_id": f"item_{idx:03d}{group_suffix}",
                    "segment_id": base_item.get("segment_id"),
                    "audio_mode": "original",
                    "audio_type": "original_audio",
                    "OST": 1,
                    "keep_original_audio": True,
                    "narration": "",
                    "tts_duration": duration,
                    "timeline_start": _round(cursor),
                    "timeline_end": _round(cursor + duration),
                    "video_clips": clips,
                    "original_subtitles": _subtitle_texts(group_rows),
                    "ref_source": {
                        "ref_start": ref_range.get("start"),
                        "ref_end": ref_range.get("end"),
                        "ref_shot_ids": _group_ref_shot_ids(group_rows),
                    },
                    "audio_decision": _range_audio_decision("original", group_rows),
                    "confidence": "medium",
                    "allocation_status": "kept_original_audio_merged",
                    "audio_pattern": base_item.get("audio_pattern") or "all_original_audio",
                }
            )
            remember_clips(clips)
            cursor = _round(cursor + duration)

    for idx, item in enumerate(rewritten_script, start=1):
        if _has_audio_decisions(item):
            rows, cursor = _compose_audio_aware_items(
                item,
                start_index=len(final_timeline) + 1,
                cursor=cursor,
                source=source,
                movie_shots=movie_shots,
                chars_per_second=chars_per_second,
                min_duration=min_duration,
            )
            final_timeline.extend(rows)
            for row in rows:
                remember_clips(row.get("video_clips") or [])
            if rows:
                continue

        pattern = str(item.get("audio_pattern") or "all_narration")
        narration = str(item.get("new_text") or item.get("old_text") or "")
        narration_part = item.get("narration_part") or {}
        original_part = item.get("original_audio_part") or {}
        narration_indexes = {int(x) for x in narration_part.get("clip_indexes") or []}
        original_indexes = {int(x) for x in original_part.get("clip_indexes") or []}
        if pattern == "all_original_audio":
            rows = _base_ranges(item)
            append_original_item(item, idx, "_original", rows)
            continue
        if pattern == "narration_then_original_audio":
            narration_rows = _ranges_for_indexes(item, narration_indexes)
            original_rows = _ranges_for_indexes(item, original_indexes)
            if narration_rows:
                append_narration_item(_sub_item(item, narration_rows), idx, "_narration", narration)
            if original_rows:
                append_original_item(_sub_item(item, original_rows), idx, "_original", original_rows)
            continue
        if pattern == "original_audio_then_narration":
            original_rows = _ranges_for_indexes(item, original_indexes)
            narration_rows = _ranges_for_indexes(item, narration_indexes)
            if original_rows:
                append_original_item(_sub_item(item, original_rows), idx, "_original", original_rows)
            if narration_rows:
                append_narration_item(_sub_item(item, narration_rows), idx, "_narration", narration)
            continue
        append_narration_item(item, idx, "", narration)
    return {
        "final_timeline": final_timeline,
        "timeline_backend": {
            "tts_duration_estimator": "char_count",
            "chars_per_second": chars_per_second,
            "min_duration": min_duration,
            "movie_shot_extension": False,
            "min_original_play_duration": MIN_ORIGINAL_PLAY_DURATION,
            "dedupe_movie_shots": True,
            "prevent_movie_rollback": False,
        },
    }


def _merge_with_script_mapping(
    rewritten_rows: list[dict[str, Any]],
    script_mapping_data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not script_mapping_data:
        return rewritten_rows
    mapping = {
        str(row.get("segment_id")): row
        for row in script_mapping_data.get("script_mapping") or []
        if row.get("segment_id")
    }
    merged = []
    for row in rewritten_rows:
        seg_id = str(row.get("segment_id") or "")
        base = dict(mapping.get(seg_id) or {})
        base.update(row)
        if not base.get("movie_time_ranges") and mapping.get(seg_id):
            base["movie_time_ranges"] = mapping[seg_id].get("movie_time_ranges") or []
        merged.append(base)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="生成新视频脚本时间轴")
    parser.add_argument("--rewritten-script", required=True, help="第 6 步输出的 rewritten_script.json")
    parser.add_argument("--script-mapping", help="可选：第 5 步输出，用于补齐画面绑定信息")
    parser.add_argument("--movie-shots", help="可选：第 3 步输出，用于文案变长时扩展相邻镜头")
    parser.add_argument("--movie-source", default="movie.mp4")
    parser.add_argument("--output-dir", default=str(default_output_dir("7_timeline_composer")))
    parser.add_argument("--chars-per-second", type=float, default=4.2)
    parser.add_argument("--min-duration", type=float, default=1.2)
    parser.add_argument("--ref-analysis", help="可选：第 1 步 ref_analysis.json")
    parser.add_argument("--output-root", help="可选：outputs 根目录，用于自动查找流水线数据")
    args = parser.parse_args()

    rewritten_data = read_json(args.rewritten_script)
    rows = rewritten_data.get("rewritten_script") or []
    if not isinstance(rows, list):
        raise SystemExit("rewritten_script.json 缺少 rewritten_script 数组")

    mapping_data = read_json(args.script_mapping) if args.script_mapping else None
    movie_shots_data = read_json(args.movie_shots) if args.movie_shots else None
    rows = _merge_with_script_mapping(rows, mapping_data)
    result = compose_timeline(
        rows,
        source=args.movie_source,
        movie_shots_data=movie_shots_data,
        chars_per_second=args.chars_per_second,
        min_duration=args.min_duration,
    )
    out = write_json(Path(args.output_dir) / "final_timeline.json", result)
    print(out)

    output_root = Path(args.output_root).resolve() if args.output_root else Path(args.output_dir).resolve().parent
    ref_analysis_data = read_json(args.ref_analysis) if args.ref_analysis else None
    shot_breakdown = build_shot_breakdown(
        result.get("final_timeline") or [],
        ref_analysis=ref_analysis_data,
        script_mapping=mapping_data,
        output_root=output_root,
    )
    breakdown_out = write_json(Path(args.output_dir) / "shot_breakdown.json", shot_breakdown)
    print(breakdown_out)


if __name__ == "__main__":
    main()
