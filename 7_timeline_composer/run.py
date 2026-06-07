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
from clone_narration_video.utils.movie_time_ranges import merge_overlapping_movie_ranges
from clone_narration_video.utils.project_paths import default_output_dir

MIN_ORIGINAL_PLAY_DURATION = 3.0


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


def _prepare_movie_ranges(
    item: dict[str, Any],
    movie_shots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return merge_overlapping_movie_ranges(_base_ranges(item))


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


def _filter_range_ids_for_audio_mode(
    range_ids: list[str],
    by_id: dict[str, dict[str, Any]],
    audio_mode: str,
) -> list[str]:
    filtered: list[str] = []
    for range_id in range_ids:
        row = by_id.get(str(range_id))
        if not row:
            continue
        row_mode = _audio_mode_from_action(_audio_action(row))
        if audio_mode == "original":
            if row_mode == "original":
                filtered.append(str(range_id))
        elif audio_mode == "mixed":
            if row_mode in {"mixed", "voiceover"}:
                filtered.append(str(range_id))
        elif row_mode != "original":
            filtered.append(str(range_id))
    return filtered


def _unused_ranges(
    range_ids: list[str],
    by_id: dict[str, dict[str, Any]],
    consumed_range_ids: set[str],
) -> list[dict[str, Any]]:
    fresh_ids = [str(rid) for rid in range_ids if str(rid) and str(rid) not in consumed_range_ids]
    return _ranges_for_ids(fresh_ids, by_id)


def _drop_rendered_voiceover_ranges(
    ranges: list[dict[str, Any]],
    rendered_movie_shot_ids: set[str],
) -> list[dict[str, Any]]:
    return ranges


def _min_range_start(ranges: list[dict[str, Any]]) -> float | None:
    starts = [float(row.get("start") or 0.0) for row in ranges if _duration(row.get("start"), row.get("end")) > 0]
    return min(starts) if starts else None


def _has_short_original_play_range(ranges: list[dict[str, Any]]) -> bool:
    return any(
        0.0 < _duration(row.get("start"), row.get("end")) <= MIN_ORIGINAL_PLAY_DURATION
        for row in merge_overlapping_movie_ranges(ranges)
    )


def _clips_from_ranges(
    ranges: list[dict[str, Any]],
    *,
    source: str,
    audio_mode: str,
    allocation: str,
    movie_shots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    clips = []
    prepared = merge_overlapping_movie_ranges(ranges)
    for row in prepared:
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
    action = actions[0] if len(set(actions)) == 1 else default_action
    if _audio_mode_from_action(action) != audio_mode:
        action = default_action
    return {
        "audio_role": roles[0] if len(set(roles)) == 1 else default_role,
        "audio_action": action,
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
        range_ids = _filter_range_ids_for_audio_mode(range_ids, by_id, mode)
        ranges = _ranges_for_ids(range_ids, by_id)
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


def _used_movie_shot_ids(ranges: list[dict[str, Any]]) -> set[str]:
    used: set[str] = set()
    for row in ranges:
        for shot_id in row.get("movie_shot_ids") or []:
            used.add(str(shot_id))
    return used


def _extend_with_adjacent_shots(
    clips: list[dict[str, Any]],
    ranges: list[dict[str, Any]],
    movie_shots: list[dict[str, Any]],
    remaining: float,
    source: str,
    reserved_movie_shot_ids: set[str] | None = None,
    max_movie_end: float | None = None,
) -> tuple[float, bool]:
    if remaining <= 0 or not ranges or not movie_shots:
        return remaining, False
    used = _used_movie_shot_ids(ranges) | set(reserved_movie_shot_ids or set())
    last_end = max(float(row.get("end") or 0.0) for row in ranges)
    extended = False
    for shot in movie_shots:
        if remaining <= 0:
            break
        shot_id = str(shot.get("movie_shot_id") or "")
        start = float(shot.get("start") or 0.0)
        end = float(shot.get("end") or 0.0)
        if max_movie_end is not None and start >= max_movie_end - 0.001:
            break
        if shot_id in used or end <= start or start < last_end - 0.05:
            continue
        if max_movie_end is not None:
            end = min(end, max_movie_end)
            if end <= start:
                break
        take = min(end - start, remaining)
        clips.append(
            _clip(
                len(clips) + 1,
                start,
                start + take,
                source,
                movie_shot_ids=[shot_id] if shot_id else [],
                allocation="adjacent_extension",
            )
        )
        remaining = _round(remaining - take)
        extended = True
    return remaining, extended


def allocate_video_clips(
    item: dict[str, Any],
    *,
    tts_duration: float,
    source: str,
    movie_shots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    ranges = _prepare_movie_ranges(item, movie_shots)
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
    reserved_movie_shot_ids = {str(x) for x in item.get("reserved_movie_shot_ids") or []}
    max_movie_end = item.get("max_extension_movie_end")
    max_movie_end = float(max_movie_end) if max_movie_end is not None else None
    remaining, used_adjacent = _extend_with_adjacent_shots(
        clips,
        ranges,
        movie_shots,
        remaining,
        source,
        reserved_movie_shot_ids=reserved_movie_shot_ids,
        max_movie_end=max_movie_end,
    )
    if remaining > 0.03 and clips:
        if max_movie_end is not None:
            return clips, "padded_with_freeze"
        clips[-1]["movie_end"] = _round(float(clips[-1]["movie_end"]) + remaining)
        clips[-1]["duration"] = _round(float(clips[-1]["duration"]) + remaining)
        clips[-1]["allocation"] = "synthetic_extension"
        return clips, "extended_last_clip"
    return clips, "extended_with_adjacent_shots" if used_adjacent else "kept_original"


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
    reserved_movie_shot_ids: set[str],
    max_extension_movie_end: float | None,
    chars_per_second: float,
    min_duration: float,
) -> tuple[dict[str, Any], float]:
    if audio_mode == "original":
        clips = _clips_from_ranges(
            ranges,
            source=source,
            audio_mode=audio_mode,
            allocation="original_audio",
            movie_shots=movie_shots,
        )
        duration = _round(sum(float(clip.get("duration") or 0.0) for clip in clips))
        allocation_status = "original_audio"
        tts_duration = duration
    else:
        tts_duration = estimate_tts_duration(narration, chars_per_second, min_duration) if narration else 0.0
        if audio_mode == "mixed" and tts_duration <= 0:
            tts_duration = _round(sum(_duration(row.get("start"), row.get("end")) for row in ranges))
        chunk_item = dict(item)
        chunk_item["movie_time_ranges"] = ranges
        chunk_item["reserved_movie_shot_ids"] = sorted(reserved_movie_shot_ids | _used_movie_shot_ids(_base_ranges(item)))
        if max_extension_movie_end is not None:
            chunk_item["max_extension_movie_end"] = _round(max_extension_movie_end)
        clips, allocation_status = allocate_video_clips(
            chunk_item,
            tts_duration=tts_duration,
            source=source,
            movie_shots=movie_shots,
        )
        clips = _mark_clip_audio(clips, audio_mode)
        duration = tts_duration

    end = _round(cursor + duration)
    row = {
        "item_id": f"item_{item_index:03d}",
        "segment_id": item.get("segment_id"),
        "audio_mode": audio_mode,
        "OST": 1 if audio_mode in {"original", "mixed"} else 0,
        "narration": "" if audio_mode == "original" else narration,
        "tts_duration": _round(tts_duration),
        "timeline_start": _round(cursor),
        "timeline_end": end,
        "video_clips": clips,
        "ref_source": _ref_source(item, ranges),
        "audio_decision": _range_audio_decision(audio_mode, ranges),
        "confidence": _confidence({**item, "movie_time_ranges": ranges}, clips, allocation_status),
        "allocation_status": allocation_status,
    }
    if audio_mode != "original":
        visual_duration = _round(sum(float(clip.get("duration") or 0.0) for clip in clips))
        freeze_padding = _round(max(0.0, duration - visual_duration))
        if freeze_padding > 0.03:
            row["freeze_padding"] = freeze_padding
    return row, end


def _compose_audio_aware_items(
    item: dict[str, Any],
    *,
    start_index: int,
    cursor: float,
    source: str,
    movie_shots: list[dict[str, Any]],
    reserved_movie_shot_ids: set[str],
    rendered_movie_shot_ids: set[str],
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
    consumed_range_ids: set[str] = set()
    for chunk_index, chunk in enumerate(chunks):
        audio_mode = str(chunk.get("audio_mode") or "voiceover")
        range_ids = [str(x) for x in chunk.get("range_ids") or []]
        ranges = _unused_ranges(range_ids, by_id, consumed_range_ids)
        if audio_mode in {"voiceover", "mixed"}:
            ranges = _drop_rendered_voiceover_ranges(ranges, rendered_movie_shot_ids)
        chunk_units = [unit for unit in chunk.get("units") or [] if isinstance(unit, dict)]
        demoted_short_original = audio_mode == "original" and _has_short_original_play_range(ranges)
        if demoted_short_original:
            audio_mode = "voiceover"
        if audio_mode == "original" and not ranges:
            continue
        narration = _join_text(
            [
                _unit_voiceover_text(unit)
                for unit in chunk_units
                if demoted_short_original or not unit.get("keep_original_audio")
            ]
        )
        if not narration and audio_mode in {"voiceover", "mixed"}:
            narration = _join_text([_unit_voiceover_text(unit) for unit in chunk_units])
        if not narration and audio_mode == "voiceover" and not units:
            narration = str(item.get("new_text") or item.get("old_text") or "")
        if audio_mode in {"voiceover", "mixed"} and not narration and not ranges:
            continue
        future_starts = []
        for future_chunk in chunks[chunk_index + 1 :]:
            future_ids = [str(x) for x in future_chunk.get("range_ids") or []]
            future_start = _min_range_start(_ranges_for_ids(future_ids, by_id))
            if future_start is not None:
                future_starts.append(future_start)
        max_extension_movie_end = min(future_starts) if future_starts else None
        row, cursor = _timeline_item(
            item_index=next_index,
            item=item,
            ranges=ranges,
            narration=narration,
            audio_mode=audio_mode,
            cursor=cursor,
            source=source,
            movie_shots=movie_shots,
            reserved_movie_shot_ids=reserved_movie_shot_ids | rendered_movie_shot_ids,
            max_extension_movie_end=max_extension_movie_end,
            chars_per_second=chars_per_second,
            min_duration=min_duration,
        )
        rows.append(row)
        rendered_movie_shot_ids.update(_used_movie_shot_ids(row.get("video_clips") or []))
        for range_row in ranges:
            range_id = str(range_row.get("range_id") or "")
            if range_id:
                consumed_range_ids.add(range_id)
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
    reserved_movie_shot_ids: set[str] = set()
    final_timeline = []
    rendered_movie_shot_ids: set[str] = set()
    cursor = 0.0
    item_index = 1
    for item in rewritten_script:
        if _has_audio_decisions(item):
            rows, cursor = _compose_audio_aware_items(
                item,
                start_index=item_index,
                cursor=cursor,
                source=source,
                movie_shots=movie_shots,
                reserved_movie_shot_ids=reserved_movie_shot_ids,
                rendered_movie_shot_ids=rendered_movie_shot_ids,
                chars_per_second=chars_per_second,
                min_duration=min_duration,
            )
            final_timeline.extend(rows)
            item_index += len(rows)
            continue

        narration = str(item.get("new_text") or item.get("old_text") or "")
        tts_duration = estimate_tts_duration(narration, chars_per_second, min_duration)
        clips, allocation_status = allocate_video_clips(
            {**item, "reserved_movie_shot_ids": sorted(reserved_movie_shot_ids | rendered_movie_shot_ids)},
            tts_duration=tts_duration,
            source=source,
            movie_shots=movie_shots,
        )
        ref_range = item.get("ref_time_range") or {}
        ref_shot_ids = [
            str(row.get("source_ref_shot_id"))
            for row in item.get("movie_time_ranges") or []
            if row.get("source_ref_shot_id")
        ]
        final_timeline.append(
            {
                "item_id": f"item_{item_index:03d}",
                "segment_id": item.get("segment_id"),
                "audio_mode": "voiceover",
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
                "audio_decision": {
                    "audio_role": "narration_overlay",
                    "audio_action": "rewrite_and_voiceover",
                    "source_range_ids": [str(row.get("range_id")) for row in item.get("movie_time_ranges") or [] if row.get("range_id")],
                },
                "confidence": _confidence(item, clips, allocation_status),
                "allocation_status": allocation_status,
            }
        )
        rendered_movie_shot_ids.update(_used_movie_shot_ids(clips))
        cursor = _round(cursor + tts_duration)
        item_index += 1
    return {
        "final_timeline": final_timeline,
        "timeline_backend": {
            "tts_duration_estimator": "char_count",
            "chars_per_second": chars_per_second,
            "min_duration": min_duration,
            "movie_shot_extension": bool(movie_shots),
            "audio_decision_split": any(_has_audio_decisions(item) for item in rewritten_script),
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


if __name__ == "__main__":
    main()
