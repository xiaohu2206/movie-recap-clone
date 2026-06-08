from __future__ import annotations

from pathlib import Path
from typing import Any

from .subtitle_tools import parse_srt


def _round(value: float) -> float:
    return round(float(value), 3)


def _duration(start: Any, end: Any) -> float:
    try:
        return max(0.0, float(end) - float(start))
    except Exception:
        return 0.0


def _item_duration_from_clips(clips: list[dict[str, Any]]) -> float:
    total = 0.0
    for clip in clips:
        total += _duration(clip.get("movie_start"), clip.get("movie_end"))
    return _round(total)


def retime_clips_to_audio(clips: list[dict[str, Any]], target_duration: float) -> list[dict[str, Any]]:
    if not clips:
        return []
    current = _item_duration_from_clips(clips)
    if current <= 0 or target_duration <= 0:
        return [dict(c) for c in clips]

    if target_duration <= current:
        ratio = target_duration / current
        adjusted: list[dict[str, Any]] = []
        used = 0.0
        for idx, clip in enumerate(clips):
            start = float(clip.get("movie_start") or 0.0)
            end = float(clip.get("movie_end") or 0.0)
            src_dur = max(0.0, end - start)
            take = max(0.0, target_duration - used) if idx == len(clips) - 1 else src_dur * ratio
            if take <= 0.03:
                continue
            row = dict(clip)
            row["movie_start"] = _round(start)
            row["movie_end"] = _round(start + take)
            row["duration"] = _round(take)
            adjusted.append(row)
            used += take
        return adjusted

    adjusted = [dict(c) for c in clips]
    extra = target_duration - current
    last = adjusted[-1]
    last["movie_end"] = _round(float(last.get("movie_end") or 0.0) + extra)
    last["duration"] = _round(float(last.get("duration") or 0.0) + extra)
    return adjusted


def _subtitle_text_for_range(
    entries: list[dict[str, Any]],
    start: float,
    end: float,
) -> str:
    if not entries or end <= start:
        return ""
    texts: list[str] = []
    for row in entries:
        row_start = float(row.get("start") or 0.0)
        row_end = float(row.get("end") or 0.0)
        overlap = min(row_end, end) - max(row_start, start)
        if overlap <= 0:
            continue
        text = str(row.get("text") or "").strip()
        if text:
            texts.append(text)
    return " ".join(texts)


def _index_ref_shots(ref_analysis: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not ref_analysis:
        return {}
    rows = ref_analysis.get("ref_shots") or []
    if not isinstance(rows, list):
        return {}
    return {str(row.get("ref_shot_id")): row for row in rows if row.get("ref_shot_id")}


def _index_segments(script_mapping: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not script_mapping:
        return {}
    rows = script_mapping.get("script_mapping") or []
    if not isinstance(rows, list):
        return {}
    return {str(row.get("segment_id")): row for row in rows if row.get("segment_id")}


def _find_movie_range(
    segment: dict[str, Any] | None,
    movie_start: float,
    movie_end: float,
) -> dict[str, Any] | None:
    if not segment:
        return None
    best: dict[str, Any] | None = None
    best_overlap = 0.0
    for row in segment.get("movie_time_ranges") or []:
        start = float(row.get("start") or 0.0)
        end = float(row.get("end") or 0.0)
        overlap = min(end, movie_end) - max(start, movie_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = row
    return best


def _is_original_play(
    clip: dict[str, Any],
    item: dict[str, Any],
    movie_range: dict[str, Any] | None,
) -> bool:
    if clip.get("keep_original_audio") is True:
        return True
    if clip.get("keep_original_audio") is False:
        return False
    if str(item.get("audio_mode") or "").lower() in {"original", "ost", "keep_original"}:
        return True
    if int(item.get("OST") or 0) == 1:
        return True
    if movie_range:
        action = str(movie_range.get("audio_action") or "").lower()
        if action in {"keep_original_audio", "play_original", "keep_original"}:
            return True
        role = str(movie_range.get("audio_role") or "").lower()
        if role in {"original_dialogue", "original_audio", "movie_dialogue"}:
            return True
    return False


def _movie_subtitle_for_clip(
    movie_range: dict[str, Any] | None,
    movie_subtitle_entries: list[dict[str, Any]],
    movie_start: float,
    movie_end: float,
) -> str:
    if movie_range:
        embedded = movie_range.get("movie_subtitles")
        if isinstance(embedded, list) and embedded:
            texts = [str(x.get("text") or x).strip() for x in embedded if str(x.get("text") or x).strip()]
            if texts:
                return " ".join(texts)
    return _subtitle_text_for_range(movie_subtitle_entries, movie_start, movie_end)


def _ref_time_range_for_clip(
    clip: dict[str, Any],
    item: dict[str, Any],
    ref_shots: dict[str, dict[str, Any]],
    *,
    clip_start_in_item: float,
    clip_end_in_item: float,
    item_duration: float,
) -> tuple[float, float]:
    ref_shot_id = clip.get("source_ref_shot_id")
    if ref_shot_id:
        ref_shot = ref_shots.get(str(ref_shot_id))
        if ref_shot:
            return float(ref_shot.get("start") or 0.0), float(ref_shot.get("end") or 0.0)

    ref_source = item.get("ref_source") or {}
    ref_range = item.get("ref_time_range") or {}
    ref_start = float(ref_source.get("ref_start") if ref_source.get("ref_start") is not None else ref_range.get("start") or 0.0)
    ref_end = float(ref_source.get("ref_end") if ref_source.get("ref_end") is not None else ref_range.get("end") or ref_start)
    if ref_end <= ref_start:
        return ref_start, ref_end

    allocation = str(clip.get("allocation") or "")
    if not ref_shot_id and allocation in {"adjacent_extension", "synthetic_extension"}:
        return ref_start, ref_end

    if item_duration <= 0:
        return ref_start, ref_end

    ratio_start = max(0.0, min(1.0, clip_start_in_item / item_duration))
    ratio_end = max(ratio_start, min(1.0, clip_end_in_item / item_duration))
    span = ref_end - ref_start
    return ref_start + span * ratio_start, ref_start + span * ratio_end


def _load_subtitle_entries(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    candidate = Path(path)
    if not candidate.exists():
        return []
    try:
        return parse_srt(candidate)
    except Exception:
        return []


def resolve_pipeline_context(
    *,
    output_root: Path | None = None,
    ref_analysis_path: str | Path | None = None,
    script_mapping_path: str | Path | None = None,
    ref_subtitle_path: str | Path | None = None,
    movie_subtitle_path: str | Path | None = None,
    ref_analysis: dict[str, Any] | None = None,
    script_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = output_root or Path(__file__).resolve().parents[1] / "outputs"

    def _pick(explicit: str | Path | None, step: str, filename: str) -> Path | None:
        if explicit:
            path = Path(explicit)
            return path if path.exists() else None
        candidate = root / step / filename
        return candidate if candidate.exists() else None

    from .json_io import read_json

    ref_path = _pick(ref_analysis_path, "1_reference_analyzer", "ref_analysis.json")
    if ref_analysis is None and ref_path:
        ref_analysis = read_json(ref_path)

    mapping_path = _pick(script_mapping_path, "5_script_visual_binder", "script_mapping.json")
    if script_mapping is None and mapping_path:
        script_mapping = read_json(mapping_path)
    if script_mapping is None:
        alt_mapping = _pick(script_mapping_path, "5.2_audio_role_classifier", "script_mapping_with_audio.json")
        if alt_mapping:
            script_mapping = read_json(alt_mapping)

    ref_srt_path = ref_subtitle_path
    if not ref_srt_path and ref_analysis:
        ref_srt_path = ref_analysis.get("subtitle_srt")
    if not ref_srt_path:
        ref_srt_path = _pick(None, "1_reference_analyzer", "ref_subtitle.srt")

    movie_srt_path = movie_subtitle_path or _pick(None, "5.1_movie_subtitle_filler", "movie_subtitle.srt")
    if not movie_srt_path:
        movie_srt_path = _pick(None, "3_movie_shot_parser", "movie_subtitle.srt")

    return {
        "ref_analysis": ref_analysis,
        "script_mapping": script_mapping,
        "ref_subtitle_entries": _load_subtitle_entries(ref_srt_path),
        "movie_subtitle_entries": _load_subtitle_entries(movie_srt_path),
        "ref_shots": _index_ref_shots(ref_analysis),
        "segments_by_id": _index_segments(script_mapping),
    }


def build_shot_breakdown(
    items: list[dict[str, Any]],
    *,
    audio_results: dict[str, dict[str, Any]] | None = None,
    ref_analysis: dict[str, Any] | None = None,
    script_mapping: dict[str, Any] | None = None,
    ref_subtitle_entries: list[dict[str, Any]] | None = None,
    movie_subtitle_entries: list[dict[str, Any]] | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if ref_subtitle_entries is None or movie_subtitle_entries is None or ref_analysis is None or script_mapping is None:
        ctx = resolve_pipeline_context(
            output_root=output_root,
            ref_analysis=ref_analysis,
            script_mapping=script_mapping,
        )
        ref_analysis = ref_analysis or ctx["ref_analysis"]
        script_mapping = script_mapping or ctx["script_mapping"]
        ref_subtitle_entries = ref_subtitle_entries if ref_subtitle_entries is not None else ctx["ref_subtitle_entries"]
        movie_subtitle_entries = movie_subtitle_entries if movie_subtitle_entries is not None else ctx["movie_subtitle_entries"]

    ref_shots = _index_ref_shots(ref_analysis)
    segments_by_id = _index_segments(script_mapping)
    ref_subtitle_entries = ref_subtitle_entries or []
    movie_subtitle_entries = movie_subtitle_entries or []

    shots: list[dict[str, Any]] = []
    timeline_cursor = 0.0

    for item in items:
        key = str(item.get("item_id") or item.get("segment_id") or "")
        segment_id = str(item.get("segment_id") or "")
        narration = str(item.get("narration") or item.get("new_text") or item.get("old_text") or "")
        segment = segments_by_id.get(segment_id) or {}

        if audio_results and key in audio_results:
            item_duration = float(audio_results[key].get("duration") or 0.0)
        else:
            item_duration = float(item.get("tts_duration") or 0.0)
        if item_duration <= 0:
            item_duration = _item_duration_from_clips(item.get("video_clips") or [])

        base_clips = [c for c in (item.get("video_clips") or []) if isinstance(c, dict)]
        clips = retime_clips_to_audio(base_clips, item_duration)
        clip_cursor_in_item = 0.0

        for clip in clips:
            movie_start = float(clip.get("movie_start") or 0.0)
            movie_end = float(clip.get("movie_end") or 0.0)
            duration = _round(_duration(movie_start, movie_end))
            start = _round(timeline_cursor)
            end = _round(timeline_cursor + duration)

            movie_range = _find_movie_range(segment, movie_start, movie_end)
            clip_start_in_item = clip_cursor_in_item
            clip_end_in_item = clip_cursor_in_item + duration
            ref_start, ref_end = _ref_time_range_for_clip(
                clip,
                {**segment, **item},
                ref_shots,
                clip_start_in_item=clip_start_in_item,
                clip_end_in_item=clip_end_in_item,
                item_duration=item_duration,
            )

            shots.append(
                {
                    "item_id": item.get("item_id"),
                    "clip_id": clip.get("clip_id"),
                    "segment_id": segment_id,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "is_original_play": _is_original_play(clip, item, movie_range),
                    "narration": narration,
                    "segment_text": narration,
                    "shot_subtitle": _movie_subtitle_for_clip(
                        movie_range,
                        movie_subtitle_entries,
                        movie_start,
                        movie_end,
                    ),
                    "ref_subtitle": _subtitle_text_for_range(ref_subtitle_entries, ref_start, ref_end),
                    "movie_start": _round(movie_start),
                    "movie_end": _round(movie_end),
                    "source_ref_shot_id": clip.get("source_ref_shot_id"),
                    "movie_shot_ids": clip.get("movie_shot_ids") or [],
                }
            )
            timeline_cursor = end
            clip_cursor_in_item = clip_end_in_item

    return {
        "shot_count": len(shots),
        "total_duration": _round(timeline_cursor),
        "shots": shots,
    }
