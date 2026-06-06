from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.project_paths import default_output_dir
from clone_narration_video.utils.subtitle_tools import extract_srt_with_bcut, parse_srt, write_srt
from clone_narration_video.utils.ffmpeg_utils import resolve_ffmpeg_bin, run_ffmpeg


def _round(value: Any) -> float:
    return round(float(value or 0.0), 3)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _subtitle_rows(subtitles: list[dict[str, Any]], start: Any, end: Any) -> list[dict[str, Any]]:
    s = float(start or 0.0)
    e = float(end or 0.0)
    rows = []
    for sub in subtitles:
        sub_start = float(sub.get("start") or 0.0)
        sub_end = float(sub.get("end") or 0.0)
        if _overlap(s, e, sub_start, sub_end) <= 0:
            continue
        rows.append(
            {
                "index": sub.get("index"),
                "start": _round(sub_start),
                "end": _round(sub_end),
                "text": str(sub.get("text") or "").strip(),
            }
        )
    return rows


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(ch for ch in normalized if ch.isalnum())


def _similarity(a: str, b: str) -> float:
    left = _normalize_text(a)
    right = _normalize_text(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _combined_text(rows: list[dict[str, Any]]) -> str:
    return "".join(str(row.get("text") or "").strip() for row in rows)


def _shot_lookup(ref_analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    shots = ref_analysis.get("ref_shots") or []
    if not isinstance(shots, list):
        return {}
    result = {}
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        shot_id = str(shot.get("ref_shot_id") or shot.get("shot_id") or "")
        if shot_id:
            result[shot_id] = shot
    return result


def _range_ref_overlap(unit: dict[str, Any], row: dict[str, Any], shots_by_id: dict[str, dict[str, Any]]) -> float:
    shot = shots_by_id.get(str(row.get("source_ref_shot_id") or ""))
    if not shot:
        return 0.0
    return _overlap(
        float(unit.get("ref_start") or 0.0),
        float(unit.get("ref_end") or 0.0),
        float(shot.get("start") or 0.0),
        float(shot.get("end") or 0.0),
    )


def _range_id(segment_id: str, idx: int) -> str:
    return f"{segment_id}_range_{idx:03d}"


def _unit_id(segment_id: str, idx: int) -> str:
    return f"{segment_id}_unit_{idx:03d}"


def _best_match(
    ref_texts: list[str],
    movie_subtitles: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    best_score = 0.0
    best_rows: list[dict[str, Any]] = []
    movie_candidates = [*movie_subtitles]
    if len(movie_subtitles) > 1:
        movie_candidates.append({"text": _combined_text(movie_subtitles), "start": movie_subtitles[0]["start"], "end": movie_subtitles[-1]["end"]})

    for ref_text in ref_texts:
        for movie_row in movie_candidates:
            score = _similarity(ref_text, str(movie_row.get("text") or ""))
            if score > best_score:
                best_score = score
                if movie_row in movie_subtitles:
                    best_rows = [movie_row]
                else:
                    best_rows = movie_subtitles
    return best_score, best_rows


def _decide_range(
    related_units: list[dict[str, Any]],
    movie_subtitles: list[dict[str, Any]],
    *,
    original_threshold: float,
    review_threshold: float,
    min_dialogue_chars: int,
) -> tuple[str, str, float, str, list[dict[str, Any]]]:
    if not movie_subtitles:
        return (
            "narration_overlay",
            "rewrite_and_voiceover",
            0.9,
            "该时间段未匹配到原电影对白，默认由解说覆盖",
            [],
        )

    ref_texts = [str(unit.get("text") or "") for unit in related_units if str(unit.get("text") or "").strip()]
    if not ref_texts:
        return (
            "narration_overlay",
            "rewrite_and_voiceover",
            0.75,
            "未切出可比较的参考字幕，按解说覆盖处理",
            [],
        )
    best_score, best_rows = _best_match(ref_texts, movie_subtitles)
    movie_norm_len = len(_normalize_text(_combined_text(movie_subtitles)))

    if movie_norm_len >= min_dialogue_chars and best_score >= original_threshold:
        return (
            "original_dialogue",
            "play_original_audio",
            round(min(0.99, max(0.82, best_score)), 3),
            "参考字幕和原电影字幕高度相似，应保留原片对白",
            best_rows,
        )
    if best_score >= review_threshold:
        return (
            "unknown",
            "manual_review",
            round(best_score, 3),
            "参考字幕和原电影字幕部分相似，需要人工复查",
            best_rows,
        )
    return (
        "narration_overlay",
        "rewrite_and_voiceover",
        round(max(0.7, 1.0 - best_score), 3),
        "参考字幕和原电影字幕相似度不足，按解说覆盖处理",
        [],
    )


def _aggregate_segment_role(ranges: list[dict[str, Any]]) -> str:
    actions = {str(row.get("audio_action") or "") for row in ranges}
    if "manual_review" in actions:
        return "manual_review"
    if actions == {"rewrite_and_voiceover"}:
        return "narration_overlay"
    if actions == {"play_original_audio"}:
        return "original_dialogue"
    if {"rewrite_and_voiceover", "play_original_audio"} <= actions:
        return "mixed"
    return "unknown"


def _movie_ranges_from_mapping(script_mapping_rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    ranges = []
    for seg in script_mapping_rows:
        if not isinstance(seg, dict):
            continue
        for row in seg.get("movie_time_ranges") or []:
            try:
                start = float(row.get("start") or 0.0)
                end = float(row.get("end") or 0.0)
            except Exception:
                continue
            if end > start:
                ranges.append((start, end))
    return sorted(ranges)


def _merge_time_ranges(
    ranges: list[tuple[float, float]],
    *,
    padding: float,
    max_gap: float,
) -> list[tuple[float, float]]:
    padded = [(max(0.0, start - padding), end + padding) for start, end in ranges if end > start]
    if not padded:
        return []
    padded.sort()
    merged: list[tuple[float, float]] = []
    cur_start, cur_end = padded[0]
    for start, end in padded[1:]:
        if start <= cur_end + max(0.0, max_gap):
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return [(round(start, 3), round(end, 3)) for start, end in merged if end > start]


def _extract_audio_clip_mp3(video_path: str | Path, audio_path: str | Path, start: float, end: float) -> Path:
    out = Path(audio_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return out
    duration = max(0.01, float(end) - float(start))
    cmd = [
        resolve_ffmpeg_bin(),
        "-y",
        "-ss",
        f"{float(start):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "3",
        str(out),
    ]
    run_ffmpeg(cmd)
    return out


def _bcut_entries_for_audio(audio_path: str | Path, offset: float) -> list[dict[str, Any]]:
    from clone_narration_video.utils.asr.asr_bcut import BcutASR

    asr = BcutASR(str(audio_path), use_cache=True)
    data = asr.run()
    utterances = data.get("utterances") if isinstance(data, dict) else None
    if not isinstance(utterances, list):
        raise RuntimeError("Bcut ASR 未返回有效 utterances")
    entries = []
    for item in utterances:
        text = str(item.get("text") or item.get("transcript") or "").strip()
        if not text:
            continue
        start = offset + (float(item.get("start_time") or 0.0) / 1000.0)
        end = offset + (float(item.get("end_time") or 0.0) / 1000.0)
        if end <= start:
            continue
        entries.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return entries


def _run_with_retries(label: str, func: Any, *, retries: int, retry_delay: float) -> Any:
    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                print(f"[audio_role_classifier] {label} retry {attempt}/{attempts}", file=sys.stderr, flush=True)
            return func()
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            print(
                f"[audio_role_classifier] {label} failed ({attempt}/{attempts}): {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(max(0.0, float(retry_delay)))
    raise RuntimeError(f"{label} 失败；可以重试，或先准备原电影字幕并传 --movie-subtitle-srt") from last_error


def _extract_movie_srt_with_retries(
    movie_path: str | Path,
    movie_srt: Path,
    out_dir: Path,
    *,
    retries: int,
    retry_delay: float,
) -> Path:
    movie_srt.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _run_with_retries(
            "Bcut ASR",
            lambda: extract_srt_with_bcut(movie_path, movie_srt, out_dir),
            retries=retries,
            retry_delay=retry_delay,
        )
    except RuntimeError:
        if movie_srt.exists() and movie_srt.stat().st_size > 0:
            return movie_srt
        raise


def _extract_matched_movie_srt_with_retries(
    movie_path: str | Path,
    movie_srt: Path,
    out_dir: Path,
    script_mapping_rows: list[dict[str, Any]],
    *,
    retries: int,
    retry_delay: float,
    padding: float,
    merge_gap: float,
) -> Path:
    movie_srt.parent.mkdir(parents=True, exist_ok=True)
    ranges = _merge_time_ranges(_movie_ranges_from_mapping(script_mapping_rows), padding=padding, max_gap=merge_gap)
    if not ranges:
        movie_srt.write_text("", encoding="utf-8")
        return movie_srt

    entries: list[dict[str, Any]] = []
    clips_dir = out_dir / "movie_subtitle_clips"
    for idx, (start, end) in enumerate(ranges, start=1):
        audio_path = clips_dir / f"range_{idx:04d}_{int(start * 1000)}_{int(end * 1000)}.mp3"
        print(
            f"[audio_role_classifier] ASR matched range {idx}/{len(ranges)}: {start:.3f}-{end:.3f}",
            file=sys.stderr,
            flush=True,
        )
        _extract_audio_clip_mp3(movie_path, audio_path, start, end)
        clip_entries = _run_with_retries(
            f"Bcut ASR range {idx}/{len(ranges)}",
            lambda audio_path=audio_path, start=start: _bcut_entries_for_audio(audio_path, start),
            retries=retries,
            retry_delay=retry_delay,
        )
        entries.extend(clip_entries)

    entries.sort(key=lambda x: (float(x.get("start") or 0.0), float(x.get("end") or 0.0)))
    write_srt(movie_srt, entries)
    return movie_srt


def _unit_decision(
    unit: dict[str, Any],
    related_ranges: list[dict[str, Any]],
) -> tuple[str, str, float, str, list[dict[str, Any]]]:
    original_ranges = [row for row in related_ranges if row.get("audio_action") == "play_original_audio"]
    review_ranges = [row for row in related_ranges if row.get("audio_action") == "manual_review"]
    if original_ranges:
        matched: list[dict[str, Any]] = []
        for row in original_ranges:
            for sub in row.get("movie_subtitles") or []:
                score = _similarity(str(unit.get("text") or ""), str(sub.get("text") or ""))
                if score >= 0.65:
                    item = {k: sub[k] for k in ("start", "end", "text") if k in sub}
                    item["similarity"] = round(score, 3)
                    matched.append(item)
        confidence = max([float(row.get("audio_confidence") or 0.0) for row in original_ranges] or [0.9])
        return "original_dialogue", "play_original_audio", round(confidence, 3), "参考字幕与原电影字幕文本一致", matched
    if review_ranges:
        confidence = max([float(row.get("audio_confidence") or 0.0) for row in review_ranges] or [0.65])
        return "unknown", "manual_review", round(confidence, 3), "该字幕与原电影字幕部分相似，需要人工复查", []
    return "narration", "rewrite", 0.9, "第三方叙述或未在原电影字幕中找到相似对白", []


def classify_audio_roles(
    script_mapping_data: dict[str, Any],
    ref_analysis: dict[str, Any],
    *,
    movie_subtitle_srt: str | Path | None = None,
    movie_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    original_threshold: float = 0.82,
    review_threshold: float = 0.65,
    min_dialogue_chars: int = 3,
    asr_retries: int = 3,
    asr_retry_delay: float = 2.0,
    movie_asr_padding: float = 0.2,
    movie_asr_merge_gap: float = 0.5,
) -> dict[str, Any]:
    subtitle_srt = ref_analysis.get("subtitle_srt")
    if not subtitle_srt:
        raise ValueError("ref_analysis 缺少 subtitle_srt")
    subtitles = parse_srt(subtitle_srt)
    rows = script_mapping_data.get("script_mapping") or []
    if not isinstance(rows, list):
        raise ValueError("script_mapping.json 缺少 script_mapping 数组")

    out_dir = Path(output_dir or default_output_dir("5_audio_role_classifier"))
    if movie_subtitle_srt:
        movie_srt = Path(movie_subtitle_srt)
        if not movie_srt.exists():
            raise ValueError(f"movie_subtitle_srt 不存在: {movie_srt}")
    else:
        if not movie_path:
            raise ValueError("缺少 movie_subtitle_srt；如需自动识别，请传 movie_path")
        movie_srt = out_dir / "movie_subtitle.srt"
        has_movie_ranges = bool(_movie_ranges_from_mapping(rows))
        if not movie_srt.exists() or (has_movie_ranges and movie_srt.stat().st_size == 0):
            _extract_matched_movie_srt_with_retries(
                movie_path,
                movie_srt,
                out_dir,
                rows,
                retries=asr_retries,
                retry_delay=asr_retry_delay,
                padding=movie_asr_padding,
                merge_gap=movie_asr_merge_gap,
            )
    movie_subtitles = parse_srt(movie_srt)

    shots_by_id = _shot_lookup(ref_analysis)

    enhanced_rows: list[dict[str, Any]] = []
    for seg in rows:
        if not isinstance(seg, dict):
            continue
        item = dict(seg)
        segment_id = str(item.get("segment_id") or f"seg_{len(enhanced_rows) + 1:03d}")
        ref_range = item.get("ref_time_range") or {}
        text_units = []
        for idx, sub in enumerate(_subtitle_rows(subtitles, ref_range.get("start"), ref_range.get("end")), start=1):
            text_units.append(
                {
                    "unit_id": _unit_id(segment_id, idx),
                    "source_ref_subtitle_index": sub.get("index"),
                    "ref_start": sub["start"],
                    "ref_end": sub["end"],
                    "text": sub["text"],
                    "role": "narration",
                    "action": "rewrite",
                    "related_range_ids": [],
                    "matched_movie_subtitles": [],
                    "confidence": 0.9,
                    "reason": "待规则判断",
                }
            )

        ranges = []
        for range_idx, row in enumerate(item.get("movie_time_ranges") or [], start=1):
            range_item = dict(row)
            range_item["range_id"] = str(range_item.get("range_id") or _range_id(segment_id, range_idx))
            range_item["movie_subtitles"] = [
                {k: sub[k] for k in ("start", "end", "text") if k in sub}
                for sub in _subtitle_rows(movie_subtitles, range_item.get("start"), range_item.get("end"))
            ]
            ranges.append(range_item)

        for range_item in ranges:
            related_units = [unit for unit in text_units if _range_ref_overlap(unit, range_item, shots_by_id) > 0]
            if not related_units:
                related_units = text_units
            role, action, confidence, reason, _ = _decide_range(
                related_units,
                range_item.get("movie_subtitles") or [],
                original_threshold=original_threshold,
                review_threshold=review_threshold,
                min_dialogue_chars=min_dialogue_chars,
            )
            range_item.update(
                {
                    "audio_role": role,
                    "audio_action": action,
                    "audio_confidence": confidence,
                    "audio_reason": reason,
                    "visual_match_locked": True,
                }
            )

        for unit in text_units:
            related = [row for row in ranges if _range_ref_overlap(unit, row, shots_by_id) > 0]
            if not related and ranges:
                related = ranges
            unit["related_range_ids"] = [str(row.get("range_id")) for row in related]
            role, action, confidence, reason, matched = _unit_decision(unit, related)
            unit.update(
                {
                    "role": role,
                    "action": action,
                    "matched_movie_subtitles": matched,
                    "confidence": confidence,
                    "reason": reason,
                }
            )

        item["text_units"] = text_units
        item["movie_time_ranges"] = ranges
        item["segment_audio_role"] = _aggregate_segment_role(ranges)
        enhanced_rows.append(item)

    return {
        "script_mapping": enhanced_rows,
        "audio_role_backend": {
            "movie_subtitle_srt": str(movie_srt),
            "decision_mode": "rules_only",
            "version": "audio_role_classifier_v1",
            "original_threshold": original_threshold,
            "review_threshold": review_threshold,
            "asr_retries": asr_retries,
            "movie_asr_scope": "matched_movie_time_ranges",
            "movie_asr_padding": movie_asr_padding,
            "movie_asr_merge_gap": movie_asr_merge_gap,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="原声片段识别模块")
    parser.add_argument("--script-mapping", required=True, help="第 5 步输出的 script_mapping.json")
    parser.add_argument("--ref-analysis", required=True, help="第 1 步输出的 ref_analysis.json")
    parser.add_argument("--movie-path", help="原电影路径；未传 movie-subtitle-srt 时用于 ASR 生成字幕")
    parser.add_argument("--movie-subtitle-srt", help="已有原电影字幕 SRT；传入后跳过 ASR")
    parser.add_argument("--output-dir", default=str(default_output_dir("5_audio_role_classifier")))
    parser.add_argument("--original-threshold", type=float, default=0.82)
    parser.add_argument("--review-threshold", type=float, default=0.65)
    parser.add_argument("--min-dialogue-chars", type=int, default=3)
    parser.add_argument("--asr-retries", type=int, default=3, help="自动识别原电影字幕失败时的重试次数")
    parser.add_argument("--asr-retry-delay", type=float, default=2.0, help="Bcut ASR 重试间隔秒数")
    parser.add_argument("--movie-asr-padding", type=float, default=0.2, help="匹配镜头前后额外识别的秒数")
    parser.add_argument("--movie-asr-merge-gap", type=float, default=0.5, help="相邻匹配镜头间隔小于该秒数时合并识别")
    args = parser.parse_args()

    result = classify_audio_roles(
        read_json(args.script_mapping),
        read_json(args.ref_analysis),
        movie_subtitle_srt=args.movie_subtitle_srt,
        movie_path=args.movie_path,
        output_dir=args.output_dir,
        original_threshold=args.original_threshold,
        review_threshold=args.review_threshold,
        min_dialogue_chars=args.min_dialogue_chars,
        asr_retries=args.asr_retries,
        asr_retry_delay=args.asr_retry_delay,
        movie_asr_padding=args.movie_asr_padding,
        movie_asr_merge_gap=args.movie_asr_merge_gap,
    )
    out = write_json(Path(args.output_dir) / "script_mapping_with_audio.json", result)
    print(out)


if __name__ == "__main__":
    main()
