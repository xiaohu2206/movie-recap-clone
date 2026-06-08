from __future__ import annotations

import json
import re
from typing import Any

from clone_narration_video.utils.ai import ChatMessage, CustomOpenAIProvider


ALLOWED_AUDIO_PATTERNS = {
    "all_narration",
    "all_original_audio",
    "narration_then_original_audio",
    "original_audio_then_narration",
}

SYSTEM_PROMPT = """你是影视解说视频的音频结构判断器。
你只判断每个 segment 中哪些连续 clip 应该使用 AI 解说，哪些连续 clip 应保留原片人物对白/原声。
硬性规则：
1. 不能改变 clip 顺序，不能改任何 clip 的 start/end。
2. 一个 segment 最多只能切一次，只允许连续的两段结构。
3. 禁止输出“解说→原声→解说”或“原声→解说→原声”的穿插结构。
4. movie_subtitles 像角色对白，且与 old_text 中某部分高度对应时，更倾向原声。
5. movie_subtitles 为空或只是旁白、剧情解释时，更倾向解说。
6. 只输出 JSON，不要解释文字。
输出字段只能包含 results；每个 result 只包含 segment_id、audio_pattern、split_clip_index。"""


def _extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S | re.I)
    if fenced:
        raw = fenced.group(1).strip()
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("AI 返回不是 JSON object")
    return data


def _subtitle_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    subtitles = row.get("movie_subtitles") or []
    if not isinstance(subtitles, list):
        value = str(subtitles).strip()
        return [value] if value else []
    for sub in subtitles:
        if isinstance(sub, dict):
            text = str(sub.get("text") or "").strip()
        else:
            text = str(sub).strip()
        if text:
            texts.append(text)
    return texts


def _clip_index(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("clip_index"))
    except Exception:
        return fallback


def _payload_segment(segment: dict[str, Any]) -> dict[str, Any]:
    clips = []
    for fallback, row in enumerate(segment.get("movie_time_ranges") or []):
        clips.append(
            {
                "clip_index": _clip_index(row, fallback),
                "start": row.get("start"),
                "end": row.get("end"),
                "movie_subtitles": _subtitle_texts(row),
            }
        )
    clips.sort(key=lambda x: int(x.get("clip_index") or 0))
    return {
        "segment_id": segment.get("segment_id"),
        "old_text": segment.get("old_text") or segment.get("text") or "",
        "clips": clips,
    }


def _build_payload(segments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": "decide_audio_pattern",
        "rules": [
            "不能改变镜头顺序，不能改 clip 的 start/end",
            "一个 segment 内最多只能切一次，split_clip_index 表示第二部分起始的 clip_index",
            "禁止解说和原声穿插，只能输出四种 allowed_audio_patterns 之一",
            "movie_subtitles 像角色对白且与 old_text 对应时倾向原声；为空时倾向解说",
        ],
        "allowed_audio_patterns": sorted(ALLOWED_AUDIO_PATTERNS),
        "segments": [_payload_segment(segment) for segment in segments],
        "output_schema": {
            "results": [
                {
                    "segment_id": "seg_001",
                    "audio_pattern": "narration_then_original_audio",
                    "split_clip_index": 3,
                }
            ]
        },
    }


def _fallback() -> dict[str, Any]:
    return {"audio_pattern": "all_narration", "split_clip_index": None, "source": "fallback"}


def _valid_split(segment: dict[str, Any], split: Any) -> int | None:
    try:
        value = int(split)
    except Exception:
        return None
    indexes = [_clip_index(row, idx) for idx, row in enumerate(segment.get("movie_time_ranges") or [])]
    if len(indexes) < 2:
        return None
    ordered = sorted(indexes)
    if value not in ordered or value <= ordered[0]:
        return None
    return value


def _validate_decision(segment: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return _fallback()
    pattern = str(row.get("audio_pattern") or "").strip()
    if pattern not in ALLOWED_AUDIO_PATTERNS:
        return _fallback()
    if pattern in {"all_narration", "all_original_audio"}:
        return {"audio_pattern": pattern, "split_clip_index": None, "source": "llm"}
    split = _valid_split(segment, row.get("split_clip_index"))
    if split is None:
        return _fallback()
    return {"audio_pattern": pattern, "split_clip_index": split, "source": "llm"}


async def classify_segments(
    segments: list[dict[str, Any]],
    *,
    provider: CustomOpenAIProvider,
    batch_size: int = 8,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    size = max(1, int(batch_size))
    for start in range(0, len(segments), size):
        batch = segments[start : start + size]
        payload = _build_payload(batch)
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        try:
            try:
                response = await provider.chat_completion(messages, extra_params={"response_format": {"type": "json_object"}})
            except Exception:
                response = await provider.chat_completion(messages)
            parsed = _extract_json(response.content)
            rows = parsed.get("results") or []
            if not isinstance(rows, list):
                rows = []
            by_segment = {str(row.get("segment_id") or ""): row for row in rows if isinstance(row, dict)}
        except Exception:
            by_segment = {}
        for segment in batch:
            seg_id = str(segment.get("segment_id") or "")
            results[seg_id] = _validate_decision(segment, by_segment.get(seg_id))
    return results
