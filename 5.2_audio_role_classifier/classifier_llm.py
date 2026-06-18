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

ORIGINAL_AUDIO_BUDGET_RATIO = 0.30
MIN_ORIGINAL_BLOCK_SECONDS = 3.0

SYSTEM_PROMPT = """你是影视解说视频的原声精选导演。

任务：从整条解说视频的所有候选镜头中，只挑少量最值得保留原片原声的连续片段，其余都使用 AI 解说。

硬性策略：
1. 原声播放是稀缺资源：全片原声总量目标不超过所有候选镜头的 30%。
2. 每个被保留的原声连续片段时长必须 >= 3 秒；短对白、碎片对白、过场声不要选。
3. 一个 segment 最多切一次，只允许四种结构：
   - all_narration
   - all_original_audio
   - narration_then_original_audio
   - original_audio_then_narration
4. split_clip_index 表示第二部分起始的 clip_index。
5. 不要做逐字匹配、字符串相似度匹配或依赖字幕语言；字幕可能是中文、英文或其他语言。
6. 可以把字幕文本当作语义线索：优先保留情绪爆点、冲突对白、关键信息、反转、角色名场面。
7. 不要因为某个 clip 有字幕就保留原声；字幕时间证据只说明这里可能有人说话。
8. hook 和 ending 通常更偏 AI 解说，除非出现非常强的角色对白爆点。
9. 如果不确定，宁可 all_narration。

输出只能是 JSON。字段只能包含 results；每个 result 只包含：
- segment_id
- audio_pattern
- split_clip_index
- priority

priority 是 0-100 的保留原声优先级；越值得挤进 30% 原声预算，分数越高。all_narration 可填 0。"""


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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clip_index(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("clip_index"))
    except Exception:
        return fallback


def _duration(row: dict[str, Any]) -> float:
    return max(0.0, _float(row.get("end")) - _float(row.get("start")))


def _subtitle_texts(row: dict[str, Any]) -> list[str]:
    subtitles = row.get("movie_subtitles") or []
    if not isinstance(subtitles, list):
        value = str(subtitles).strip()
        return [value] if value else []

    texts: list[str] = []
    for sub in subtitles:
        if isinstance(sub, dict):
            text = str(sub.get("text") or "").strip()
        else:
            text = str(sub).strip()
        if text:
            texts.append(text)
    return texts


def _payload_segment(segment: dict[str, Any]) -> dict[str, Any]:
    timing_signals_by_index = {
        int(signal.get("clip_index")): {
            "coverage": signal.get("coverage"),
            "subtitle_count": signal.get("subtitle_count"),
            "subtitle_overlap": signal.get("subtitle_overlap"),
            "timing_score": signal.get("score"),
            "timing_hint": signal.get("audio_role"),
        }
        for signal in segment.get("_timing_signals") or []
        if isinstance(signal, dict)
    }

    clips = []
    for fallback, row in enumerate(segment.get("movie_time_ranges") or []):
        idx = _clip_index(row, fallback)
        clips.append(
            {
                "clip_index": idx,
                "start": row.get("start"),
                "end": row.get("end"),
                "duration": round(_duration(row), 3),
                "movie_subtitles": _subtitle_texts(row),
                "timing_signals": timing_signals_by_index.get(idx, {}),
            }
        )
    clips.sort(key=lambda x: int(x.get("clip_index") or 0))
    return {
        "segment_id": segment.get("segment_id"),
        "text_role": segment.get("text_role") or "narration",
        "old_text": segment.get("old_text") or segment.get("text") or "",
        "segment_duration": round(sum(float(clip.get("duration") or 0.0) for clip in clips), 3),
        "clips": clips,
    }


def _build_payload(segments: list[dict[str, Any]]) -> dict[str, Any]:
    total_duration = sum(_duration(row) for segment in segments for row in segment.get("movie_time_ranges") or [] if isinstance(row, dict))
    total_clips = sum(1 for segment in segments for row in segment.get("movie_time_ranges") or [] if isinstance(row, dict))
    return {
        "task": "select_original_audio_under_budget",
        "global_budget": {
            "target_original_ratio": ORIGINAL_AUDIO_BUDGET_RATIO,
            "max_original_duration": round(total_duration * ORIGINAL_AUDIO_BUDGET_RATIO, 3),
            "max_original_clips": int(total_clips * ORIGINAL_AUDIO_BUDGET_RATIO),
            "min_original_block_seconds": MIN_ORIGINAL_BLOCK_SECONDS,
            "total_candidate_duration": round(total_duration, 3),
            "total_candidate_clips": total_clips,
        },
        "selection_guidance": [
            "只选择最有戏剧价值、情绪价值、信息价值的原声片段",
            "无价值寒暄、铺垫、解释性对白、过短对白全部用 AI 解说",
            "每个原声块必须至少 3 秒，且全片原声总量要控制在 30% 左右或更少",
            "priority 用于预算冲突时排序，最高价值片段给最高分",
        ],
        "allowed_audio_patterns": sorted(ALLOWED_AUDIO_PATTERNS),
        "segments": [_payload_segment(segment) for segment in segments],
        "output_schema": {
            "results": [
                {
                    "segment_id": "seg_001",
                    "audio_pattern": "narration_then_original_audio",
                    "split_clip_index": 3,
                    "priority": 92,
                }
            ]
        },
    }


def _fallback(source: str = "fallback") -> dict[str, Any]:
    return {"audio_pattern": "all_narration", "split_clip_index": None, "priority": 0, "source": source}


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


def _priority(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except Exception:
        return 50.0


def _validate_decision(segment: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return _fallback()
    pattern = str(row.get("audio_pattern") or "").strip()
    if pattern not in ALLOWED_AUDIO_PATTERNS:
        return _fallback()
    priority = _priority(row.get("priority"))
    if pattern == "all_narration":
        priority = 0.0
    if pattern in {"all_narration", "all_original_audio"}:
        return {"audio_pattern": pattern, "split_clip_index": None, "priority": priority, "source": "llm"}
    split = _valid_split(segment, row.get("split_clip_index"))
    if split is None:
        return _fallback()
    return {"audio_pattern": pattern, "split_clip_index": split, "priority": priority, "source": "llm"}


async def classify_segments(
    segments: list[dict[str, Any]],
    *,
    provider: CustomOpenAIProvider,
    batch_size: int = 8,
) -> dict[str, dict[str, Any]]:
    del batch_size
    results: dict[str, dict[str, Any]] = {}
    payload = _build_payload(segments)
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

    for segment in segments:
        seg_id = str(segment.get("segment_id") or "")
        results[seg_id] = _validate_decision(segment, by_segment.get(seg_id))
    return results
