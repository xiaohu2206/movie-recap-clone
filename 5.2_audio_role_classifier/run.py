from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

import classifier_llm
import rules
from clone_narration_video.utils.ai import AIModelConfig, CustomOpenAIProvider
from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.progress import emit_progress
from clone_narration_video.utils.project_paths import default_output_dir


def _clip_index(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("clip_index"))
    except Exception:
        return fallback


def _duration(row: dict[str, Any]) -> float:
    try:
        return max(0.0, float(row.get("end") or 0.0) - float(row.get("start") or 0.0))
    except Exception:
        return 0.0


def _text_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _slice_text_by_ratio(text: str, start_ratio: float, end_ratio: float) -> str:
    raw = text or ""
    if not raw:
        return ""
    length = len(raw)
    start = max(0, min(length, int(round(length * start_ratio))))
    end = max(start, min(length, int(round(length * end_ratio))))
    return raw[start:end].strip()


def _subtitle_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for sub in row.get("movie_subtitles") or []:
        text = str(sub.get("text") if isinstance(sub, dict) else sub or "").strip()
        if text:
            texts.append(text)
    return texts


def _groups_for_decision(segment: dict[str, Any], decision: dict[str, Any]) -> tuple[set[int], set[int]]:
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


def _narration_text(segment: dict[str, Any], narration_indexes: set[int]) -> str:
    old_text = str(segment.get("old_text") or segment.get("text") or "")
    ranges = segment.get("movie_time_ranges") or []
    total_duration = sum(_duration(row) for row in ranges)
    if not old_text or not ranges or not narration_indexes or len(narration_indexes) == len(ranges) or total_duration <= 0:
        return old_text if narration_indexes else ""

    ordered = sorted((_clip_index(row, idx), row) for idx, row in enumerate(ranges))
    cursor = 0.0
    parts: list[str] = []
    for idx, row in ordered:
        start_ratio = cursor / total_duration
        cursor += _duration(row)
        end_ratio = cursor / total_duration
        if idx in narration_indexes:
            parts.append(_slice_text_by_ratio(old_text, start_ratio, end_ratio))
    joined = "".join(parts).strip()
    return joined or old_text


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

    narration_text = _narration_text(item, narration_indexes)
    item["audio_pattern"] = decision.get("audio_pattern") or "all_narration"
    item["split_clip_index"] = decision.get("split_clip_index")
    item["narration_part"] = {
        "clip_indexes": sorted(narration_indexes),
        "old_text": narration_text,
    }
    item["original_audio_part"] = {
        "clip_indexes": sorted(original_indexes),
        "subtitles": original_subtitles,
    }
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
    decisions: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for segment in script_mapping:
        seg_id = str(segment.get("segment_id") or "")
        shortcut = rules.try_shortcut(segment)
        if shortcut:
            decisions[seg_id] = shortcut
        else:
            pending.append(segment)

    if progress_callback:
        progress_callback(15.0, f"Rule shortcut {len(decisions)}/{len(script_mapping)}")

    llm_calls = 0
    if pending:
        if not api_key:
            raise ValueError("存在需要 LLM 判定的片段，请传 --api-key 或设置 CLONE_AI_API_KEY / OPENAI_API_KEY")
        cfg = AIModelConfig(
            provider="custom_openai",
            api_key=api_key,
            base_url=base_url,
            model_name=model,
            temperature=temperature,
        )
        provider = CustomOpenAIProvider(cfg)
        try:
            llm_decisions = await classifier_llm.classify_segments(pending, provider=provider, batch_size=batch_size)
            decisions.update(llm_decisions)
            llm_calls = (len(pending) + max(1, batch_size) - 1) // max(1, batch_size)
        finally:
            await provider.close()

    if progress_callback:
        progress_callback(70.0, "Writing audio decisions")

    output_rows: list[dict[str, Any]] = []
    fallback = 0
    for idx, segment in enumerate(script_mapping, start=1):
        seg_id = str(segment.get("segment_id") or "")
        decision = decisions.get(seg_id) or {"audio_pattern": "all_narration", "split_clip_index": None, "source": "fallback"}
        if decision.get("source") == "fallback":
            fallback += 1
        output_rows.append(_apply_decision(segment, decision))
        if progress_callback and (idx == 1 or idx == len(script_mapping) or idx % 10 == 0):
            progress_callback(70.0 + (idx / max(1, len(script_mapping))) * 30.0, f"Classified audio roles {idx}/{len(script_mapping)}")

    return {
        "script_mapping": output_rows,
        "audio_backend": {
            "provider": "custom_openai" if pending else "rule",
            "model": model if pending else "",
            "rule_hit": len(script_mapping) - len(pending),
            "llm_segments": len(pending),
            "llm_calls": llm_calls,
            "fallback": fallback,
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
    script_mapping = data.get("script_mapping") or []
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
