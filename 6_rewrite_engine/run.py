from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.ai import AIModelConfig, ChatMessage, CustomOpenAIProvider
from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.progress import emit_progress
from clone_narration_video.utils.project_paths import default_output_dir


USE_LLM_REWRITE = False  # True=大模型生成仿稿；False=直接使用原稿（默认）


SYSTEM_PROMPT = """你是影视解说仿稿助手。
目标：基于旧文案结构，生成结构相似但表达不同的新文案。
约束：
1. 不改变剧情事实，不编造画面里没有的信息。
2. 保留叙事功能：钩子仍是钩子，转折仍是转折，收束仍是收束。
3. 新文案长度尽量接近旧文案，建议控制在旧文案字数的 80% 到 120%。
4. 不能逐字洗稿，句式、措辞、节奏都要有明显变化。
5. 必须只输出 JSON，不要解释。"""


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


def _char_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _text_units(item: dict[str, Any]) -> list[dict[str, Any]]:
    rows = item.get("text_units") or []
    return rows if isinstance(rows, list) else []


def _unit_id(unit: dict[str, Any], index: int) -> str:
    return str(unit.get("unit_id") or f"unit_{index:03d}")


def _unit_text(unit: dict[str, Any]) -> str:
    return str(unit.get("text") or unit.get("old_text") or "")


def _is_narration_unit(unit: dict[str, Any]) -> bool:
    role = str(unit.get("role") or "narration")
    action = str(unit.get("action") or "rewrite")
    return role == "narration" and action not in {"play_original_audio", "manual_review"}


def _is_original_audio_unit(unit: dict[str, Any]) -> bool:
    role = str(unit.get("role") or "")
    action = str(unit.get("action") or "")
    return role == "original_dialogue" or action == "play_original_audio"


def _join_text(parts: list[str]) -> str:
    return "".join(x.strip() for x in parts if x and x.strip())


def _build_rewritten_units(
    item: dict[str, Any],
    *,
    unit_rewrites: dict[str, str],
    passthrough_narration: bool,
) -> list[dict[str, Any]]:
    rewritten_units: list[dict[str, Any]] = []
    for idx, unit in enumerate(_text_units(item), start=1):
        uid = _unit_id(unit, idx)
        old_text = _unit_text(unit)
        keep_original = _is_original_audio_unit(unit)
        if keep_original:
            new_text = ""
        elif _is_narration_unit(unit):
            new_text = old_text if passthrough_narration else unit_rewrites.get(uid, "")
        else:
            new_text = ""
        row = {
            "unit_id": uid,
            "role": str(unit.get("role") or "narration"),
            "action": str(unit.get("action") or ("play_original_audio" if keep_original else "rewrite")),
            "old_text": old_text,
            "new_text": new_text,
            "keep_original_audio": keep_original,
            "related_range_ids": unit.get("related_range_ids") or [],
        }
        if unit.get("matched_movie_subtitles"):
            row["matched_movie_subtitles"] = unit.get("matched_movie_subtitles") or []
        rewritten_units.append(row)
    return rewritten_units


def _segment_new_text_from_units(rewritten_units: list[dict[str, Any]]) -> str:
    return _join_text([str(unit.get("new_text") or "") for unit in rewritten_units if not unit.get("keep_original_audio")])


def _narration_old_text(item: dict[str, Any]) -> str:
    narration_part = item.get("narration_part") or {}
    if isinstance(narration_part, dict):
        text = str(narration_part.get("old_text") or "").strip()
        if text:
            return text
    return str(item.get("old_text") or item.get("text") or "")


def _audio_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "audio_pattern": item.get("audio_pattern") or "all_narration",
        "split_clip_index": item.get("split_clip_index"),
        "narration_part": item.get("narration_part") or {},
        "original_audio_part": item.get("original_audio_part") or {},
        "movie_time_ranges": item.get("movie_time_ranges") or [],
    }


def _build_batch_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    rewrite_input = []
    has_units = any(_text_units(item) for item in items)
    for item in items:
        ranges = item.get("movie_time_ranges") or []
        payload_item = {
            "segment_id": item.get("segment_id"),
            "old_text": _narration_old_text(item),
            "text_role": item.get("text_role") or "narration",
            "segment_audio_role": item.get("segment_audio_role"),
            "ref_time_range": item.get("ref_time_range") or {},
            "movie_time_ranges": [
                {
                    "start": r.get("start"),
                    "end": r.get("end"),
                    "confidence": r.get("confidence"),
                    "audio_action": r.get("audio_action"),
                }
                for r in ranges
            ],
        }
        units = _text_units(item)
        if units:
            payload_item["text_units"] = [
                {
                    "unit_id": _unit_id(unit, idx),
                    "old_text": _unit_text(unit),
                    "role": unit.get("role") or "narration",
                    "action": unit.get("action") or "rewrite",
                }
                for idx, unit in enumerate(units, start=1)
                if _is_narration_unit(unit)
            ]
        rewrite_input.append(payload_item)
    output_schema: dict[str, Any] = {
        "rewritten_script": [
            {
                "segment_id": "seg_001",
                "new_text": "新的中文解说文案",
            }
        ]
    }
    if has_units:
        output_schema["rewritten_units"] = [
            {
                "unit_id": "seg_001_unit_001",
                "new_text": "新的中文解说文案",
            }
        ]
    return {
        "task": "rewrite_narration_segments",
        "input": rewrite_input,
        "output_schema": output_schema,
    }


def _is_connectivity_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    return "connect" in name or "timeout" in name


async def _call_ai_batch(
    provider: CustomOpenAIProvider,
    batch: list[dict[str, Any]],
) -> dict[str, str]:
    payload = _build_batch_payload(batch)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
    ]
    try:
        response = await provider.chat_completion(
            messages,
            extra_params={"response_format": {"type": "json_object"}},
        )
    except Exception as exc:
        if _is_connectivity_error(exc):
            raise
        response = await provider.chat_completion(messages)

    result: dict[str, str] = {}
    parsed = _extract_json(response.content)
    unit_rows = parsed.get("rewritten_units") or []
    if isinstance(unit_rows, list):
        for row in unit_rows:
            if not isinstance(row, dict):
                continue
            unit_id = str(row.get("unit_id") or "")
            new_text = str(row.get("new_text") or "").strip()
            if unit_id and new_text:
                result[unit_id] = new_text
    rows = parsed.get("rewritten_script") or []
    if not isinstance(rows, list):
        raise ValueError("AI JSON 缺少 rewritten_script 数组")
    for row in rows:
        if not isinstance(row, dict):
            continue
        seg_id = str(row.get("segment_id") or "")
        new_text = str(row.get("new_text") or "").strip()
        if seg_id and new_text:
            result[seg_id] = new_text
        for unit in row.get("rewritten_units") or []:
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("unit_id") or "")
            unit_text = str(unit.get("new_text") or "").strip()
            if unit_id and unit_text:
                result[unit_id] = unit_text
    return result


def pass_through_script(
    script_mapping: list[dict[str, Any]],
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    rewritten: list[dict[str, Any]] = []
    total = len(script_mapping)
    for idx, item in enumerate(script_mapping, start=1):
        seg_id = str(item.get("segment_id") or "")
        audio_pattern = str(item.get("audio_pattern") or "all_narration")
        old_text = _narration_old_text(item)
        rewritten_units = _build_rewritten_units(item, unit_rewrites={}, passthrough_narration=True)
        if audio_pattern == "all_original_audio":
            new_text = ""
        elif rewritten_units:
            new_text = _segment_new_text_from_units(rewritten_units)
        else:
            new_text = old_text
        role = str(item.get("text_role") or "narration")
        rewritten.append(
            {
                "segment_id": seg_id,
                "old_text": old_text,
                "new_text": new_text,
                "old_char_count": _char_len(old_text),
                "new_char_count": _char_len(new_text),
                "text_role": role,
                "segment_audio_role": item.get("segment_audio_role"),
                "rewritten_units": rewritten_units,
                "ref_time_range": item.get("ref_time_range") or {},
                **_audio_fields(item),
                "rewrite_status": "original_audio_skip" if audio_pattern == "all_original_audio" else "original",
            }
        )
        if progress_callback:
            progress_callback(
                (idx / max(1, total)) * 100.0,
                f"Pass-through segments {idx}/{total}",
            )

    return {
        "rewritten_script": rewritten,
        "rewrite_backend": {
            "provider": "original",
            "use_ai_rewrite": False,
            "use_llm_rewrite": False,
        },
    }


def pass_through_original_script(
    script_mapping: list[dict[str, Any]],
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    return pass_through_script(script_mapping, progress_callback=progress_callback)


async def rewrite_script(
    script_mapping: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    batch_size: int,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    api_key = (api_key or "").strip()
    base_url = (base_url or "").strip()
    model = (model or "").strip()
    llm_items = [item for item in script_mapping if str(item.get("audio_pattern") or "all_narration") != "all_original_audio"]
    if llm_items and not api_key:
        raise ValueError("缺少 AI API Key")

    cfg = AIModelConfig(
        provider="custom_openai",
        api_key=api_key,
        base_url=base_url,
        model_name=model,
        temperature=temperature,
    )
    provider = CustomOpenAIProvider(cfg)

    rewritten: list[dict[str, Any]] = []

    try:
        skipped = {
            str(item.get("segment_id") or ""): item
            for item in script_mapping
            if str(item.get("audio_pattern") or "all_narration") == "all_original_audio"
        }
        rewritten_by_id: dict[str, dict[str, Any]] = {}

        for seg_id, item in skipped.items():
            old_text = _narration_old_text(item)
            rewritten_by_id[seg_id] = {
                "segment_id": seg_id,
                "old_text": old_text,
                "new_text": "",
                "old_char_count": _char_len(old_text),
                "new_char_count": 0,
                "text_role": str(item.get("text_role") or "narration"),
                "segment_audio_role": item.get("segment_audio_role"),
                "rewritten_units": _build_rewritten_units(item, unit_rewrites={}, passthrough_narration=False),
                "ref_time_range": item.get("ref_time_range") or {},
                **_audio_fields(item),
                "rewrite_status": "original_audio_skip",
            }

        total = len(llm_items)
        for start in range(0, len(llm_items), max(1, batch_size)):
            batch = llm_items[start : start + max(1, batch_size)]
            if progress_callback:
                progress_callback((start / max(1, total)) * 100.0, f"Rewriting batch {start + 1}-{start + len(batch)} of {total}")
            ai_rows = await _call_ai_batch(provider, batch)

            for item in batch:
                seg_id = str(item.get("segment_id") or "")
                old_text = _narration_old_text(item)
                role = str(item.get("text_role") or "narration")
                units = _text_units(item)
                if units:
                    missing_units = [
                        _unit_id(unit, idx)
                        for idx, unit in enumerate(units, start=1)
                        if _is_narration_unit(unit) and not ai_rows.get(_unit_id(unit, idx))
                    ]
                    if missing_units:
                        raise ValueError(f"AI 返回缺少 narration unit 的 new_text: {', '.join(missing_units)}")
                    rewritten_units = _build_rewritten_units(item, unit_rewrites=ai_rows, passthrough_narration=False)
                    new_text = _segment_new_text_from_units(rewritten_units)
                else:
                    new_text = ai_rows.get(seg_id)
                    if not new_text:
                        raise ValueError(f"AI 返回缺少 segment_id={seg_id} 的 new_text")
                    rewritten_units = []
                rewritten_by_id[seg_id] = {
                    "segment_id": seg_id,
                    "old_text": old_text,
                    "new_text": new_text,
                    "old_char_count": _char_len(old_text),
                    "new_char_count": _char_len(new_text),
                    "text_role": role,
                    "segment_audio_role": item.get("segment_audio_role"),
                    "rewritten_units": rewritten_units,
                    "ref_time_range": item.get("ref_time_range") or {},
                    **_audio_fields(item),
                    "rewrite_status": "ai_rewritten",
                }
            if progress_callback:
                done = min(total, start + len(batch))
                progress_callback((done / max(1, total)) * 100.0, f"Rewritten segments {done}/{total}")

        for item in script_mapping:
            seg_id = str(item.get("segment_id") or "")
            if seg_id in rewritten_by_id:
                rewritten.append(rewritten_by_id[seg_id])
    finally:
        await provider.close()

    return {
        "rewritten_script": rewritten,
        "rewrite_backend": {
            "provider": "custom_openai",
            "model": model,
            "batch_size": batch_size,
            "use_ai_rewrite": True,
            "use_llm_rewrite": True,
        },
    }


def _load_dotenv() -> None:
    """从 clone_narration_video/.env 读取配置并注入 os.environ（不覆盖已有变量）。"""
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
    parser = argparse.ArgumentParser(description="仿稿模块")
    parser.add_argument("--script-mapping", required=True, help="第 5 步输出的 script_mapping.json")
    parser.add_argument("--output-dir", default=str(default_output_dir("6_rewrite_engine")))
    parser.add_argument("--provider", choices=["custom_openai"], default=os.getenv("CLONE_AI_PROVIDER", "custom_openai"))
    parser.add_argument("--api-key", default=os.getenv("CLONE_AI_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    parser.add_argument("--base-url", default=os.getenv("CLONE_AI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    parser.add_argument("--model", default=os.getenv("CLONE_AI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("CLONE_AI_TEMPERATURE", "0.7")))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.api_key = (args.api_key or "").strip()
    args.base_url = (args.base_url or "").strip()
    args.model = (args.model or "").strip()

    data = read_json(args.script_mapping)
    script_mapping = data.get("script_mapping") or []
    if not isinstance(script_mapping, list):
        raise SystemExit("script_mapping.json 缺少 script_mapping 数组")

    progress_callback = lambda percent, message: emit_progress("rewrite", percent, message)
    if USE_LLM_REWRITE:
        if not args.api_key:
            raise SystemExit("缺少 AI API Key：请传 --api-key 或设置 CLONE_AI_API_KEY / OPENAI_API_KEY")
        try:
            result = asyncio.run(
                rewrite_script(
                    script_mapping,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    temperature=args.temperature,
                    batch_size=args.batch_size,
                    progress_callback=progress_callback,
                )
            )
        except ConnectionError as exc:
            raise SystemExit(str(exc)) from None
    else:
        result = pass_through_script(script_mapping, progress_callback=progress_callback)
    out = write_json(Path(args.output_dir) / "rewritten_script.json", result)
    print(out)


if __name__ == "__main__":
    main()
