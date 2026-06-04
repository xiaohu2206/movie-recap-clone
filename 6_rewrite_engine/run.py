from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.ai import AIModelConfig, ChatMessage, CustomOpenAIProvider
from clone_narration_video.utils.json_io import read_json, write_json
from clone_narration_video.utils.project_paths import default_output_dir


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


def _build_batch_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    rewrite_input = []
    for item in items:
        ranges = item.get("movie_time_ranges") or []
        rewrite_input.append(
            {
                "segment_id": item.get("segment_id"),
                "old_text": item.get("old_text") or item.get("text") or "",
                "text_role": item.get("text_role") or "narration",
                "ref_time_range": item.get("ref_time_range") or {},
                "movie_time_ranges": [
                    {
                        "start": r.get("start"),
                        "end": r.get("end"),
                        "confidence": r.get("confidence"),
                    }
                    for r in ranges
                ],
            }
        )
    return {
        "task": "rewrite_narration_segments",
        "input": rewrite_input,
        "output_schema": {
            "rewritten_script": [
                {
                    "segment_id": "seg_001",
                    "new_text": "新的中文解说文案",
                }
            ]
        },
    }


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
    except Exception:
        response = await provider.chat_completion(messages)

    parsed = _extract_json(response.content)
    rows = parsed.get("rewritten_script") or []
    if not isinstance(rows, list):
        raise ValueError("AI JSON 缺少 rewritten_script 数组")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        seg_id = str(row.get("segment_id") or "")
        new_text = str(row.get("new_text") or "").strip()
        if seg_id and new_text:
            result[seg_id] = new_text
    return result


async def rewrite_script(
    script_mapping: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    batch_size: int,
) -> dict[str, Any]:
    if not api_key:
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
        for start in range(0, len(script_mapping), max(1, batch_size)):
            batch = script_mapping[start : start + max(1, batch_size)]
            ai_rows = await _call_ai_batch(provider, batch)

            for item in batch:
                seg_id = str(item.get("segment_id") or "")
                old_text = str(item.get("old_text") or "")
                role = str(item.get("text_role") or "narration")
                new_text = ai_rows.get(seg_id)
                if not new_text:
                    raise ValueError(f"AI 返回缺少 segment_id={seg_id} 的 new_text")
                rewritten.append(
                    {
                        "segment_id": seg_id,
                        "old_text": old_text,
                        "new_text": new_text,
                        "old_char_count": _char_len(old_text),
                        "new_char_count": _char_len(new_text),
                        "text_role": role,
                        "ref_time_range": item.get("ref_time_range") or {},
                        "movie_time_ranges": item.get("movie_time_ranges") or [],
                        "rewrite_status": "ai_rewritten",
                    }
                )
    finally:
        await provider.close()

    return {
        "rewritten_script": rewritten,
        "rewrite_backend": {
            "provider": "custom_openai",
            "model": model,
            "batch_size": batch_size,
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

    data = read_json(args.script_mapping)
    script_mapping = data.get("script_mapping") or []
    if not isinstance(script_mapping, list):
        raise SystemExit("script_mapping.json 缺少 script_mapping 数组")
    if not args.api_key:
        raise SystemExit("缺少 AI API Key：请传 --api-key 或设置 CLONE_AI_API_KEY / OPENAI_API_KEY")

    result = asyncio.run(
        rewrite_script(
            script_mapping,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )
    )
    out = write_json(Path(args.output_dir) / "rewritten_script.json", result)
    print(out)


if __name__ == "__main__":
    main()
