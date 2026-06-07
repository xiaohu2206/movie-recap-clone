from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rewrite_engine = _load_module("rewrite_engine_audio_units_test", ROOT / "6_rewrite_engine" / "run.py")


class RewriteEngineAudioUnitsTests(unittest.TestCase):
    def test_pass_through_keeps_original_dialogue_units_empty(self) -> None:
        result = rewrite_engine.pass_through_script(
            [
                {
                    "segment_id": "seg_001",
                    "old_text": "旁白一句原片对白",
                    "segment_audio_role": "mixed",
                    "text_units": [
                        {
                            "unit_id": "seg_001_unit_001",
                            "text": "旁白一句",
                            "role": "narration",
                            "action": "rewrite",
                            "related_range_ids": ["seg_001_range_001"],
                        },
                        {
                            "unit_id": "seg_001_unit_002",
                            "text": "原片对白",
                            "role": "original_dialogue",
                            "action": "play_original_audio",
                            "related_range_ids": ["seg_001_range_002"],
                        },
                    ],
                    "movie_time_ranges": [
                        {"range_id": "seg_001_range_001", "audio_action": "rewrite_and_voiceover"},
                        {"range_id": "seg_001_range_002", "audio_action": "play_original_audio"},
                    ],
                }
            ]
        )

        row = result["rewritten_script"][0]
        self.assertEqual(row["segment_audio_role"], "mixed")
        self.assertEqual(row["new_text"], "旁白一句")
        self.assertEqual(row["rewritten_units"][0]["new_text"], "旁白一句")
        self.assertFalse(row["rewritten_units"][0]["keep_original_audio"])
        self.assertEqual(row["rewritten_units"][1]["new_text"], "")
        self.assertTrue(row["rewritten_units"][1]["keep_original_audio"])

    def test_ai_payload_only_sends_narration_units(self) -> None:
        captured: dict[str, object] = {}

        class FakeProvider:
            async def chat_completion(self, messages, extra_params=None):
                captured["payload"] = json.loads(messages[-1].content)
                return SimpleNamespace(content=json.dumps({"rewritten_units": [{"unit_id": "seg_001_unit_001", "new_text": "新的旁白"}]}))

        result = asyncio.run(
            rewrite_engine._call_ai_batch(
                FakeProvider(),
                [
                    {
                        "segment_id": "seg_001",
                        "old_text": "旁白一句原片对白",
                        "text_units": [
                            {
                                "unit_id": "seg_001_unit_001",
                                "text": "旁白一句",
                                "role": "narration",
                                "action": "rewrite",
                            },
                            {
                                "unit_id": "seg_001_unit_002",
                                "text": "原片对白",
                                "role": "original_dialogue",
                                "action": "play_original_audio",
                            },
                        ],
                    }
                ],
            )
        )

        units = captured["payload"]["input"][0]["text_units"]  # type: ignore[index]
        self.assertEqual(result, {"seg_001_unit_001": "新的旁白"})
        self.assertEqual([unit["unit_id"] for unit in units], ["seg_001_unit_001"])
        self.assertEqual(units[0]["old_text"], "旁白一句")


if __name__ == "__main__":
    unittest.main()
