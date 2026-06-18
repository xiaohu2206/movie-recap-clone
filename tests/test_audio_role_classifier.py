from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from copy import deepcopy
from pathlib import Path

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


audio_role = _load_module("audio_role_classifier_test", ROOT / "5.2_audio_role_classifier" / "run.py")


def _subtitle(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


def _clip(index: int, start: float, end: float, subtitles: list[dict] | None = None) -> dict:
    return {
        "clip_index": index,
        "start": start,
        "end": end,
        "movie_subtitles": subtitles or [],
    }


def _classify(segment: dict) -> dict:
    return asyncio.run(
        audio_role.classify_audio_roles(
            [segment],
            api_key="",
            base_url="",
            model="",
            temperature=0.0,
            batch_size=8,
        )
    )["script_mapping"][0]


class AudioRoleClassifierTests(unittest.TestCase):
    def test_timing_signals_ignore_subtitle_text_language(self) -> None:
        segment = {
            "segment_id": "seg_001",
            "old_text": "narration",
            "text_role": "narration",
            "movie_time_ranges": [
                _clip(0, 0.0, 2.0, [_subtitle(0.1, 1.9, "hello there")]),
                _clip(1, 2.0, 4.0, [_subtitle(2.1, 3.9, "how are you")]),
            ],
        }
        translated = deepcopy(segment)
        translated["movie_time_ranges"][0]["movie_subtitles"][0]["text"] = "你好"
        translated["movie_time_ranges"][1]["movie_subtitles"][0]["text"] = "最近怎么样"

        english_signals = audio_role._timing_signals_for_segment(segment)
        translated_signals = audio_role._timing_signals_for_segment(translated)

        self.assertEqual(english_signals, translated_signals)

    def test_budget_rejects_original_blocks_shorter_than_three_seconds(self) -> None:
        segment = {
            "segment_id": "seg_short",
            "old_text": "short original",
            "movie_time_ranges": [
                _clip(0, 0.0, 2.0, [_subtitle(0.1, 1.9, "short")]),
                _clip(1, 2.0, 10.0),
            ],
        }
        decisions = {
            "seg_short": {
                "audio_pattern": "original_audio_then_narration",
                "split_clip_index": 1,
                "priority": 99,
                "source": "llm",
            }
        }

        budgeted, summary = audio_role._apply_budget_constraints([segment], decisions)

        self.assertEqual(budgeted["seg_short"]["audio_pattern"], "all_narration")
        self.assertEqual(budgeted["seg_short"]["source"], "budget_min_original_block")
        self.assertEqual(summary["rejected_short"], 1)

    def test_budget_keeps_high_priority_original_under_thirty_percent(self) -> None:
        segments = [
            {
                "segment_id": "seg_low",
                "old_text": "low priority",
                "movie_time_ranges": [_clip(0, 0.0, 4.0, [_subtitle(0.0, 4.0, "low")])],
            },
            {
                "segment_id": "seg_high",
                "old_text": "high priority",
                "movie_time_ranges": [_clip(0, 10.0, 14.0, [_subtitle(10.0, 14.0, "high")])],
            },
            {
                "segment_id": "seg_pad",
                "old_text": "padding narration",
                "movie_time_ranges": [_clip(0, 20.0, 32.0)],
            },
        ]
        decisions = {
            "seg_low": {"audio_pattern": "all_original_audio", "split_clip_index": None, "priority": 40, "source": "llm"},
            "seg_high": {"audio_pattern": "all_original_audio", "split_clip_index": None, "priority": 95, "source": "llm"},
            "seg_pad": {"audio_pattern": "all_narration", "split_clip_index": None, "priority": 0, "source": "llm"},
        }

        budgeted, summary = audio_role._apply_budget_constraints(segments, decisions)

        self.assertEqual(summary["max_original_duration"], 6.0)
        self.assertEqual(summary["used_original_duration"], 4.0)
        self.assertEqual(budgeted["seg_high"]["audio_pattern"], "all_original_audio")
        self.assertEqual(budgeted["seg_low"]["audio_pattern"], "all_narration")
        self.assertEqual(budgeted["seg_low"]["source"], "budget_over_limit")

    def test_no_api_fallback_still_reports_budget_strategy(self) -> None:
        segment = {
            "segment_id": "seg_ending",
            "old_text": "ending narration",
            "text_role": "ending",
            "movie_time_ranges": [
                _clip(0, 0.0, 5.0, [_subtitle(0.1, 4.9, "closing")]),
                _clip(1, 5.0, 10.0, [_subtitle(5.1, 9.9, "closing")]),
            ],
        }

        result = _classify(segment)

        self.assertFalse(result["audio_decision"]["strategy"]["subtitle_text_used_by_code"])
        self.assertEqual(result["audio_decision"]["strategy"]["min_original_block_seconds"], 3.0)
        self.assertEqual(result["audio_decision"]["strategy"]["original_audio_budget_ratio"], 0.3)


if __name__ == "__main__":
    unittest.main()
