from __future__ import annotations

import importlib.util
import sys
import unittest
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


timeline = _load_module("timeline_audio_modes_test", ROOT / "7_timeline_composer" / "run.py")


class TimelineAudioModesTests(unittest.TestCase):
    def test_mixed_segment_splits_voiceover_and_original_items(self) -> None:
        result = timeline.compose_timeline(
            [
                {
                    "segment_id": "seg_001",
                    "old_text": "旁白一句原片对白",
                    "new_text": "新的旁白",
                    "rewrite_status": "ai_rewritten",
                    "ref_time_range": {"start": 0.0, "end": 4.0},
                    "segment_audio_role": "mixed",
                    "rewritten_units": [
                        {
                            "unit_id": "seg_001_unit_001",
                            "role": "narration",
                            "old_text": "旁白一句",
                            "new_text": "新的旁白",
                            "keep_original_audio": False,
                            "related_range_ids": ["seg_001_range_001"],
                        },
                        {
                            "unit_id": "seg_001_unit_002",
                            "role": "original_dialogue",
                            "old_text": "原片对白",
                            "new_text": "",
                            "keep_original_audio": True,
                            "related_range_ids": ["seg_001_range_002"],
                        },
                    ],
                    "movie_time_ranges": [
                        {
                            "range_id": "seg_001_range_001",
                            "start": 10.0,
                            "end": 14.0,
                            "source_ref_shot_id": "ref_shot_001",
                            "movie_shot_ids": ["movie_shot_001"],
                            "confidence": "high",
                            "audio_role": "narration_overlay",
                            "audio_action": "rewrite_and_voiceover",
                        },
                        {
                            "range_id": "seg_001_range_002",
                            "start": 20.0,
                            "end": 23.0,
                            "source_ref_shot_id": "ref_shot_002",
                            "movie_shot_ids": ["movie_shot_002"],
                            "confidence": "high",
                            "audio_role": "original_dialogue",
                            "audio_action": "play_original_audio",
                        },
                    ],
                }
            ],
            source="movie.mp4",
            movie_shots_data=None,
            chars_per_second=10.0,
            min_duration=1.0,
        )

        rows = result["final_timeline"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["audio_mode"], "voiceover")
        self.assertEqual(rows[0]["OST"], 0)
        self.assertEqual(rows[0]["narration"], "新的旁白")
        self.assertFalse(rows[0]["video_clips"][0]["keep_original_audio"])
        self.assertEqual(rows[0]["audio_decision"]["audio_action"], "rewrite_and_voiceover")

        self.assertEqual(rows[1]["audio_mode"], "original")
        self.assertEqual(rows[1]["OST"], 1)
        self.assertEqual(rows[1]["narration"], "")
        self.assertEqual(rows[1]["tts_duration"], 3.0)
        self.assertEqual(rows[1]["timeline_end"] - rows[1]["timeline_start"], 3.0)
        self.assertEqual(rows[1]["video_clips"][0]["movie_start"], 20.0)
        self.assertEqual(rows[1]["video_clips"][0]["movie_end"], 23.0)
        self.assertTrue(rows[1]["video_clips"][0]["keep_original_audio"])
        self.assertEqual(rows[1]["video_clips"][0]["allocation"], "original_audio")
        self.assertEqual(rows[1]["audio_decision"]["source_range_ids"], ["seg_001_range_002"])

    def test_legacy_segments_still_emit_single_voiceover_item(self) -> None:
        result = timeline.compose_timeline(
            [
                {
                    "segment_id": "seg_001",
                    "old_text": "旧旁白",
                    "new_text": "新旁白",
                    "rewrite_status": "ai_rewritten",
                    "movie_time_ranges": [
                        {
                            "start": 1.0,
                            "end": 4.0,
                            "source_ref_shot_id": "ref_shot_001",
                            "movie_shot_ids": ["movie_shot_001"],
                            "confidence": "high",
                        }
                    ],
                }
            ],
            source="movie.mp4",
            movie_shots_data=None,
            chars_per_second=10.0,
            min_duration=1.0,
        )

        rows = result["final_timeline"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["audio_mode"], "voiceover")
        self.assertEqual(rows[0]["OST"], 0)
        self.assertEqual(rows[0]["audio_decision"]["audio_action"], "rewrite_and_voiceover")
        self.assertFalse(rows[0]["video_clips"][0]["keep_original_audio"])

    def test_manual_review_unit_falls_back_to_old_text_for_voiceover(self) -> None:
        result = timeline.compose_timeline(
            [
                {
                    "segment_id": "seg_001",
                    "old_text": "我能啊原片对白",
                    "new_text": "原片对白",
                    "rewrite_status": "original",
                    "rewritten_units": [
                        {
                            "unit_id": "seg_001_unit_001",
                            "role": "unknown",
                            "action": "manual_review",
                            "old_text": "我能啊",
                            "new_text": "",
                            "keep_original_audio": False,
                            "related_range_ids": ["seg_001_range_001"],
                        },
                        {
                            "unit_id": "seg_001_unit_002",
                            "role": "original_dialogue",
                            "action": "play_original_audio",
                            "old_text": "原片对白",
                            "new_text": "",
                            "keep_original_audio": True,
                            "related_range_ids": ["seg_001_range_002"],
                        },
                    ],
                    "movie_time_ranges": [
                        {
                            "range_id": "seg_001_range_001",
                            "start": 10.0,
                            "end": 12.0,
                            "source_ref_shot_id": "ref_shot_001",
                            "movie_shot_ids": ["movie_shot_001"],
                            "confidence": "high",
                            "audio_role": "unknown",
                            "audio_action": "manual_review",
                        },
                        {
                            "range_id": "seg_001_range_002",
                            "start": 12.0,
                            "end": 14.0,
                            "source_ref_shot_id": "ref_shot_002",
                            "movie_shot_ids": ["movie_shot_002"],
                            "confidence": "high",
                            "audio_role": "original_dialogue",
                            "audio_action": "play_original_audio",
                        },
                    ],
                }
            ],
            source="movie.mp4",
            movie_shots_data=None,
            chars_per_second=10.0,
            min_duration=1.0,
        )

        rows = result["final_timeline"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["audio_mode"], "voiceover")
        self.assertEqual(rows[0]["narration"], "我能啊")
        self.assertGreater(len(rows[0]["video_clips"]), 0)
        self.assertEqual(rows[0]["audio_decision"]["audio_action"], "manual_review")
        self.assertEqual(rows[1]["audio_mode"], "original")


if __name__ == "__main__":
    unittest.main()
