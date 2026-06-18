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


timeline = _load_module("timeline_composer_test", ROOT / "7_timeline_composer" / "run.py")


class TimelineComposerTests(unittest.TestCase):
    def test_allocate_extends_last_clip_without_adjacent_movie_shots(self) -> None:
        item = {
            "movie_time_ranges": [
                {
                    "start": 10.0,
                    "end": 11.0,
                    "movie_shot_ids": ["movie_shot_001"],
                    "source_ref_shot_id": "ref_shot_001",
                }
            ]
        }
        clips, status = timeline.allocate_video_clips(
            item,
            tts_duration=3.0,
            source="movie.mp4",
            movie_shots=[
                {"start": 11.0, "end": 12.0, "movie_shot_id": "movie_shot_002"},
                {"start": 12.0, "end": 13.0, "movie_shot_id": "movie_shot_003"},
            ],
        )

        self.assertEqual(status, "extended_last_clip")
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["movie_shot_ids"], ["movie_shot_001"])
        self.assertEqual(clips[0]["movie_end"], 13.0)
        self.assertEqual(clips[0]["allocation"], "synthetic_extension")

    def test_compose_dedupes_rollbacks_and_short_original_play(self) -> None:
        result = timeline.compose_timeline(
            [
                {
                    "segment_id": "seg_001",
                    "new_text": "abc",
                    "audio_pattern": "all_narration",
                    "movie_time_ranges": [
                        {
                            "start": 10.0,
                            "end": 11.0,
                            "movie_shot_ids": ["movie_shot_001"],
                            "source_ref_shot_id": "ref_shot_001",
                        }
                    ],
                },
                {
                    "segment_id": "seg_002",
                    "old_text": "short original",
                    "audio_pattern": "all_original_audio",
                    "movie_time_ranges": [
                        {
                            "start": 10.5,
                            "end": 11.5,
                            "movie_shot_ids": ["movie_shot_001"],
                            "source_ref_shot_id": "ref_shot_002",
                            "movie_subtitles": [{"start": 10.5, "end": 11.5, "text": "duplicate"}],
                        },
                        {
                            "start": 12.0,
                            "end": 13.0,
                            "movie_shot_ids": ["movie_shot_002"],
                            "source_ref_shot_id": "ref_shot_003",
                            "movie_subtitles": [{"start": 12.0, "end": 13.0, "text": "short line"}],
                        },
                    ],
                },
                {
                    "segment_id": "seg_003",
                    "audio_pattern": "all_original_audio",
                    "movie_time_ranges": [
                        {
                            "start": 15.0,
                            "end": 16.0,
                            "movie_shot_ids": ["movie_shot_003"],
                            "source_ref_shot_id": "ref_shot_004",
                            "movie_subtitles": [{"start": 15.0, "end": 16.0, "text": "long"}],
                        },
                        {
                            "start": 16.1,
                            "end": 18.4,
                            "movie_shot_ids": ["movie_shot_004"],
                            "source_ref_shot_id": "ref_shot_005",
                            "movie_subtitles": [{"start": 16.1, "end": 18.4, "text": "original"}],
                        },
                    ],
                },
            ],
            source="movie.mp4",
            movie_shots_data={"movie_shots": []},
            chars_per_second=10.0,
            min_duration=1.0,
        )

        items = result["final_timeline"]
        clips = [clip for item in items for clip in item.get("video_clips") or []]
        shot_ids = [shot_id for clip in clips for shot_id in clip.get("movie_shot_ids") or []]

        self.assertEqual(shot_ids.count("movie_shot_001"), 1)
        self.assertNotIn("extended_with_adjacent_shots", [item.get("allocation_status") for item in items])
        self.assertTrue(
            any(
                item["audio_type"] == "original_audio"
                and item["segment_id"] == "seg_002"
                and item.get("video_clips")
                for item in items
            )
        )
        self.assertFalse(
            any(
                item["audio_type"] == "narration"
                and item["segment_id"] == "seg_002"
                for item in items
            )
        )

        movie_starts = [float(clip["movie_start"]) for clip in clips]
        movie_ends = [float(clip["movie_end"]) for clip in clips]
        for previous_end, next_start in zip(movie_ends, movie_starts[1:]):
            self.assertGreaterEqual(next_start + timeline.MOVIE_ROLLBACK_TOLERANCE, previous_end)

    def test_compose_keeps_distinct_earlier_movie_matches(self) -> None:
        result = timeline.compose_timeline(
            [
                {
                    "segment_id": "seg_001",
                    "new_text": "first",
                    "audio_pattern": "all_narration",
                    "movie_time_ranges": [
                        {
                            "start": 100.0,
                            "end": 103.0,
                            "movie_shot_ids": ["movie_shot_late"],
                            "source_ref_shot_id": "ref_shot_001",
                        }
                    ],
                },
                {
                    "segment_id": "seg_002",
                    "new_text": "second",
                    "audio_pattern": "all_narration",
                    "movie_time_ranges": [
                        {
                            "start": 10.0,
                            "end": 13.0,
                            "movie_shot_ids": ["movie_shot_early"],
                            "source_ref_shot_id": "ref_shot_002",
                        }
                    ],
                },
            ],
            source="movie.mp4",
            movie_shots_data={"movie_shots": []},
            chars_per_second=10.0,
            min_duration=1.0,
        )

        items = result["final_timeline"]
        self.assertEqual(items[1]["allocation_status"], "trimmed_to_tts")
        self.assertEqual(items[1]["video_clips"][0]["movie_shot_ids"], ["movie_shot_early"])
        self.assertFalse(result["timeline_backend"]["prevent_movie_rollback"])


if __name__ == "__main__":
    unittest.main()
