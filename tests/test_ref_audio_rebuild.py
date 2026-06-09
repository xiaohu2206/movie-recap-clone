from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


rebuild = _load_module("ref_audio_rebuild_test", ROOT / "4.1_ref_audio_rebuild_composer" / "run.py")
generate = _load_module("generate_video_ref_audio_test", ROOT / "8_generate_video" / "run.py")


class RefAudioRebuildTests(unittest.TestCase):
    def test_compose_outputs_movie_fallback_merge_and_extension_items(self) -> None:
        ref_analysis = {
            "ref_video_id": "ref_demo",
            "ref_shots": [
                {"ref_shot_id": "ref_shot_001", "start": 0.0, "end": 2.0},
                {"ref_shot_id": "ref_shot_002", "start": 2.1, "end": 3.1},
                {"ref_shot_id": "ref_shot_003", "start": 5.0, "end": 6.0},
                {"ref_shot_id": "ref_shot_004", "start": 7.0, "end": 9.0},
            ],
        }
        timeline_data = {
            "ref_to_movie_timeline": [
                {
                    "ref_shot_id": "ref_shot_001",
                    "ref_start": 0.0,
                    "ref_end": 2.0,
                    "movie_start": 10.0,
                    "movie_end": 13.0,
                    "movie_shot_ids": ["movie_shot_010"],
                    "status": "matched",
                    "confidence": "high",
                    "match_score": 0.9,
                },
                {
                    "ref_shot_id": "ref_shot_002",
                    "ref_start": 2.1,
                    "ref_end": 3.1,
                    "movie_start": 12.1,
                    "movie_end": 13.1,
                    "movie_shot_ids": ["movie_shot_011"],
                    "status": "matched",
                    "confidence": "high",
                    "match_score": 0.88,
                },
                {
                    "ref_shot_id": "ref_shot_003",
                    "ref_start": 5.0,
                    "ref_end": 6.0,
                    "status": "needs_review",
                    "confidence": "low",
                    "match_score": 0.2,
                },
                {
                    "ref_shot_id": "ref_shot_004",
                    "ref_start": 7.0,
                    "ref_end": 9.0,
                    "movie_start": 20.0,
                    "movie_end": 21.0,
                    "movie_shot_ids": ["movie_shot_020"],
                    "status": "matched_low_confidence",
                    "confidence": "low",
                    "match_score": 0.5,
                },
            ]
        }

        result = rebuild.compose_ref_audio_rebuild_timeline(
            ref_analysis,
            {"movie_shots": []},
            timeline_data,
            ref_video_path="data/ref.mp4",
            movie_path="data/movie.mp4",
        )

        items = result["final_timeline"]
        self.assertEqual(result["mode"], "ref_audio_rebuild")
        self.assertEqual(len(items), 3)

        merged = items[0]
        self.assertEqual(merged["status"], "ready")
        self.assertEqual(merged["external_audio"]["start"], 0.0)
        self.assertEqual(merged["external_audio"]["end"], 3.1)
        self.assertEqual(merged["source_ref_shot_ids"], ["ref_shot_001", "ref_shot_002"])
        self.assertEqual(len(merged["video_clips"]), 2)
        self.assertEqual(merged["video_clips"][0]["source"], "data/movie.mp4")
        self.assertEqual(merged["video_clips"][0]["movie_end"], 12.0)
        self.assertEqual(merged["video_clips"][0]["fit_mode"], "cut_to_ref_duration")

        fallback = items[1]
        self.assertEqual(fallback["status"], "fallback")
        self.assertEqual(fallback["video_clips"][0]["source"], "data/ref.mp4")
        self.assertEqual(fallback["video_clips"][0]["movie_start"], 5.0)
        self.assertEqual(fallback["video_clips"][0]["fit_mode"], "fallback_use_ref_video")

        extended = items[2]
        self.assertEqual(extended["video_clips"][0]["movie_end"], 22.0)
        self.assertIn("movie_clip_short", extended["duration_warnings"])
        self.assertEqual(result["quality_report"]["fallback_ref_shots"], 1)

    def test_generate_video_cuts_reference_audio_without_tts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ref.mp4"
            source.write_bytes(b"placeholder")
            audio_dir = root / "audio"
            calls: list[list[str]] = []

            def fake_run_ffmpeg(args: list[str], *, check: bool = True):
                calls.append(args)
                Path(args[-1]).write_bytes(b"mp3")
                return None

            item = {
                "item_id": "rebuild_item_001",
                "audio_type": "reference_audio",
                "external_audio": {
                    "path": str(source),
                    "start": 0.5,
                    "end": 1.7,
                    "duration": 1.2,
                },
                "video_clips": [{"source": str(source), "movie_start": 10.0, "movie_end": 11.2}],
            }

            with patch.object(generate, "run_ffmpeg", side_effect=fake_run_ffmpeg), patch.object(
                generate,
                "probe_duration",
                return_value=1.2,
            ), patch.object(generate.edge_tts_service, "synthesize", new=AsyncMock(side_effect=AssertionError("tts called"))):
                result = asyncio.run(
                    generate._synthesize_one(
                        item,
                        audio_dir,
                        voice_id="unused",
                        speed_ratio=1.0,
                        proxy=None,
                        reuse=False,
                    )
                )

            self.assertEqual(result["source"], "reference_audio")
            self.assertEqual(result["duration"], 1.2)
            self.assertTrue(Path(result["path"]).exists())
            self.assertIn("-ss", calls[0])
            self.assertIn("0.500", calls[0])
            self.assertIn("-t", calls[0])
            self.assertIn("1.200", calls[0])


if __name__ == "__main__":
    unittest.main()
