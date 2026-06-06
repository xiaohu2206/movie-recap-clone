from __future__ import annotations

import importlib.util
import sys
import tempfile
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


audio_role = _load_module("audio_role_classifier_test", ROOT / "5_audio_role_classifier" / "run.py")


class AudioRoleClassifierTests(unittest.TestCase):
    def test_classifies_text_units_and_movie_subtitles_with_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_srt = root / "ref.srt"
            movie_srt = root / "movie.srt"
            ref_srt.write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:00,000 --> 00:00:02,000",
                        "职场如战场而他是那个最没用的炮灰",
                        "",
                        "2",
                        "00:00:02,200 --> 00:00:04,000",
                        "因为你业绩差又不努力",
                        "",
                        "3",
                        "00:00:05,000 --> 00:00:06,000",
                        "每天只有无尽的加班",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            movie_srt.write_text(
                "\n".join(
                    [
                        "1",
                        "00:01:40,100 --> 00:01:40,600",
                        "路人甲",
                        "",
                        "2",
                        "00:03:20,100 --> 00:03:21,900",
                        "因为你业绩差又不努力",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            script_mapping = {
                "script_mapping": [
                    {
                        "segment_id": "seg_001",
                        "old_text": "职场如战场而他是那个最没用的炮灰因为你业绩差又不努力",
                        "ref_time_range": {"start": 0.0, "end": 4.0},
                        "movie_time_ranges": [
                            {
                                "start": 100.0,
                                "end": 102.0,
                                "source_ref_shot_id": "ref_shot_001",
                                "movie_shot_ids": ["movie_shot_001"],
                                "confidence": "high",
                            },
                            {
                                "start": 200.0,
                                "end": 202.0,
                                "source_ref_shot_id": "ref_shot_002",
                                "movie_shot_ids": ["movie_shot_002"],
                                "confidence": "high",
                            },
                        ],
                    },
                    {
                        "segment_id": "seg_002",
                        "old_text": "每天只有无尽的加班",
                        "ref_time_range": {"start": 5.0, "end": 6.0},
                        "movie_time_ranges": [
                            {
                                "start": 300.0,
                                "end": 301.0,
                                "source_ref_shot_id": "ref_shot_003",
                                "movie_shot_ids": ["movie_shot_003"],
                                "confidence": "medium",
                            }
                        ],
                    },
                ]
            }
            ref_analysis = {
                "subtitle_srt": str(ref_srt),
                "ref_shots": [
                    {"ref_shot_id": "ref_shot_001", "start": 0.0, "end": 2.0},
                    {"ref_shot_id": "ref_shot_002", "start": 2.0, "end": 4.0},
                    {"ref_shot_id": "ref_shot_003", "start": 5.0, "end": 6.0},
                ],
            }

            result = audio_role.classify_audio_roles(
                script_mapping,
                ref_analysis,
                movie_subtitle_srt=movie_srt,
                output_dir=root / "out",
            )

            rows = result["script_mapping"]
            self.assertEqual(len(rows[0]["text_units"]), 2)
            self.assertEqual(rows[0]["movie_time_ranges"][0]["start"], 100.0)
            self.assertEqual(rows[0]["movie_time_ranges"][0]["end"], 102.0)
            self.assertEqual(rows[0]["movie_time_ranges"][0]["audio_action"], "rewrite_and_voiceover")
            self.assertEqual(rows[0]["movie_time_ranges"][1]["audio_role"], "original_dialogue")
            self.assertEqual(rows[0]["movie_time_ranges"][1]["audio_action"], "play_original_audio")
            self.assertEqual(rows[0]["segment_audio_role"], "mixed")
            self.assertEqual(rows[0]["text_units"][1]["role"], "original_dialogue")
            self.assertEqual(rows[0]["text_units"][1]["action"], "play_original_audio")
            self.assertEqual(rows[1]["text_units"][0]["role"], "narration")
            self.assertEqual(rows[1]["movie_time_ranges"][0]["movie_subtitles"], [])
            self.assertEqual(rows[1]["movie_time_ranges"][0]["audio_role"], "narration_overlay")

    def test_retries_bcut_asr_when_movie_subtitle_generation_is_flaky(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_srt = root / "ref.srt"
            movie_path = root / "movie.mp4"
            ref_srt.write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:00,000 --> 00:00:01,000",
                        "他又迟到了",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            movie_path.write_bytes(b"fake video")
            script_mapping = {
                "script_mapping": [
                    {
                        "segment_id": "seg_001",
                        "old_text": "他又迟到了",
                        "ref_time_range": {"start": 0.0, "end": 1.0},
                        "movie_time_ranges": [{"start": 10.0, "end": 11.0, "source_ref_shot_id": "ref_shot_001"}],
                    }
                ]
            }
            ref_analysis = {
                "subtitle_srt": str(ref_srt),
                "ref_shots": [{"ref_shot_id": "ref_shot_001", "start": 0.0, "end": 1.0}],
            }
            calls = {"count": 0}
            extracted_clips = []
            original_extract_clip = audio_role._extract_audio_clip_mp3
            original_bcut_entries = audio_role._bcut_entries_for_audio

            def fake_extract_clip(_movie_path: Path, audio_path: Path, start: float, end: float) -> Path:
                extracted_clips.append((round(start, 3), round(end, 3)))
                Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
                Path(audio_path).write_bytes(b"fake audio")
                return Path(audio_path)

            def flaky_bcut_entries(_audio_path: Path, offset: float) -> list[dict[str, object]]:
                calls["count"] += 1
                if calls["count"] < 3:
                    raise ConnectionError("remote disconnected")
                return [{"start": offset, "end": offset + 1.0, "text": "他又迟到了"}]

            audio_role._extract_audio_clip_mp3 = fake_extract_clip
            audio_role._bcut_entries_for_audio = flaky_bcut_entries
            try:
                result = audio_role.classify_audio_roles(
                    script_mapping,
                    ref_analysis,
                    movie_path=movie_path,
                    output_dir=root / "out",
                    asr_retries=3,
                    asr_retry_delay=0,
                )
            finally:
                audio_role._extract_audio_clip_mp3 = original_extract_clip
                audio_role._bcut_entries_for_audio = original_bcut_entries

            self.assertEqual(calls["count"], 3)
            self.assertEqual(extracted_clips, [(9.8, 11.2)])
            self.assertEqual(result["script_mapping"][0]["movie_time_ranges"][0]["audio_action"], "play_original_audio")


if __name__ == "__main__":
    unittest.main()
