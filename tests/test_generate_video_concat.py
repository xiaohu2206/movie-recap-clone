from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()


def _load_generate_module():
    path = ROOT / "8_generate_video" / "run.py"
    spec = importlib.util.spec_from_file_location("generate_video_concat_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate = _load_generate_module()


class GenerateVideoConcatTests(unittest.TestCase):
    def test_batch_size_accounts_for_normalization_filters(self) -> None:
        paths = [Path(f"clip_{i:04d}.mp4") for i in range(337)]

        batch_size = generate._filter_concat_batch_size(paths, has_audio=True)

        self.assertLess(batch_size, 30)
        self.assertGreaterEqual(batch_size, 2)

    def test_concat_normalizes_video_size_and_audio_layout(self) -> None:
        paths = [Path("wide.mp4"), Path("small.mp4")]

        with (
            patch.object(
                generate,
                "_probe_video_meta",
                return_value={"width": 1920, "height": 801, "fps": 30.0, "duration": 1.0},
            ),
            patch.object(generate, "resolve_ffmpeg_bin", return_value="ffmpeg"),
            patch.object(generate, "_video_codec_args", return_value=["-c:v", "libx264"]),
            patch.object(generate, "run_ffmpeg") as run_ffmpeg,
        ):
            generate._concat_clips_filter_once(
                paths,
                Path("output.mp4"),
                encoder="libx264",
                has_audio=True,
            )

        cmd = run_ffmpeg.call_args.args[0]
        filter_complex = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("scale=1920:800:force_original_aspect_ratio=decrease", filter_complex)
        self.assertEqual(filter_complex.count("pad=1920:800"), 2)
        self.assertEqual(filter_complex.count("setsar=1"), 2)
        self.assertEqual(filter_complex.count("channel_layouts=stereo"), 2)
        self.assertIn("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]", filter_complex)

    def test_render_video_uses_blank_visual_for_narration_without_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            audio_path = output_dir / "narration.mp3"
            audio_path.write_bytes(b"audio")
            items = [
                {
                    "item_id": "item_005_narration",
                    "audio_type": "narration",
                    "narration": "test narration",
                    "video_clips": [],
                }
            ]
            audio_results = {
                "item_005_narration": {
                    "path": str(audio_path),
                    "duration": 1.855,
                }
            }

            def fake_mux(visual_path: Path, audio_path_arg: Path, out_path: Path, duration: float, *, encoder: str):
                out_path.write_bytes(b"muxed")
                return out_path

            def fake_concat(paths: list[Path], out_path: Path, **kwargs):
                out_path.write_bytes(b"final")
                return out_path

            with (
                patch.object(generate, "_select_video_encoder", return_value="libx264"),
                patch.object(generate, "_first_clip_video_meta", return_value={"width": 1920, "height": 1080, "fps": 30.0}),
                patch.object(generate, "_make_blank_video") as make_blank,
                patch.object(generate, "_mux_item_with_audio", side_effect=fake_mux) as mux,
                patch.object(generate, "_concat_videos", side_effect=fake_concat),
                patch.object(generate, "_apply_item_audio_boundary_fades"),
                patch.object(generate, "probe_duration", return_value=1.855),
            ):
                result = generate.render_video(
                    items,
                    audio_results,
                    output_dir,
                    output_name="out.mp4",
                    encoder="auto",
                )

            make_blank.assert_called_once()
            mux.assert_called_once()
            self.assertEqual(result["segments_count"], 1)
            self.assertEqual(result["duration"], 1.855)

    def test_render_video_still_requires_clips_for_original_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            items = [
                {
                    "item_id": "item_005_original",
                    "audio_type": "original_audio",
                    "keep_original_audio": True,
                    "video_clips": [],
                }
            ]
            audio_results = {
                "item_005_original": {
                    "path": "",
                    "duration": 1.5,
                    "use_original_audio": True,
                }
            }

            with (
                patch.object(generate, "_select_video_encoder", return_value="libx264"),
                patch.object(generate, "_first_clip_video_meta", return_value={"width": 1920, "height": 1080, "fps": 30.0}),
            ):
                with self.assertRaises(ValueError):
                    generate.render_video(
                        items,
                        audio_results,
                        Path(tmp),
                        output_name="out.mp4",
                        encoder="auto",
                    )

    def test_jianying_draft_reflows_from_real_audio_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()
            source = root / "movie.mp4"
            source.write_bytes(b"video")
            audio1 = root / "audio1.mp3"
            audio2 = root / "audio2.mp3"
            audio1.write_bytes(b"audio1")
            audio2.write_bytes(b"audio2")
            items = [
                {
                    "item_id": "item_001",
                    "audio_type": "narration",
                    "timeline_start": 0.0,
                    "timeline_end": 10.0,
                    "video_clips": [{"source": str(source), "movie_start": 0.0, "movie_end": 10.0}],
                },
                {
                    "item_id": "item_002",
                    "audio_type": "narration",
                    "timeline_start": 10.0,
                    "timeline_end": 12.0,
                    "video_clips": [{"source": str(source), "movie_start": 10.0, "movie_end": 12.0}],
                },
            ]
            audio_results = {
                "item_001": {"path": str(audio1), "duration": 5.0},
                "item_002": {"path": str(audio2), "duration": 2.0},
            }

            def fake_cut(_source: Path, _start: float, _duration: float, out_path: Path, **_kwargs):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"clip")
                return out_path

            with (
                patch.object(generate, "_select_video_encoder", return_value="libx264"),
                patch.object(generate, "_cut_video_clip", side_effect=fake_cut),
                patch.object(generate, "_probe_video_meta", return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 5.0}),
                patch.object(generate, "probe_duration", return_value=100.0),
                patch.object(generate, "run_ffmpeg"),
            ):
                result = generate.generate_jianying_draft(
                    items,
                    audio_results,
                    output_dir,
                    draft_name="UnitDraft",
                    target_draft_root=root / "drafts",
                )

            draft_info = json.loads((Path(result["draft_dir"]) / "draft_info.json").read_text(encoding="utf-8"))
            video_track = next(track for track in draft_info["tracks"] if track["type"] == "video")
            starts = [segment["target_timerange"]["start"] for segment in video_track["segments"]]

            self.assertEqual(starts, [0, 5_000_000])
            self.assertEqual(draft_info["duration"], 7_000_000)


if __name__ == "__main__":
    unittest.main()
