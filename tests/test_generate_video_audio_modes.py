from __future__ import annotations

import asyncio
import importlib.util
import json
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


generate_video = _load_module("generate_video_audio_modes_test", ROOT / "8_generate_video" / "run.py")


def _original_item() -> dict[str, object]:
    return {
        "item_id": "item_original",
        "audio_mode": "original",
        "narration": "",
        "video_clips": [
            {
                "source": "movie.mp4",
                "movie_start": 10.0,
                "movie_end": 12.5,
            }
        ],
    }


class GenerateVideoAudioModeTests(unittest.TestCase):
    def test_synthesize_one_original_returns_marker_without_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(
                generate_video._synthesize_one(
                    _original_item(),
                    Path(tmp) / "audio",
                    voice_id="voice",
                    speed_ratio=None,
                    proxy=None,
                    reuse=False,
                )
            )

            self.assertEqual(result["path"], "")
            self.assertEqual(result["duration"], 2.5)
            self.assertTrue(result["original_audio"])
            self.assertFalse(result["silent"])
            self.assertFalse((Path(tmp) / "audio").exists())

    def test_cut_video_clip_uses_audio_map_only_when_requested(self) -> None:
        commands: list[list[str]] = []
        original_run_ffmpeg = generate_video.run_ffmpeg
        original_resolve = generate_video.resolve_ffmpeg_bin
        original_codec_args = generate_video._video_codec_args

        def fake_run_ffmpeg(cmd: list[str], **_kwargs: object) -> object:
            commands.append(cmd)
            return object()

        generate_video.run_ffmpeg = fake_run_ffmpeg
        generate_video.resolve_ffmpeg_bin = lambda: "ffmpeg"
        generate_video._video_codec_args = lambda _encoder: ["-c:v", "libx264"]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                generate_video._cut_video_clip(root / "source.mp4", 0, 1, root / "silent.mp4", encoder="libx264")
                generate_video._cut_video_clip(root / "source.mp4", 0, 1, root / "original.mp4", encoder="libx264", keep_audio=True)
        finally:
            generate_video.run_ffmpeg = original_run_ffmpeg
            generate_video.resolve_ffmpeg_bin = original_resolve
            generate_video._video_codec_args = original_codec_args

        self.assertIn("-an", commands[0])
        self.assertNotIn("0:a?", commands[0])
        self.assertIn("0:a?", commands[1])
        self.assertIn("-c:a", commands[1])
        self.assertIn("-af", commands[1])
        self.assertIn("aresample=async=1:first_pts=0", commands[1][commands[1].index("-af") + 1])
        self.assertEqual(commands[1][commands[1].index("-ar") + 1], "48000")
        self.assertEqual(commands[1][commands[1].index("-ac") + 1], "2")

    def test_mux_item_with_audio_normalizes_audio_for_concat(self) -> None:
        commands: list[list[str]] = []
        original_run_ffmpeg = generate_video.run_ffmpeg
        original_resolve = generate_video.resolve_ffmpeg_bin
        original_codec_args = generate_video._video_codec_args
        original_probe = generate_video.probe_duration

        def fake_run_ffmpeg(cmd: list[str], **_kwargs: object) -> object:
            commands.append(cmd)
            return object()

        generate_video.run_ffmpeg = fake_run_ffmpeg
        generate_video.resolve_ffmpeg_bin = lambda: "ffmpeg"
        generate_video._video_codec_args = lambda _encoder: ["-c:v", "libx264"]
        generate_video.probe_duration = lambda _path: 1.0
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                generate_video._mux_item_with_audio(root / "visual.mp4", root / "audio.mp3", root / "out.mp4", 1.2, encoder="libx264")
        finally:
            generate_video.run_ffmpeg = original_run_ffmpeg
            generate_video.resolve_ffmpeg_bin = original_resolve
            generate_video._video_codec_args = original_codec_args
            generate_video.probe_duration = original_probe

        self.assertEqual(len(commands), 1)
        self.assertIn("aresample=async=1:first_pts=0", commands[0][commands[0].index("-filter_complex") + 1])
        self.assertEqual(commands[0][commands[0].index("-ar") + 1], "48000")
        self.assertEqual(commands[0][commands[0].index("-ac") + 1], "2")

    def test_render_video_original_keeps_audio_and_skips_mux(self) -> None:
        calls: dict[str, object] = {"cut_keep_audio": [], "concat_reencode": [], "mux": 0}
        originals = {
            "select": generate_video._select_video_encoder,
            "source": generate_video._source_path,
            "cut": generate_video._cut_video_clip,
            "concat": generate_video._concat_videos,
            "mux": generate_video._mux_item_with_audio,
            "probe": generate_video.probe_duration,
        }

        def fake_cut(_source: Path, _start: float, _duration: float, out_path: Path, *, encoder: str, keep_audio: bool = False, audio_volume: float = 1.0) -> Path:
            calls["cut_keep_audio"].append(keep_audio)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"video")
            return out_path

        def fake_concat(_paths: list[Path], out_path: Path, *, reencode: bool = False, encoder: str = "auto") -> Path:
            calls["concat_reencode"].append(reencode)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"concat")
            return out_path

        def fake_mux(*_args: object, **_kwargs: object) -> Path:
            calls["mux"] = int(calls["mux"]) + 1
            raise AssertionError("original item should not mux TTS audio")

        generate_video._select_video_encoder = lambda encoder: encoder
        generate_video._source_path = lambda raw: Path(raw)
        generate_video._cut_video_clip = fake_cut
        generate_video._concat_videos = fake_concat
        generate_video._mux_item_with_audio = fake_mux
        generate_video.probe_duration = lambda _path: 2.5
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = generate_video.render_video(
                    [_original_item()],
                    {"item_original": {"path": "", "duration": 2.5, "original_audio": True}},
                    Path(tmp),
                    output_name="out.mp4",
                    encoder="libx264",
                )
        finally:
            generate_video._select_video_encoder = originals["select"]
            generate_video._source_path = originals["source"]
            generate_video._cut_video_clip = originals["cut"]
            generate_video._concat_videos = originals["concat"]
            generate_video._mux_item_with_audio = originals["mux"]
            generate_video.probe_duration = originals["probe"]

        self.assertEqual(calls["cut_keep_audio"], [True])
        self.assertEqual(calls["concat_reencode"], [True, True])
        self.assertEqual(calls["mux"], 0)
        self.assertEqual(result["segments_count"], 1)

    def test_jianying_draft_original_keeps_video_volume_and_skips_audio_track(self) -> None:
        originals = {
            "source": generate_video._source_path,
            "probe_meta": generate_video._probe_video_meta,
            "probe_duration": generate_video.probe_duration,
            "run_ffmpeg": generate_video.run_ffmpeg,
            "resolve_ffmpeg": generate_video.resolve_ffmpeg_bin,
        }
        generate_video._probe_video_meta = lambda _path: {"width": 1920, "height": 1080, "fps": 30.0, "duration": 2.5}
        generate_video.probe_duration = lambda _path: 30.0
        generate_video.run_ffmpeg = lambda *_args, **_kwargs: object()
        generate_video.resolve_ffmpeg_bin = lambda: "ffmpeg"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "movie.mp4"
                source.write_bytes(b"video")
                generate_video._source_path = lambda _raw: source
                result = generate_video.generate_jianying_draft(
                    [_original_item()],
                    {"item_original": {"path": "", "duration": 2.5, "original_audio": True}},
                    root / "out",
                    draft_name="TestDraft",
                    target_draft_root=root / "drafts",
                )
                draft_info = json.loads((Path(result["draft_dir"]) / "draft_info.json").read_text(encoding="utf-8"))
        finally:
            generate_video._source_path = originals["source"]
            generate_video._probe_video_meta = originals["probe_meta"]
            generate_video.probe_duration = originals["probe_duration"]
            generate_video.run_ffmpeg = originals["run_ffmpeg"]
            generate_video.resolve_ffmpeg_bin = originals["resolve_ffmpeg"]

        video_segments = draft_info["tracks"][0]["segments"]
        audio_segments = draft_info["tracks"][1]["segments"]
        self.assertEqual(video_segments[0]["volume"], 1)
        self.assertEqual(audio_segments, [])
        self.assertEqual(draft_info["materials"]["audios"], [])


if __name__ == "__main__":
    unittest.main()
