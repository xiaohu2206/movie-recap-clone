from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()


def _load_generate_module():
    path = ROOT / "8_generate_video" / "run.py"
    spec = importlib.util.spec_from_file_location("generate_video_audio_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate = _load_generate_module()


class GenerateVideoAudioTests(unittest.TestCase):
    def test_synthesis_keeps_original_when_trimmed_mp3_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp) / "audio"

            async def fake_synthesize(*args, **kwargs):
                Path(kwargs["out_path"]).write_bytes(b"valid original audio placeholder")
                return {"success": True, "duration": 4.397}

            def fake_run_ffmpeg(args: list[str], *, check: bool = True):
                Path(args[-1]).write_bytes(b"invalid mp3")
                return subprocess.CompletedProcess(args, 0, "", "")

            def fake_probe_duration(path: str | Path) -> float:
                if Path(path).stem.endswith("_trim"):
                    raise RuntimeError("Failed to find two consecutive MPEG audio frames")
                return 4.397

            with patch.object(
                generate.edge_tts_service,
                "synthesize",
                new=AsyncMock(side_effect=fake_synthesize),
            ), patch.object(generate, "run_ffmpeg", side_effect=fake_run_ffmpeg), patch.object(
                generate,
                "probe_duration",
                side_effect=fake_probe_duration,
            ):
                result = asyncio.run(
                    generate._synthesize_one(
                        {"item_id": "item_006_narration", "narration": "测试配音"},
                        audio_dir,
                        voice_id="zh-CN-XiaoxiaoNeural",
                        speed_ratio=1.0,
                        proxy=None,
                        reuse=False,
                    )
                )

            output = audio_dir / "item_006_narration.mp3"
            self.assertEqual(result["duration"], 4.397)
            self.assertEqual(output.read_bytes(), b"valid original audio placeholder")
            self.assertFalse((audio_dir / "item_006_narration_trim.mp3").exists())

    def test_trim_rejects_nonempty_but_invalid_mp3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp3"
            trimmed = root / "source_trim.mp3"
            source.write_bytes(b"valid original audio placeholder")

            def fake_run_ffmpeg(args: list[str], *, check: bool = True):
                Path(args[-1]).write_bytes(b"invalid mp3")
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch.object(generate, "run_ffmpeg", side_effect=fake_run_ffmpeg), patch.object(
                generate,
                "probe_duration",
                side_effect=RuntimeError("Invalid data found when processing input"),
            ):
                accepted = generate._trim_tts_audio(source, trimmed)

            self.assertFalse(accepted)
            self.assertTrue(source.exists())
            self.assertFalse(trimmed.exists())

    def test_trim_accepts_decodable_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp3"
            trimmed = root / "source_trim.mp3"
            source.write_bytes(b"valid original audio placeholder")

            def fake_run_ffmpeg(args: list[str], *, check: bool = True):
                Path(args[-1]).write_bytes(b"valid trimmed audio placeholder")
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch.object(generate, "run_ffmpeg", side_effect=fake_run_ffmpeg), patch.object(
                generate,
                "probe_duration",
                return_value=1.25,
            ):
                accepted = generate._trim_tts_audio(source, trimmed)

            self.assertTrue(accepted)
            self.assertTrue(trimmed.exists())


if __name__ == "__main__":
    unittest.main()
