from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.shot_detection import detect_shots
import clone_narration_video.utils.shot_detection as shot_detection
import clone_narration_video.utils.transnetv2_torch as transnet_torch
from clone_narration_video.utils.asr.asr_bcut import BcutASR, BcutASRError, BCUT_MODEL_ID, _require_data
from clone_narration_video.utils.visual_features import (
    compare_homography_frames,
    compare_normalized_frames,
    load_normalized_frame,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


refinement = _load_module("alignment_refinement_test", ROOT / "4_visual_alignment_engine" / "refinement.py")
path_solver = _load_module("alignment_path_solver_test", ROOT / "4_visual_alignment_engine" / "path_solver.py")


def _write_image(path: Path, color: tuple[int, int, int], *, square: tuple[int, int] | None = None) -> str:
    img = np.full((96, 128, 3), color, dtype=np.uint8)
    if square:
        x, y = square
        cv2.rectangle(img, (x, y), (x + 28, y + 28), (255, 255, 255), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)


class VisualAlignmentV3Tests(unittest.TestCase):
    def test_detect_shots_exports_three_keyframes_and_optional_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "clip.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
            for idx in range(30):
                frame = np.full((64, 96, 3), 40 + idx, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            result = detect_shots(
                video_path,
                shot_prefix="ref_shot",
                keyframe_dir=root / "keyframes",
                backend="opencv",
                keyframe_positions="0.12,0.5,0.88",
                sample_fps=2,
                max_sample_frames_per_shot=4,
            )

            self.assertGreaterEqual(len(result["shots"]), 1)
            shot = result["shots"][0]
            self.assertEqual(len(shot["keyframes"]), 3)
            self.assertEqual(len(shot["keyframe_times"]), 3)
            self.assertLessEqual(len(shot["sample_frames"]), 4)
            for path in shot["keyframes"]:
                self.assertTrue(Path(path).exists())

    def test_bcut_missing_data_response_raises_descriptive_error(self) -> None:
        with self.assertRaisesRegex(BcutASRError, "code=-400"):
            _require_data({"code": -400, "message": "invalid task"}, "query result")

    def test_bcut_query_uses_configured_model_id(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"data": {"state": 4, "result": "{}"}}

        class FakeSession:
            def __init__(self) -> None:
                self.params = None

            def get(self, url, params=None, headers=None):
                self.params = params
                return FakeResponse()

        session = FakeSession()
        asr = BcutASR.__new__(BcutASR)
        asr.session = session
        asr.task_id = "task_1"

        self.assertEqual(asr.result()["state"], 4)
        self.assertEqual(session.params["model_id"], BCUT_MODEL_ID)

    def test_auto_backend_raises_when_transnet_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "clip.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
            for idx in range(30):
                color = 20 if idx < 15 else 210
                frame = np.full((64, 96, 3), color, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            original = shot_detection._detect_with_transnet

            def fail_transnet(*args, **kwargs):
                raise RuntimeError("torch dll failed")

            shot_detection._detect_with_transnet = fail_transnet
            try:
                with self.assertRaisesRegex(RuntimeError, "torch dll failed"):
                    detect_shots(
                        video_path,
                        shot_prefix="ref_shot",
                        keyframe_dir=root / "keyframes",
                        backend="auto",
                    )
            finally:
                shot_detection._detect_with_transnet = original

    def test_transnet_cuda_inference_failure_retries_on_cpu(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.devices: list[str] = []

            def to(self, device):
                self.devices.append(str(device))
                return self

        model = transnet_torch.TransNetV2Torch.__new__(transnet_torch.TransNetV2Torch)
        model._device = transnet_torch.torch.device("cuda")
        model._device_fallback_reason = None
        model._model = FakeModel()
        calls = {"count": 0}

        def predict_once(frames):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("cuda init failed")
            self.assertEqual(str(model._device), "cpu")
            return np.zeros((1, 1), dtype=np.float32), None

        model._predict_raw_on_current_device = predict_once
        frames = np.zeros((1, 100, 27, 48, 3), dtype=np.uint8)
        single, many = model.predict_raw(frames)

        self.assertEqual(calls["count"], 2)
        self.assertEqual(str(model._device), "cpu")
        self.assertIn("cuda init failed", model.get_backend_info()["device_fallback_reason"])
        self.assertEqual(single.shape, (1, 1))
        self.assertIsNone(many)

    def test_normalized_frame_comparison_survives_borders_and_luma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = np.zeros((120, 180, 3), dtype=np.uint8)
            cv2.rectangle(base, (40, 25), (145, 95), (80, 140, 210), -1)
            bordered = cv2.copyMakeBorder(base + 20, 16, 16, 22, 22, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            p1 = root / "base.jpg"
            p2 = root / "bordered.jpg"
            cv2.imwrite(str(p1), base)
            cv2.imwrite(str(p2), bordered)

            img1 = load_normalized_frame(p1)
            img2 = load_normalized_frame(p2)
            self.assertIsNotNone(img1)
            self.assertIsNotNone(img2)
            score = compare_normalized_frames(img1, img2)["score"]
            self.assertGreater(score, 0.65)

    def test_temporal_refinement_finds_best_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_frames = [
                {"path": _write_image(root / "ref0.jpg", (20, 20, 20), square=(8, 20)), "time": 0.0},
                {"path": _write_image(root / "ref1.jpg", (20, 20, 20), square=(40, 20)), "time": 1.0},
            ]
            movie_frames = [
                {"path": _write_image(root / "movie0.jpg", (20, 20, 20), square=(8, 20)), "time": 9.5},
                {"path": _write_image(root / "movie1.jpg", (20, 20, 20), square=(40, 20)), "time": 10.5},
                {"path": _write_image(root / "movie2.jpg", (120, 20, 20), square=(60, 20)), "time": 11.0},
            ]
            ref = {"ref_shot_id": "ref_001", "start": 0.0, "end": 2.0, "sample_frames": ref_frames}
            movie_shots = [
                {
                    "movie_shot_id": "movie_001",
                    "start": 9.0,
                    "end": 12.0,
                    "sample_frames": movie_frames,
                }
            ]
            candidates = [
                {
                    "movie_index": 0,
                    "movie_shot_id": "movie_001",
                    "movie_start": 9.0,
                    "movie_end": 12.0,
                    "visual_score": 0.45,
                    "recall_score": 0.45,
                    "detail": {},
                }
            ]

            refined = refinement.refine_candidates_for_ref(
                ref,
                candidates,
                movie_shots,
                alignment_mode="temporal",
                refine_top_k=1,
                temporal_radius_sec=1.0,
                temporal_step_sec=0.5,
            )

            self.assertEqual(refined[0]["movie_shot_id"], "movie_001")
            self.assertGreater(refined[0]["refinement_score"], 0.85)
            self.assertAlmostEqual(refined[0]["refinement"]["temporal_offset"], 9.5, delta=0.15)

    def test_homography_score_handles_shifted_frame(self) -> None:
        img = np.zeros((180, 240, 3), dtype=np.uint8)
        for x in range(20, 220, 30):
            cv2.circle(img, (x, 40 + (x % 70)), 8, (255, 255, 255), -1)
            cv2.putText(img, str(x), (x - 8, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 240), 1)
        matrix = np.float32([[1, 0, 14], [0, 1, 9]])
        shifted = cv2.warpAffine(img, matrix, (240, 180), borderMode=cv2.BORDER_REPLICATE)
        temporal = compare_normalized_frames(img, shifted)["score"]
        spatial = compare_homography_frames(img, shifted)["score"]
        self.assertGreaterEqual(spatial, temporal - 0.05)

    def test_segmented_path_solver_preserves_high_confidence_jump_anchor(self) -> None:
        ref_shots = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
        ]
        candidates = [
            [
                {
                    "movie_index": 0,
                    "movie_shot_id": "m001",
                    "movie_start": 100.0,
                    "movie_end": 101.0,
                    "visual_score": 0.9,
                    "refinement_score": 0.82,
                },
                {
                    "movie_index": 1,
                    "movie_shot_id": "m002",
                    "movie_start": 200.0,
                    "movie_end": 201.0,
                    "visual_score": 0.7,
                    "refinement_score": 0.7,
                },
            ],
            [
                {
                    "movie_index": 2,
                    "movie_shot_id": "m500",
                    "movie_start": 500.0,
                    "movie_end": 501.0,
                    "visual_score": 0.91,
                    "refinement_score": 0.83,
                },
                {
                    "movie_index": 3,
                    "movie_shot_id": "m102",
                    "movie_start": 102.0,
                    "movie_end": 103.0,
                    "visual_score": 0.7,
                    "refinement_score": 0.7,
                },
            ],
        ]

        chosen = path_solver.solve_segmented_global_path(ref_shots, candidates, min_visual_score=0.35)
        self.assertEqual(chosen[0]["movie_shot_id"], "m001")
        self.assertEqual(chosen[1]["movie_shot_id"], "m500")
        self.assertTrue(chosen[1]["diagnostics"]["path"]["anchor"])


if __name__ == "__main__":
    unittest.main()
