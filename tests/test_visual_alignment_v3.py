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


alignment = _load_module("alignment_run_test", ROOT / "4_visual_alignment_engine" / "run.py")


class VisualAlignmentTests(unittest.TestCase):
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

    def test_shot_localizer_selects_best_matching_movie_clip(self) -> None:
        def make_frame(color: tuple[int, int, int], square_x: int) -> np.ndarray:
            frame = np.full((96, 128, 3), color, dtype=np.uint8)
            cv2.rectangle(frame, (square_x, 28), (square_x + 30, 58), (255, 255, 255), -1)
            return frame

        def make_feature(name: str, frames: list[np.ndarray]) -> object:
            localizer = alignment.shot_localizer
            frame_features = np.stack([localizer.frame_descriptor(frame) for frame in frames]).astype(np.float32)
            return localizer.ShotFeature(
                path=Path(name),
                duration=1.0,
                frame_features=frame_features,
                clip_feature=localizer.normalize_vector(frame_features.mean(axis=0)),
                local_features=[],
            )

        ref = make_feature("ref_001.mp4", [make_frame((20, 30, 40), 12), make_frame((20, 30, 40), 62)])
        wrong = make_feature("movie_wrong.mp4", [make_frame((170, 50, 50), 16), make_frame((170, 50, 50), 70)])
        best = make_feature("movie_best.mp4", [make_frame((20, 30, 40), 12), make_frame((20, 30, 40), 62)])
        other = make_feature("movie_other.mp4", [make_frame((30, 120, 60), 38), make_frame((30, 120, 60), 78)])

        movie_features = [wrong, best, other]
        localizer = alignment.shot_localizer
        similarity = localizer.cosine_matrix([ref], movie_features)
        matches = localizer.independent_localize(
            [ref],
            movie_features,
            similarity,
            neighbor_radius=0,
            candidate_count=3,
            top_k=3,
        )

        self.assertEqual(matches[0]["movie_name"], "movie_best.mp4")
        self.assertGreater(matches[0]["score"], matches[0]["top_candidates"][1]["score"])

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

    def test_alignment_adapter_maps_localizer_index_to_pipeline_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_dir = root / "ref"
            movie_dir = root / "movie"
            ref_dir.mkdir()
            movie_dir.mkdir()
            (ref_dir / "ref_shot_001.mp4").touch()
            (movie_dir / "movie_shot_000001.mp4").touch()
            (movie_dir / "movie_shot_000002.mp4").touch()

            localizer = alignment.shot_localizer
            original_extract = localizer.extract_all_features
            original_localize = localizer.independent_localize

            def fake_extract(paths, sample_count, size, workers, mask_text_bands):
                features = []
                for index, path in enumerate(paths):
                    vector = np.zeros(4, dtype=np.float32)
                    vector[index % 4] = 1.0
                    features.append(
                        localizer.ShotFeature(
                            path=path,
                            duration=1.0,
                            frame_features=vector[None, :],
                            clip_feature=vector,
                            local_features=[],
                        )
                    )
                return features

            def fake_localize(ref_features, movie_features, similarity, neighbor_radius, candidate_count, top_k):
                selected = {
                    "movie_index": 1,
                    "movie_name": movie_features[1].path.name,
                    "score": 0.82,
                    "geometry_score": 0.8,
                    "geometry_inliers": 24,
                    "good_matches": 30,
                    "local_score": 0.7,
                    "global_score": 0.75,
                    "fine_score": 0.8,
                    "sequence_support": 0.0,
                    "movie_duration": 1.0,
                }
                return [{
                    "ref_index": 0,
                    "movie_index": 1,
                    "score": 0.82,
                    "geometry_inliers": 24,
                    "good_matches": 30,
                    "top_candidates": [selected],
                }]

            localizer.extract_all_features = fake_extract
            localizer.independent_localize = fake_localize
            try:
                result = alignment.align_visual_timeline(
                    {"ref_shots": [{"ref_shot_id": "ref_shot_001", "start": 1.0, "end": 2.5}]},
                    {"movie_shots": [
                        {"movie_shot_id": "movie_shot_000001", "start": 10.0, "end": 11.0},
                        {"movie_shot_id": "movie_shot_000002", "start": 20.0, "end": 21.5},
                    ]},
                    ref_clip_dir=ref_dir,
                    movie_clip_dir=movie_dir,
                    export_pairs=False,
                    min_geometry_inliers=20,
                )
            finally:
                localizer.extract_all_features = original_extract
                localizer.independent_localize = original_localize

            row = result["ref_to_movie_timeline"][0]
            self.assertEqual(row["ref_shot_id"], "ref_shot_001")
            self.assertEqual(row["movie_shot_ids"], ["movie_shot_000002"])
            self.assertEqual((row["movie_start"], row["movie_end"]), (20.0, 21.5))
            self.assertEqual(row["status"], "matched")

    def test_alignment_uses_keyframes_when_shot_clips_are_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_frame = root / "ref.jpg"
            movie_frame = root / "movie.jpg"
            image = np.zeros((96, 128, 3), dtype=np.uint8)
            cv2.rectangle(image, (24, 20), (92, 72), (60, 180, 240), -1)
            cv2.putText(image, "MATCH", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(ref_frame), image)
            cv2.imwrite(str(movie_frame), image)

            result = alignment.align_visual_timeline(
                {"ref_shots": [{
                    "ref_shot_id": "ref_shot_001",
                    "start": 0.0,
                    "end": 1.0,
                    "keyframes": [str(ref_frame)],
                }]},
                {"movie_shots": [{
                    "movie_shot_id": "movie_shot_000001",
                    "start": 10.0,
                    "end": 11.0,
                    "keyframes": [str(movie_frame)],
                }]},
                ref_clip_dir=root / "missing_ref_clips",
                movie_clip_dir=root / "missing_movie_clips",
                output_dir=root / "alignment",
                diagnostics_dir=root / "diagnostics",
                export_pairs=False,
                candidate_count=1,
                top_k=1,
                min_geometry_inliers=1,
            )

            row = result["ref_to_movie_timeline"][0]
            self.assertEqual(row["movie_shot_ids"], ["movie_shot_000001"])
            self.assertEqual(result["metadata"]["reference_keyframe_fallback_count"], 1)
            self.assertEqual(result["metadata"]["movie_keyframe_fallback_count"], 1)


if __name__ == "__main__":
    unittest.main()
