#!/usr/bin/env python3
"""
Shot-to-shot localization pipeline for matching reference clips to movie clips.

This script is designed for already-segmented shot clips. It combines compact
frame descriptors with ORB candidate retrieval and geometric verification, then
exports independently localized reference/movie clip pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cli_bootstrap import add_project_to_syspath

add_project_to_syspath()

from clone_narration_video.utils.ffmpeg_utils import resolve_ffmpeg_bin, resolve_ffprobe_bin


@dataclass
class LocalFrameFeature:
    keypoints: np.ndarray
    descriptors: np.ndarray


@dataclass
class ShotFeature:
    path: Path
    duration: float
    frame_features: np.ndarray
    clip_feature: np.ndarray
    local_features: list[LocalFrameFeature]


@dataclass(frozen=True)
class KeyframeSource:
    path: Path
    frame_paths: tuple[Path, ...]
    duration: float


@dataclass
class LocalHashIndex:
    movie_count: int
    postings: dict[int, tuple[np.ndarray, np.ndarray]]


def list_video_files(root: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def probe_duration(path: Path) -> float:
    cmd = [
        resolve_ffprobe_bin(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    kwargs = {"capture_output": True, "text": True, "check": True}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    result = subprocess.run(cmd, **kwargs)
    duration_text = result.stdout.strip()
    return max(float(duration_text), 0.1) if duration_text else 0.1


def sample_frames(path: Path, duration: float, sample_count: int, size: int) -> list[np.ndarray]:
    fps = max(sample_count / max(duration, 0.1), 0.5)
    cmd = [
        resolve_ffmpeg_bin(),
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:black",
        "-frames:v",
        str(sample_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    kwargs = {"capture_output": True, "check": True}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    result = subprocess.run(cmd, **kwargs)
    frame_size = size * size * 3
    raw = result.stdout
    frame_count = len(raw) // frame_size
    frames = []
    for idx in range(frame_count):
        start = idx * frame_size
        end = start + frame_size
        frame = np.frombuffer(raw[start:end], dtype=np.uint8).reshape(size, size, 3)
        frames.append(frame.copy())
    return frames


def load_keyframes(paths: Iterable[Path], size: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for path in paths:
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except OSError:
            image = None
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        scale = min(size / max(1, width), size / max(1, height))
        resized = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        y = (size - resized.shape[0]) // 2
        x = (size - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        frames.append(canvas)
    return frames


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def frame_descriptor(frame: np.ndarray) -> np.ndarray:
    # Drop the subtitle-heavy bottom band and preserve a center crop view.
    height, width = frame.shape[:2]
    useful = frame[: max(1, int(height * 0.82)), :]
    x1 = int(width * 0.1)
    x2 = max(x1 + 1, int(width * 0.9))
    y1 = int(useful.shape[0] * 0.1)
    y2 = max(y1 + 1, int(useful.shape[0] * 0.9))
    center = useful[y1:y2, x1:x2]

    useful_gray = cv2.cvtColor(useful, cv2.COLOR_RGB2GRAY)
    center_gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
    useful_gray = cv2.resize(useful_gray, (32, 32), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    center_gray = cv2.resize(center_gray, (32, 32), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    useful_gray = (useful_gray - useful_gray.mean()) / (useful_gray.std() + 1e-6)
    center_gray = (center_gray - center_gray.mean()) / (center_gray.std() + 1e-6)

    resized = cv2.resize(useful, (64, 64), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    hist /= hist.sum() + 1e-6

    descriptor = np.concatenate([useful_gray.reshape(-1), center_gray.reshape(-1), hist], axis=0)
    return normalize_vector(descriptor)


def extract_local_features(frames: list[np.ndarray], mask_text_bands: bool) -> list[LocalFrameFeature]:
    orb = cv2.ORB_create(nfeatures=900, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    selected_indices = sorted({0, len(frames) // 2, len(frames) - 1})
    local_features = []
    for idx in selected_indices:
        gray = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
        mask = None
        if mask_text_bands:
            height, width = gray.shape
            mask = np.zeros_like(gray)
            mask[int(height * 0.10):int(height * 0.82), int(width * 0.02):int(width * 0.98)] = 255
        keypoints, descriptors = orb.detectAndCompute(gray, mask)
        if descriptors is None or not keypoints:
            continue
        local_features.append(
            LocalFrameFeature(
                keypoints=np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32),
                descriptors=descriptors,
            )
        )
    return local_features


def extract_shot_feature(path: Path, sample_count: int, size: int, mask_text_bands: bool) -> ShotFeature:
    duration = probe_duration(path)
    frames = sample_frames(path, duration=duration, sample_count=sample_count, size=size)
    if not frames:
        raise RuntimeError(f"No frames extracted from {path}")

    frame_features = np.stack([frame_descriptor(frame) for frame in frames]).astype(np.float32)
    clip_feature = normalize_vector(frame_features.mean(axis=0))
    local_features = extract_local_features(frames, mask_text_bands=mask_text_bands)
    return ShotFeature(
        path=path,
        duration=duration,
        frame_features=frame_features,
        clip_feature=clip_feature,
        local_features=local_features,
    )


def extract_source_feature(
    source: KeyframeSource,
    sample_count: int,
    size: int,
    mask_text_bands: bool,
) -> ShotFeature:
    if not source.frame_paths:
        return extract_shot_feature(source.path, sample_count, size, mask_text_bands)

    frames = load_keyframes(source.frame_paths, size)
    if not frames:
        raise RuntimeError(f"No keyframes could be loaded for {source.path}")

    frame_features = np.stack([frame_descriptor(frame) for frame in frames]).astype(np.float32)
    return ShotFeature(
        path=source.path,
        duration=max(float(source.duration), 0.1),
        frame_features=frame_features,
        clip_feature=normalize_vector(frame_features.mean(axis=0)),
        local_features=extract_local_features(frames, mask_text_bands=mask_text_bands),
    )


def extract_all_features(
    paths: Iterable[Path],
    sample_count: int,
    size: int,
    workers: int,
    mask_text_bands: bool,
) -> list[ShotFeature]:
    paths = list(paths)
    results: dict[Path, ShotFeature] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_shot_feature, path, sample_count, size, mask_text_bands): path
            for path in paths
        }
        for future in as_completed(futures):
            feature = future.result()
            results[feature.path] = feature
            print(f"[feature] {len(results)}/{len(paths)} {feature.path.name}", flush=True)
    return [results[path] for path in paths]


def extract_all_keyframe_features(
    sources: Iterable[KeyframeSource],
    sample_count: int,
    size: int,
    workers: int,
    mask_text_bands: bool,
) -> list[ShotFeature]:
    sources = list(sources)
    results: dict[Path, ShotFeature] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                extract_source_feature,
                source,
                sample_count,
                size,
                mask_text_bands,
            ): source
            for source in sources
        }
        for future in as_completed(futures):
            feature = future.result()
            results[feature.path] = feature
            print(f"[feature] {len(results)}/{len(sources)} {feature.path.name} (keyframes)", flush=True)
    return [results[source.path] for source in sources]


def cosine_matrix(ref_features: list[ShotFeature], movie_features: list[ShotFeature]) -> np.ndarray:
    ref = np.stack([item.clip_feature for item in ref_features]).astype(np.float32)
    movie = np.stack([item.clip_feature for item in movie_features]).astype(np.float32)
    return ref @ movie.T


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = max(1, min(int(count), len(values)))
    all_indices = np.arange(len(values))
    if count >= len(values):
        return all_indices[np.lexsort((all_indices, values))[::-1]]
    indices = np.argpartition(values, -count)[-count:]
    return indices[np.lexsort((indices, values[indices]))[::-1]]


def fine_score(ref_item: ShotFeature, movie_item: ShotFeature) -> float:
    sim = ref_item.frame_features @ movie_item.frame_features.T
    row_max = np.max(sim, axis=1).mean()
    col_max = np.max(sim, axis=0).mean()
    duration_ratio = min(ref_item.duration, movie_item.duration) / max(ref_item.duration, movie_item.duration)
    return float(0.45 * (ref_item.clip_feature @ movie_item.clip_feature) + 0.45 * ((row_max + col_max) / 2.0) + 0.10 * duration_ratio)


LOCAL_HASH_BYTE_PAIRS = ((0, 7), (4, 25), (8, 15), (11, 18), (16, 27), (22, 29))
LOCAL_HASH_BUCKETS = 4096


def descriptor_hashes(descriptors: np.ndarray) -> np.ndarray:
    columns = []
    for table_idx, (left, right) in enumerate(LOCAL_HASH_BYTE_PAIRS):
        values = (
            descriptors[:, left].astype(np.int32) << 4
        ) | (descriptors[:, right].astype(np.int32) & 0x0F)
        columns.append(values + table_idx * LOCAL_HASH_BUCKETS)
    return np.concatenate(columns)


def build_local_hash_index(movie_features: list[ShotFeature]) -> tuple[LocalHashIndex, np.ndarray]:
    movie_hash_counts: list[tuple[np.ndarray, np.ndarray] | None] = []
    document_frequency = np.zeros(
        len(LOCAL_HASH_BYTE_PAIRS) * LOCAL_HASH_BUCKETS,
        dtype=np.int32,
    )
    for feature in movie_features:
        descriptors = [item.descriptors for item in feature.local_features if len(item.descriptors)]
        if not descriptors:
            movie_hash_counts.append(None)
            continue
        hashes = descriptor_hashes(np.vstack(descriptors))
        unique_hashes, counts = np.unique(hashes, return_counts=True)
        movie_hash_counts.append((unique_hashes, counts))
        document_frequency[unique_hashes] += 1

    idf = np.log((len(movie_features) + 1.0) / (document_frequency + 1.0)) + 1.0
    posting_lists: dict[int, list[tuple[int, float]]] = {}
    for movie_idx, hash_counts in enumerate(movie_hash_counts):
        if hash_counts is None:
            continue
        unique_hashes, counts = hash_counts
        weights = np.log1p(counts).astype(np.float32) * idf[unique_hashes]
        norm = float(np.linalg.norm(weights))
        if norm > 1e-6:
            weights /= norm
        for hash_value, weight in zip(unique_hashes, weights):
            posting_lists.setdefault(int(hash_value), []).append((movie_idx, float(weight)))
    postings = {
        hash_value: (
            np.fromiter((item[0] for item in rows), dtype=np.int32, count=len(rows)),
            np.fromiter((item[1] for item in rows), dtype=np.float32, count=len(rows)),
        )
        for hash_value, rows in posting_lists.items()
    }
    return LocalHashIndex(movie_count=len(movie_features), postings=postings), idf.astype(np.float32)


def local_hash_scores(
    ref_item: ShotFeature,
    movie_index: LocalHashIndex,
    idf: np.ndarray,
) -> np.ndarray:
    descriptors = [item.descriptors for item in ref_item.local_features if len(item.descriptors)]
    if not descriptors:
        return np.zeros(movie_index.movie_count, dtype=np.float32)
    hashes = descriptor_hashes(np.vstack(descriptors))
    unique_hashes, counts = np.unique(hashes, return_counts=True)
    query_weights = np.log1p(counts).astype(np.float32) * idf[unique_hashes]
    norm = float(np.linalg.norm(query_weights))
    if norm > 1e-6:
        query_weights /= norm
    scores = np.zeros(movie_index.movie_count, dtype=np.float32)
    for hash_value, query_weight in zip(unique_hashes, query_weights):
        posting = movie_index.postings.get(int(hash_value))
        if posting is None:
            continue
        movie_indices, movie_weights = posting
        scores[movie_indices] += float(query_weight) * movie_weights
    return scores


def build_local_ann(movie_features: list[ShotFeature]) -> tuple[cv2.flann_Index, np.ndarray] | None:
    descriptor_blocks = []
    owners = []
    for movie_idx, feature in enumerate(movie_features):
        for frame in feature.local_features:
            descriptor_blocks.append(frame.descriptors)
            owners.append(np.full(len(frame.descriptors), movie_idx, dtype=np.int32))
    if not descriptor_blocks:
        return None
    descriptors = np.vstack(descriptor_blocks)
    descriptor_owners = np.concatenate(owners)
    index = cv2.flann_Index(
        descriptors,
        {
            "algorithm": 6,
            "table_number": 12,
            "key_size": 20,
            "multi_probe_level": 2,
        },
    )
    return index, descriptor_owners


def ann_local_candidates(
    ref_item: ShotFeature,
    index: cv2.flann_Index,
    descriptor_owners: np.ndarray,
    movie_count: int,
    candidate_count: int,
) -> np.ndarray:
    votes = np.zeros(movie_count, dtype=np.float32)
    for frame in ref_item.local_features:
        descriptors = frame.descriptors
        if len(descriptors) > 320:
            positions = np.linspace(0, len(descriptors) - 1, 320, dtype=np.int32)
            descriptors = descriptors[positions]
        indices, distances = index.knnSearch(descriptors, 6, params={"checks": 32})
        for descriptor_idx in range(len(indices)):
            seen = set()
            for neighbor_idx, distance in zip(indices[descriptor_idx], distances[descriptor_idx]):
                owner = int(descriptor_owners[neighbor_idx])
                if owner in seen or distance > 60:
                    continue
                seen.add(owner)
                votes[owner] += max(0.0, (68.0 - float(distance)) / 68.0)
    return top_indices(votes, candidate_count)


def _limited_local_feature(feature: LocalFrameFeature, max_descriptors: int = 600) -> LocalFrameFeature:
    if len(feature.descriptors) <= max_descriptors:
        return feature
    positions = np.linspace(0, len(feature.descriptors) - 1, max_descriptors, dtype=np.int32)
    return LocalFrameFeature(
        keypoints=feature.keypoints[positions],
        descriptors=feature.descriptors[positions],
    )


def geometric_match_score(ref_item: ShotFeature, movie_item: ShotFeature) -> tuple[float, int, int]:
    if not ref_item.local_features or not movie_item.local_features:
        return 0.0, 0, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    best_inliers = 0
    best_good = 0
    ref_frames = [_limited_local_feature(frame) for frame in ref_item.local_features]
    movie_frames = [_limited_local_feature(frame) for frame in movie_item.local_features]
    frame_pairs = []
    for ref_pos, ref_frame in enumerate(ref_frames):
        for movie_pos, movie_frame in enumerate(movie_frames):
            frame_pairs.append((abs(ref_pos - movie_pos), ref_pos, movie_pos, ref_frame, movie_frame))
    frame_pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    for _, _, _, ref_frame, movie_frame in frame_pairs:
        pairs = matcher.knnMatch(ref_frame.descriptors, movie_frame.descriptors, k=2)
        good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        inliers = 0
        if len(good) >= 4 and len(good) >= best_inliers:
            source = np.asarray(
                [ref_frame.keypoints[match.queryIdx] for match in good],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            target = np.asarray(
                [movie_frame.keypoints[match.trainIdx] for match in good],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())
        if (inliers, len(good)) > (best_inliers, best_good):
            best_inliers = inliers
            best_good = len(good)
            if best_inliers >= 35 and best_good >= 50:
                break

    score = 0.75 * min(best_inliers / 35.0, 1.0) + 0.25 * min(best_good / 50.0, 1.0)
    return float(score), best_inliers, best_good


def independent_localize(
    ref_features: list[ShotFeature],
    movie_features: list[ShotFeature],
    global_similarity: np.ndarray,
    neighbor_radius: int,
    candidate_count: int,
    top_k: int,
    geometry_candidate_count: int | None = None,
    workers: int = 1,
) -> list[dict]:
    if not ref_features:
        return []
    if not movie_features:
        raise RuntimeError("No movie shot features are available for localization")
    candidate_count = max(1, min(int(candidate_count), len(movie_features)))
    top_k = max(1, int(top_k))
    geometry_limit = len(movie_features) if geometry_candidate_count is None else int(geometry_candidate_count)
    geometry_limit = max(top_k, min(max(1, geometry_limit), len(movie_features)))
    workers = max(1, int(workers))
    print("[match] building local feature index", flush=True)
    movie_index, idf = build_local_hash_index(movie_features)
    local_scores = []
    local_orders = []
    for idx, ref_item in enumerate(ref_features):
        scores = local_hash_scores(ref_item, movie_index, idf)
        local_scores.append(scores)
        local_orders.append(top_indices(scores, candidate_count))
        print(f"[retrieve] {idx + 1}/{len(ref_features)} {ref_item.path.name}", flush=True)

    global_order_count = max(15, min(len(movie_features), geometry_limit))
    global_orders = [
        top_indices(global_similarity[ref_idx], global_order_count)
        for ref_idx in range(len(ref_features))
    ]

    candidate_sets = []
    for ref_idx, ref_item in enumerate(ref_features):
        candidate_indices = set(int(item) for item in local_orders[ref_idx])
        for neighbor_idx in range(
            max(0, ref_idx - neighbor_radius),
            min(len(ref_features), ref_idx + neighbor_radius + 1),
        ):
            candidate_indices.update(int(item) for item in local_orders[neighbor_idx][:5])
            candidate_indices.update(int(item) for item in global_orders[neighbor_idx][:5])
        candidate_indices.update(int(item) for item in global_orders[ref_idx][:15])
        candidate_sets.append(candidate_indices)

    geometry_cache: dict[tuple[int, int], tuple[float, int, int]] = {}
    fine_cache: dict[tuple[int, int], float] = {}
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def get_fine(ref_idx: int, movie_idx: int) -> float:
        cache_key = (ref_idx, movie_idx)
        if cache_key not in fine_cache:
            fine_cache[cache_key] = float(np.clip(fine_score(ref_features[ref_idx], movie_features[movie_idx]), 0.0, 1.0))
        return fine_cache[cache_key]

    def cheap_candidate_score(ref_idx: int, movie_idx: int, sequence_support: dict[int, float]) -> float:
        ref_item = ref_features[ref_idx]
        movie_item = movie_features[movie_idx]
        max_local = max(float(local_scores[ref_idx].max()), 1e-6)
        local_score = float(local_scores[ref_idx][movie_idx] / max_local)
        global_score = float(np.clip(global_similarity[ref_idx, movie_idx], 0.0, 1.0))
        fine = get_fine(ref_idx, movie_idx)
        duration_ratio = min(ref_item.duration, movie_item.duration) / max(ref_item.duration, movie_item.duration)
        support = min(sequence_support.get(movie_idx, 0.0), 0.30)
        return 0.35 * local_score + 0.25 * fine + 0.25 * global_score + 0.10 * duration_ratio + support

    def geometry_indices(ref_idx: int, sequence_support: dict[int, float]) -> list[int]:
        candidates = candidate_sets[ref_idx]
        if len(candidates) <= geometry_limit:
            return list(candidates)

        forced = set(int(item) for item in local_orders[ref_idx][:min(10, geometry_limit)])
        forced.update(int(item) for item in global_orders[ref_idx][:min(8, geometry_limit)])
        forced.update(int(item) for item in sequence_support)

        ranked = sorted(
            candidates,
            key=lambda movie_idx: cheap_candidate_score(ref_idx, int(movie_idx), sequence_support),
            reverse=True,
        )
        selected: list[int] = []
        seen: set[int] = set()
        for movie_idx in [*forced, *ranked]:
            movie_idx = int(movie_idx)
            if movie_idx in seen or movie_idx not in candidates:
                continue
            seen.add(movie_idx)
            selected.append(movie_idx)
            if len(selected) >= max(geometry_limit, len(forced)):
                break
        return selected

    def ensure_geometry(ref_idx: int, movie_indices: list[int]) -> None:
        missing = [
            (ref_idx, movie_idx)
            for movie_idx in movie_indices
            if (ref_idx, movie_idx) not in geometry_cache
        ]
        if not missing:
            return
        if executor is None:
            for ref_item_idx, movie_idx in missing:
                geometry_cache[(ref_item_idx, movie_idx)] = geometric_match_score(
                    ref_features[ref_item_idx],
                    movie_features[movie_idx],
                )
            return
        futures = {
            executor.submit(geometric_match_score, ref_features[ref_item_idx], movie_features[movie_idx]): (
                ref_item_idx,
                movie_idx,
            )
            for ref_item_idx, movie_idx in missing
        }
        for future in as_completed(futures):
            geometry_cache[futures[future]] = future.result()

    def rank_candidates(ref_idx: int, sequence_support: dict[int, float]) -> list[dict]:
        ref_item = ref_features[ref_idx]
        max_local = max(float(local_scores[ref_idx].max()), 1e-6)
        selected_indices = geometry_indices(ref_idx, sequence_support)
        ensure_geometry(ref_idx, selected_indices)
        ranked = []
        for movie_idx in selected_indices:
            movie_item = movie_features[movie_idx]
            cache_key = (ref_idx, movie_idx)
            geometry, inliers, good_matches = geometry_cache[cache_key]
            local_score = float(local_scores[ref_idx][movie_idx] / max_local)
            global_score = float(np.clip(global_similarity[ref_idx, movie_idx], 0.0, 1.0))
            fine = get_fine(ref_idx, movie_idx)
            duration_ratio = min(ref_item.duration, movie_item.duration) / max(ref_item.duration, movie_item.duration)
            support = min(sequence_support.get(movie_idx, 0.0), 0.30)
            if inliers >= 8:
                score = 0.55 + 0.30 * geometry + 0.05 * local_score + 0.05 * fine + 0.02 * global_score + 0.03 * duration_ratio
            else:
                score = 0.35 * geometry + 0.20 * local_score + 0.25 * fine + 0.15 * global_score + 0.05 * duration_ratio
            score += support
            ranked.append(
                {
                    "movie_index": movie_idx,
                    "movie_name": movie_item.path.name,
                    "score": float(score),
                    "geometry_score": geometry,
                    "geometry_inliers": inliers,
                    "good_matches": good_matches,
                    "local_score": local_score,
                    "global_score": global_score,
                    "fine_score": fine,
                    "sequence_support": support,
                    "movie_duration": float(movie_item.duration),
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    try:
        preliminary = [rank_candidates(ref_idx, {})[0] for ref_idx in range(len(ref_features))]
        weak_indices = [
            ref_idx
            for ref_idx, best in enumerate(preliminary)
            if best["geometry_inliers"] < 8
        ]
        if weak_indices:
            print(f"[match] ANN fallback for {len(weak_indices)} weak reference clips", flush=True)
            ann_data = build_local_ann(movie_features)
            if ann_data is not None:
                ann_index, descriptor_owners = ann_data
                for position, ref_idx in enumerate(weak_indices, 1):
                    if not ref_features[ref_idx].local_features:
                        continue
                    candidates = ann_local_candidates(
                        ref_features[ref_idx],
                        ann_index,
                        descriptor_owners,
                        len(movie_features),
                        max(candidate_count, 24),
                    )
                    candidate_sets[ref_idx].update(int(item) for item in candidates)
                    print(
                        f"[ann] {position}/{len(weak_indices)} {ref_features[ref_idx].path.name}",
                        flush=True,
                    )
                for ref_idx in weak_indices:
                    preliminary[ref_idx] = rank_candidates(ref_idx, {})[0]

        sequence_supports: list[dict[int, float]] = [dict() for _ in ref_features]
        for ref_idx in range(len(ref_features)):
            for neighbor_idx in range(
                max(0, ref_idx - neighbor_radius),
                min(len(ref_features), ref_idx + neighbor_radius + 1),
            ):
                if neighbor_idx == ref_idx or preliminary[neighbor_idx]["geometry_inliers"] < 10:
                    continue
                ref_distance = abs(neighbor_idx - ref_idx)
                center = preliminary[neighbor_idx]["movie_index"]
                for movie_offset in range(-2, 3):
                    movie_idx = center + movie_offset
                    if not 0 <= movie_idx < len(movie_features):
                        continue
                    candidate_sets[ref_idx].add(movie_idx)
                    bonus = (0.18 / ref_distance) * (1.0 - 0.2 * abs(movie_offset))
                    sequence_supports[ref_idx][movie_idx] = sequence_supports[ref_idx].get(movie_idx, 0.0) + bonus

        results = []
        for ref_idx, ref_item in enumerate(ref_features):
            ranked = rank_candidates(ref_idx, sequence_supports[ref_idx])
            best = ranked[0]
            results.append(
                {
                    "ref_index": ref_idx,
                    "ref_name": ref_item.path.name,
                    "movie_index": best["movie_index"],
                    "movie_name": best["movie_name"],
                    "score": best["score"],
                    "ref_duration": float(ref_item.duration),
                    "movie_duration": best["movie_duration"],
                    "status": "localized",
                    "geometry_inliers": best["geometry_inliers"],
                    "good_matches": best["good_matches"],
                    "top_candidates": ranked[:top_k],
                }
            )
            print(
                f"[verify] {ref_idx + 1}/{len(ref_features)} {ref_item.path.name} -> "
                f"{best['movie_name']} score={best['score']:.4f} inliers={best['geometry_inliers']}",
                flush=True,
            )
        return results
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def clear_directory_children(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def export_results(
    matches: list[dict],
    ref_features: list[ShotFeature],
    movie_features: list[ShotFeature],
    output_root: Path,
    low_score_threshold: float,
    min_geometry_inliers: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    pair_root = output_root / "pairs"
    low_conf_root = output_root / "low_confidence_pairs"
    clear_directory_children(pair_root)
    clear_directory_children(low_conf_root)

    manifest_path = output_root / "matches.csv"
    json_path = output_root / "matches.json"

    for match in matches:
        ref_item = ref_features[match["ref_index"]]
        movie_item = movie_features[match["movie_index"]]
        folder_name = (
            f"{match['ref_index'] + 1:04d}_"
            f"{ref_item.path.stem}__{movie_item.path.stem}__{match['score']:.4f}"
        )
        target_dir = pair_root / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref_item.path, target_dir / f"reference{ref_item.path.suffix.lower()}")
        shutil.copy2(movie_item.path, target_dir / f"movie{movie_item.path.suffix.lower()}")
        with (target_dir / "match.json").open("w", encoding="utf-8") as fh:
            json.dump(match, fh, ensure_ascii=False, indent=2)
        if (
            match["score"] < low_score_threshold
            or match["geometry_inliers"] < min_geometry_inliers
        ):
            low_target_dir = low_conf_root / folder_name
            low_target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ref_item.path, low_target_dir / f"reference{ref_item.path.suffix.lower()}")
            shutil.copy2(movie_item.path, low_target_dir / f"movie{movie_item.path.suffix.lower()}")
            with (low_target_dir / "match.json").open("w", encoding="utf-8") as fh:
                json.dump(match, fh, ensure_ascii=False, indent=2)

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "ref_index",
                "ref_name",
                "movie_index",
                "movie_name",
                "score",
                "ref_duration",
                "movie_duration",
                "status",
                "geometry_inliers",
                "good_matches",
                "top1_movie_index",
                "top1_movie_name",
                "top1_score",
                "top2_movie_index",
                "top2_movie_name",
                "top2_score",
                "top3_movie_index",
                "top3_movie_name",
                "top3_score",
            ],
        )
        writer.writeheader()
        for match in matches:
            row = {
                k: match[k]
                for k in [
                    "ref_index",
                    "ref_name",
                    "movie_index",
                    "movie_name",
                    "score",
                    "ref_duration",
                    "movie_duration",
                    "status",
                    "geometry_inliers",
                    "good_matches",
                ]
            }
            for candidate_idx in range(3):
                prefix = f"top{candidate_idx + 1}"
                if candidate_idx < len(match["top_candidates"]):
                    candidate = match["top_candidates"][candidate_idx]
                    row[f"{prefix}_movie_index"] = candidate["movie_index"]
                    row[f"{prefix}_movie_name"] = candidate["movie_name"]
                    row[f"{prefix}_score"] = candidate["score"]
                else:
                    row[f"{prefix}_movie_index"] = ""
                    row[f"{prefix}_movie_name"] = ""
                    row[f"{prefix}_score"] = ""
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(matches, fh, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match segmented reference shots to movie shots.")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/1_reference_analyzer/shot_clips"),
        help="Directory containing reference shot clips.",
    )
    parser.add_argument(
        "--movie-dir",
        type=Path,
        default=Path("data/3_movie_shot_parser/shot_clips"),
        help="Directory containing movie shot clips.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/shot_localization"),
        help="Output directory for paired clip folders and manifests.",
    )
    parser.add_argument("--sample-count", type=int, default=6, help="Frames sampled from each shot clip.")
    parser.add_argument("--frame-size", type=int, default=384, help="Square frame size used for descriptors.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for feature extraction and geometric verification.")
    parser.add_argument("--low-score-threshold", type=float, default=0.35, help="Localized results below this score are copied into low_confidence_pairs.")
    parser.add_argument("--min-geometry-inliers", type=int, default=20, help="Results below this geometric inlier count are copied into low_confidence_pairs.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidate movie shots to export per localized reference shot.")
    parser.add_argument("--candidate-count", type=int, default=30, help="Local-feature candidates retained for recall.")
    parser.add_argument(
        "--geometry-candidate-count",
        type=int,
        default=24,
        help="Recalled candidates per reference shot that receive ORB geometric verification.",
    )
    parser.add_argument("--neighbor-radius", type=int, default=2, help="Adjacent reference shots used to supplement weak candidate recall.")
    parser.add_argument("--reference-limit", type=int, default=0, help="Limit reference clips for diagnostics; 0 processes all clips.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference_paths = list_video_files(args.reference_dir)
    movie_paths = list_video_files(args.movie_dir)
    if args.reference_limit > 0:
        reference_paths = reference_paths[:args.reference_limit]
    if not reference_paths:
        raise RuntimeError(f"No reference clips found in {args.reference_dir}")
    if not movie_paths:
        raise RuntimeError(f"No movie clips found in {args.movie_dir}")

    print(f"[scan] reference clips: {len(reference_paths)}", flush=True)
    print(f"[scan] movie clips: {len(movie_paths)}", flush=True)

    ref_features = extract_all_features(
        reference_paths,
        args.sample_count,
        args.frame_size,
        args.workers,
        mask_text_bands=True,
    )
    movie_features = extract_all_features(
        movie_paths,
        args.sample_count,
        args.frame_size,
        args.workers,
        mask_text_bands=False,
    )

    print("[match] building global similarity matrix", flush=True)
    similarity = cosine_matrix(ref_features, movie_features)
    matches = independent_localize(
        ref_features,
        movie_features,
        similarity,
        args.neighbor_radius,
        args.candidate_count,
        args.top_k,
        geometry_candidate_count=args.geometry_candidate_count,
        workers=args.workers,
    )

    print(f"[export] writing results to {args.output_dir}", flush=True)
    export_results(
        matches,
        ref_features,
        movie_features,
        args.output_dir,
        args.low_score_threshold,
        args.min_geometry_inliers,
    )
    print("[done] shot localization finished", flush=True)


if __name__ == "__main__":
    main()
