from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .project_paths import resolve_existing_path


def _read_image(path: str | Path) -> np.ndarray | None:
    img = cv2.imread(str(resolve_existing_path(path)), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def crop_black_borders(img: np.ndarray, *, threshold: int = 10, min_content_ratio: float = 0.35) -> np.ndarray:
    if img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    coords = cv2.findNonZero(mask.astype("uint8"))
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    if w * h < img.shape[0] * img.shape[1] * min_content_ratio:
        return img
    return img[y : y + h, x : x + w]


def normalize_frame_image(
    img: np.ndarray,
    *,
    target_width: int = 854,
    crop_borders: bool = True,
    equalize_luma: bool = True,
) -> np.ndarray:
    if crop_borders:
        img = crop_black_borders(img)
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img
    if w > target_width:
        scale = target_width / float(w)
        img = cv2.resize(img, (target_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    if equalize_luma:
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return img


def load_normalized_frame(
    path: str | Path,
    *,
    spatial_normalize: str = "auto",
    target_width: int = 854,
) -> np.ndarray | None:
    img = _read_image(path)
    if img is None:
        return None
    return normalize_frame_image(
        img,
        target_width=target_width,
        crop_borders=spatial_normalize != "off",
        equalize_luma=spatial_normalize != "off",
    )


def _resize_pair(a: np.ndarray, b: np.ndarray, width: int = 320) -> tuple[np.ndarray, np.ndarray]:
    def resize(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return img
        scale = width / float(w)
        return cv2.resize(img, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

    a2 = resize(a)
    b2 = resize(b)
    height = min(a2.shape[0], b2.shape[0])
    if height <= 0:
        return a2, b2
    return a2[:height, :width], b2[:height, :width]


def _weighted_mean(diff: np.ndarray, *, subtitle_mask_ratio: float = 0.18) -> float:
    if subtitle_mask_ratio <= 0:
        return float(np.mean(diff))
    weights = np.ones(diff.shape[:2], dtype=np.float32)
    masked_rows = int(round(weights.shape[0] * min(0.45, max(0.0, subtitle_mask_ratio))))
    if masked_rows > 0:
        weights[-masked_rows:, :] = 0.35
    return float(np.sum(diff.astype("float32") * weights) / max(1e-6, float(np.sum(weights))))


def compare_normalized_frames(
    ref_img: np.ndarray,
    movie_img: np.ndarray,
    *,
    subtitle_mask_ratio: float = 0.18,
) -> dict[str, float]:
    ref, movie = _resize_pair(ref_img, movie_img)
    if ref.size == 0 or movie.size == 0:
        return {"score": 0.0, "gray": 0.0, "edge": 0.0, "hist": 0.0, "ssim": 0.0}

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    movie_gray = cv2.cvtColor(movie, cv2.COLOR_BGR2GRAY)
    gray_diff = _weighted_mean(cv2.absdiff(ref_gray, movie_gray), subtitle_mask_ratio=subtitle_mask_ratio) / 255.0
    gray_score = 1.0 - gray_diff

    ref_edge = cv2.Canny(ref_gray, 80, 160)
    movie_edge = cv2.Canny(movie_gray, 80, 160)
    edge_diff = _weighted_mean(cv2.absdiff(ref_edge, movie_edge), subtitle_mask_ratio=subtitle_mask_ratio) / 255.0
    edge_score = 1.0 - edge_diff

    ref_hist = _hist(ref)
    movie_hist = _hist(movie)
    hist_score = float(cv2.compareHist(ref_hist.astype("float32"), movie_hist.astype("float32"), cv2.HISTCMP_CORREL))
    hist_score = _clip01((hist_score + 1.0) / 2.0)

    ref_f = ref_gray.astype("float32")
    movie_f = movie_gray.astype("float32")
    mu_ref = float(np.mean(ref_f))
    mu_movie = float(np.mean(movie_f))
    var_ref = float(np.var(ref_f))
    var_movie = float(np.var(movie_f))
    cov = float(np.mean((ref_f - mu_ref) * (movie_f - mu_movie)))
    c1 = 6.5025
    c2 = 58.5225
    ssim = ((2 * mu_ref * mu_movie + c1) * (2 * cov + c2)) / ((mu_ref**2 + mu_movie**2 + c1) * (var_ref + var_movie + c2))
    ssim_score = _clip01(ssim)

    score = gray_score * 0.34 + edge_score * 0.26 + hist_score * 0.18 + ssim_score * 0.22
    return {
        "score": round(_clip01(score), 4),
        "gray": round(_clip01(gray_score), 4),
        "edge": round(_clip01(edge_score), 4),
        "hist": round(_clip01(hist_score), 4),
        "ssim": round(_clip01(ssim_score), 4),
    }


def compare_homography_frames(
    ref_img: np.ndarray,
    movie_img: np.ndarray,
    *,
    subtitle_mask_ratio: float = 0.18,
) -> dict[str, Any]:
    ref, movie = _resize_pair(ref_img, movie_img)
    if ref.size == 0 or movie.size == 0:
        return {"score": 0.0, "transform": "failed", "inliers": 0, "detail": {}}

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    movie_gray = cv2.cvtColor(movie, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create(nfeatures=800) if hasattr(cv2, "SIFT_create") else cv2.ORB_create(nfeatures=900)
    kp_movie, desc_movie = detector.detectAndCompute(movie_gray, None)
    kp_ref, desc_ref = detector.detectAndCompute(ref_gray, None)
    if desc_movie is None or desc_ref is None or len(kp_movie) < 12 or len(kp_ref) < 12:
        detail = compare_normalized_frames(ref, movie, subtitle_mask_ratio=subtitle_mask_ratio)
        return {"score": detail["score"], "transform": "failed", "inliers": 0, "detail": detail}

    if desc_movie.dtype == np.float32:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(desc_movie, desc_ref, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 10:
        detail = compare_normalized_frames(ref, movie, subtitle_mask_ratio=subtitle_mask_ratio)
        return {"score": detail["score"], "transform": "failed", "inliers": len(good), "detail": detail}

    src_pts = np.float32([kp_movie[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if matrix is None or inlier_mask is None:
        detail = compare_normalized_frames(ref, movie, subtitle_mask_ratio=subtitle_mask_ratio)
        return {"score": detail["score"], "transform": "failed", "inliers": 0, "detail": detail}

    warped = cv2.warpPerspective(movie, matrix, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    detail = compare_normalized_frames(ref, warped, subtitle_mask_ratio=subtitle_mask_ratio)
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / max(1, len(good))
    score = _clip01(float(detail["score"]) * 0.85 + min(1.0, inlier_ratio) * 0.15)
    return {"score": round(score, 4), "transform": "homography", "inliers": inliers, "detail": detail}


def _dhash(gray: np.ndarray, size: int = 8) -> int:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def _hash_similarity(a: int, b: int, bits: int = 64) -> float:
    dist = bin(int(a ^ b)).count("1")
    return max(0.0, 1.0 - dist / float(bits))


def _hist(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [24, 16, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _orb_desc(img: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=600)
    _kp, desc = orb.detectAndCompute(gray, None)
    return desc


def build_frame_feature(path: str | Path) -> dict[str, Any]:
    img = _read_image(path)
    if img is None:
        return {"path": str(path), "ok": False}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "path": str(path),
        "ok": True,
        "dhash": _dhash(gray),
        "hist": _hist(img),
        "orb": _orb_desc(img),
    }


def compare_features(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    if not a.get("ok") or not b.get("ok"):
        return {"score": 0.0, "hash": 0.0, "hist": 0.0, "orb": 0.0}

    hash_score = _hash_similarity(int(a["dhash"]), int(b["dhash"]))
    hist_score = float(cv2.compareHist(a["hist"].astype("float32"), b["hist"].astype("float32"), cv2.HISTCMP_CORREL))
    hist_score = max(0.0, min(1.0, (hist_score + 1.0) / 2.0))

    orb_score = 0.0
    da = a.get("orb")
    db = b.get("orb")
    if da is not None and db is not None and len(da) >= 4 and len(db) >= 4:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(da, db)
        if matches:
            good = [m for m in matches if m.distance <= 64]
            orb_score = min(1.0, len(good) / max(12.0, min(len(da), len(db)) * 0.35))

    score = hash_score * 0.45 + hist_score * 0.35 + orb_score * 0.20
    return {
        "score": round(float(score), 4),
        "hash": round(float(hash_score), 4),
        "hist": round(float(hist_score), 4),
        "orb": round(float(orb_score), 4),
    }


def compare_features_lightweight(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    if not a.get("ok") or not b.get("ok"):
        return {"score": 0.0, "hash": 0.0, "hist": 0.0, "orb": 0.0}

    hash_score = _hash_similarity(int(a["dhash"]), int(b["dhash"]))
    hist_score = float(cv2.compareHist(a["hist"].astype("float32"), b["hist"].astype("float32"), cv2.HISTCMP_CORREL))
    hist_score = max(0.0, min(1.0, (hist_score + 1.0) / 2.0))
    score = hash_score * 0.55 + hist_score * 0.45
    return {
        "score": round(float(score), 4),
        "hash": round(float(hash_score), 4),
        "hist": round(float(hist_score), 4),
        "orb": 0.0,
    }


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def interior_sample_positions(count: int) -> list[float]:
    """在 (0, 1) 内采样，避开镜头首尾。

    count=4 时把镜头均分为 5 段，取 1/5、2/5、3/5、4/5 位置。
    """
    count = max(1, int(count))
    if count == 1:
        return [0.5]
    segments = count + 1
    return [i / segments for i in range(1, segments)]


def _index_at_interior_position(length: int, position: float) -> int:
    if length <= 1:
        return 0
    if length == 2:
        return 0
    raw = int(round(position * (length - 1)))
    return max(1, min(length - 2, raw))


def select_keyframes(item: dict[str, Any], max_frames: int = 4) -> list[str]:
    frames = item.get("keyframes")
    if not isinstance(frames, list) or not frames:
        return []

    cleaned = [str(frame) for frame in frames if frame]
    if not cleaned:
        return []

    max_frames = max(1, int(max_frames))
    if len(cleaned) <= 2:
        return _unique_keep_order(cleaned)
    if len(cleaned) == max_frames:
        return _unique_keep_order(cleaned)

    positions = interior_sample_positions(max_frames)
    picks: list[int] = []
    seen_indexes: set[int] = set()
    for position in positions:
        index = _index_at_interior_position(len(cleaned), position)
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        picks.append(index)
    return _unique_keep_order([cleaned[index] for index in picks])


def _frame_role(index: int, total: int) -> str:
    if total <= 1:
        return "middle"
    if total == 4:
        return f"fifth_{index + 1}"
    return "middle"


def build_shot_feature(
    item: dict[str, Any],
    *,
    max_frames: int = 4,
    feature_mode: str = "classic",
) -> dict[str, Any]:
    frames = select_keyframes(item, max_frames=max_frames)
    features = []
    for index, path in enumerate(frames):
        features.append(
            {
                "frame_role": _frame_role(index, len(frames)),
                "feature": build_frame_feature(path),
            }
        )

    return {
        "shot_id": str(item.get("movie_shot_id") or item.get("ref_shot_id") or item.get("shot_id") or ""),
        "start": float(item.get("start") or 0.0),
        "end": float(item.get("end") or 0.0),
        "keyframes": frames,
        "features": features,
        "feature_mode": feature_mode,
        "ok": any(row["feature"].get("ok") for row in features),
    }


def _lightweight_from_detail(detail: dict[str, float]) -> float:
    return detail["hash"] * 0.55 + detail["hist"] * 0.45


def _aggregate_pair_scores(rows: list[dict[str, Any]], score_key: str) -> float:
    if not rows:
        return 0.0
    ordered = sorted((float(row[score_key]) for row in rows), reverse=True)
    top_score = ordered[0]
    top3_avg = sum(ordered[:3]) / min(3, len(ordered))
    return top_score * 0.6 + top3_avg * 0.4


def compare_shot_features(a: dict[str, Any], b: dict[str, Any], *, include_orb: bool = True) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    for ref_frame in a.get("features") or []:
        for movie_frame in b.get("features") or []:
            detail = (
                compare_features(ref_frame.get("feature") or {}, movie_frame.get("feature") or {})
                if include_orb
                else compare_features_lightweight(ref_frame.get("feature") or {}, movie_frame.get("feature") or {})
            )
            pair_rows.append(
                {
                    "ref_frame_role": ref_frame.get("frame_role") or "",
                    "ref_frame_path": (ref_frame.get("feature") or {}).get("path"),
                    "movie_frame_role": movie_frame.get("frame_role") or "",
                    "movie_frame_path": (movie_frame.get("feature") or {}).get("path"),
                    "score": detail["score"],
                    "lightweight_score": round(float(detail["score"] if not include_orb else _lightweight_from_detail(detail)), 4),
                    "detail": detail,
                }
            )

    if not pair_rows:
        return {
            "score": 0.0,
            "lightweight_score": 0.0,
            "hash": 0.0,
            "hist": 0.0,
            "orb": 0.0,
            "best_pair": None,
            "top_pairs": [],
        }

    score = _aggregate_pair_scores(pair_rows, "score")
    lightweight_score = _aggregate_pair_scores(pair_rows, "lightweight_score")
    top_pairs = sorted(pair_rows, key=lambda row: row["score"], reverse=True)[:3]
    best_detail = top_pairs[0]["detail"]
    return {
        "score": round(float(score), 4),
        "lightweight_score": round(float(lightweight_score), 4),
        "hash": best_detail["hash"],
        "hist": best_detail["hist"],
        "orb": best_detail["orb"],
        "best_pair": {
            "ref_frame_role": top_pairs[0]["ref_frame_role"],
            "movie_frame_role": top_pairs[0]["movie_frame_role"],
            "ref_frame_path": top_pairs[0]["ref_frame_path"],
            "movie_frame_path": top_pairs[0]["movie_frame_path"],
            "score": top_pairs[0]["score"],
        },
        "top_pairs": [
            {
                "ref_frame_role": row["ref_frame_role"],
                "movie_frame_role": row["movie_frame_role"],
                "score": row["score"],
                "detail": row["detail"],
            }
            for row in top_pairs
        ],
    }


def first_keyframe(item: dict[str, Any]) -> str | None:
    frames = select_keyframes(item, max_frames=1)
    return frames[0] if frames else None
