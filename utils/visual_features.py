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


def select_keyframes(item: dict[str, Any], max_frames: int = 3) -> list[str]:
    frames = item.get("keyframes")
    if not isinstance(frames, list) or not frames:
        return []

    cleaned = [str(frame) for frame in frames if frame]
    if not cleaned:
        return []

    max_frames = max(1, int(max_frames))
    if len(cleaned) <= max_frames:
        return _unique_keep_order(cleaned)

    if max_frames == 1:
        mid = len(cleaned) // 2
        return [cleaned[mid]]

    picks = [0, len(cleaned) // 2, len(cleaned) - 1]
    if max_frames > 3:
        step = (len(cleaned) - 1) / float(max_frames - 1)
        picks = [round(i * step) for i in range(max_frames)]
    return _unique_keep_order([cleaned[int(i)] for i in picks])


def _frame_role(index: int, total: int) -> str:
    if total <= 1:
        return "middle"
    if index == 0:
        return "start"
    if index == total - 1:
        return "end"
    return "middle"


def build_shot_feature(
    item: dict[str, Any],
    *,
    max_frames: int = 3,
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
