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


def first_keyframe(item: dict[str, Any]) -> str | None:
    frames = item.get("keyframes")
    if isinstance(frames, list) and frames:
        return str(frames[0])
    return None
