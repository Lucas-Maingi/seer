"""Corner post-processing and homography rectification."""

from __future__ import annotations

import cv2
import numpy as np

from seer.synth.template import ID_SIZE, PASSPORT_SIZE

# aspect = w/h; midpoint between card (1.585) and passport page (1.42)
_ID_ASPECT = ID_SIZE[0] / ID_SIZE[1]
_PASSPORT_ASPECT = PASSPORT_SIZE[0] / PASSPORT_SIZE[1]


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order an unordered 4-point set as TL, TR, BR, BL.

    Uses the sum/diff heuristic which is rotation-tolerant up to ~45°;
    the model is trained to emit ordered corners already, so this is a
    safety net for external corner sources (e.g. classical fallbacks).
    """
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.stack([
        pts[np.argmin(s)],   # TL: smallest x+y
        pts[np.argmin(d)],   # TR: smallest y-x
        pts[np.argmax(s)],   # BR: largest x+y
        pts[np.argmax(d)],   # BL: largest y-x
    ])


def estimate_aspect(corners: np.ndarray) -> float:
    """Perspective-robust aspect estimate from side lengths."""
    c = corners
    top = np.linalg.norm(c[1] - c[0])
    bottom = np.linalg.norm(c[2] - c[3])
    left = np.linalg.norm(c[3] - c[0])
    right = np.linalg.norm(c[2] - c[1])
    return ((top + bottom) / 2) / max((left + right) / 2, 1e-6)


def classify_kind(corners: np.ndarray) -> str:
    a = estimate_aspect(corners)
    mid = (_ID_ASPECT + _PASSPORT_ASPECT) / 2
    return "national_id" if a >= mid else "passport"


def rectify(image: np.ndarray, corners: np.ndarray,
            kind: str | None = None) -> tuple[np.ndarray, str, np.ndarray]:
    """Warp the document quad to its canonical fronto-parallel frame.

    image: HxWx3 RGB. corners: 4x2 TL,TR,BR,BL in image pixels.
    Returns (canonical image, kind, homography image->canonical).
    """
    corners = order_corners(corners)
    if kind is None:
        kind = classify_kind(corners)
    w, h = ID_SIZE if kind == "national_id" else PASSPORT_SIZE
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    out = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_CUBIC)
    return out, kind, H


def quad_iou(a: np.ndarray, b: np.ndarray, canvas: tuple[int, int]) -> float:
    """IoU of two quads by rasterization — the localization eval metric."""
    w, h = canvas
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.round(a).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(b).astype(np.int32)], 1)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter) / max(float(union), 1.0)
