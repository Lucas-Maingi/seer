"""Face detection, alignment and ArcFace embedding.

Model files (fetched by ``scripts/fetch_models.py``, never committed):

- YuNet face detector (OpenCV Zoo) — boxes + 5 landmarks, CPU-fast
- ArcFace recognition model (InsightFace buffalo_l ``w600k_r50.onnx``)

Alignment is the part people get wrong: ArcFace embeddings are only
meaningful if the input is warped so the five facial landmarks land on the
model's canonical template coordinates. We estimate the 4-DoF similarity
transform (rotation, uniform scale, translation) landmark→template and warp
with it. Feeding an unaligned crop quietly costs 5-15 points of accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ARCFACE_SIZE = 112

# canonical 5-point template (lefteye, righteye, nose, leftmouth, rightmouth)
# for 112x112 ArcFace input — from the InsightFace reference implementation
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


@dataclass
class DetectedFace:
    box: tuple[float, float, float, float]  # x, y, w, h
    landmarks: np.ndarray                   # 5x2
    score: float


class YuNetDetector:
    def __init__(self, model_path: str | Path, score_threshold: float = 0.7):
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, rgb: np.ndarray) -> list[DetectedFace]:
        h, w = rgb.shape[:2]
        self._detector.setInputSize((w, h))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, faces = self._detector.detect(bgr)
        out: list[DetectedFace] = []
        if faces is None:
            return out
        for f in faces:
            out.append(DetectedFace(
                box=(float(f[0]), float(f[1]), float(f[2]), float(f[3])),
                landmarks=f[4:14].reshape(5, 2).astype(np.float32),
                score=float(f[14]),
            ))
        out.sort(key=lambda d: d.box[2] * d.box[3], reverse=True)  # largest first
        return out


def align_face(rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Warp so the 5 landmarks land on the ArcFace template (112x112)."""
    M, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:  # degenerate landmarks; fall back to a center crop resize
        return cv2.resize(rgb, (ARCFACE_SIZE, ARCFACE_SIZE))
    return cv2.warpAffine(rgb, M, (ARCFACE_SIZE, ARCFACE_SIZE),
                          flags=cv2.INTER_LINEAR, borderValue=0)


class ArcFaceEmbedder:
    """ONNX Runtime wrapper producing L2-normalized 512-d embeddings."""

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def embed_aligned(self, aligned_rgb: np.ndarray) -> np.ndarray:
        """aligned_rgb: (112,112,3) uint8 already template-aligned."""
        x = aligned_rgb.astype(np.float32)
        x = (x - 127.5) / 127.5                       # ArcFace normalization
        x = x.transpose(2, 0, 1)[None]                # (1,3,112,112)
        emb = self.session.run(None, {self.input_name: x})[0][0]
        return emb / max(np.linalg.norm(emb), 1e-10)


class FacePipeline:
    """detect → align → embed, returning None when no face is found."""

    def __init__(self, detector: YuNetDetector, embedder: ArcFaceEmbedder):
        self.detector = detector
        self.embedder = embedder

    def embed(self, rgb: np.ndarray) -> tuple[np.ndarray, DetectedFace] | None:
        faces = self.detector.detect(rgb)
        if not faces:
            return None
        face = faces[0]
        aligned = align_face(rgb, face.landmarks)
        return self.embedder.embed_aligned(aligned), face
