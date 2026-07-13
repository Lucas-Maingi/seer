"""Face stage invariants: alignment geometry and calibration math.

Model-dependent paths (YuNet, ArcFace) are exercised only when the weights
exist locally; the math must hold regardless.
"""

import numpy as np
import pytest

from seer.face.embed import ARCFACE_SIZE, ARCFACE_TEMPLATE, align_face
from seer.face.verify import Calibration, calibrate, cosine, decide


def test_cosine_bounds_and_identity():
    rng = np.random.RandomState(0)
    a = rng.randn(512)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, -a) == pytest.approx(-1.0)
    b = rng.randn(512)
    assert -1.0 <= cosine(a, b) <= 1.0


def test_align_face_maps_landmarks_to_template():
    # build an image whose landmarks are a rotated/scaled/shifted template
    rng = np.random.RandomState(1)
    theta, scale = 0.3, 2.1
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]) * scale
    shift = np.array([80.0, 60.0])
    landmarks = (ARCFACE_TEMPLATE @ R.T + shift).astype(np.float32)

    img = (rng.rand(400, 400, 3) * 255).astype(np.uint8)
    aligned = align_face(img, landmarks)
    assert aligned.shape == (ARCFACE_SIZE, ARCFACE_SIZE, 3)

    # the recovered warp must send the synthetic landmarks onto the template
    import cv2
    M, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    mapped = landmarks @ M[:, :2].T + M[:, 2]
    assert np.abs(mapped - ARCFACE_TEMPLATE).max() < 0.5


def test_calibration_hits_target_fmr():
    rng = np.random.RandomState(2)
    genuine = rng.normal(0.62, 0.10, 20000)
    impostor = rng.normal(0.05, 0.10, 200000)
    cal = calibrate(genuine, impostor)
    for key, f in (("1e-2", 1e-2), ("1e-3", 1e-3)):
        thr = cal.thresholds[key]
        measured_fmr = (impostor >= thr).mean()
        assert measured_fmr == pytest.approx(f, rel=0.25)
    # stricter FMR must mean a higher threshold and lower TPR
    assert cal.thresholds["1e-4"] > cal.thresholds["1e-2"]
    assert cal.tpr_at_fmr["1e-4"] <= cal.tpr_at_fmr["1e-2"]


def test_decide_uses_requested_operating_point():
    cal = Calibration(thresholds={"1e-2": 0.3, "1e-3": 0.45},
                      tpr_at_fmr={"1e-2": 0.99, "1e-3": 0.97},
                      n_genuine=10, n_impostor=10)
    assert decide(0.40, cal, "1e-2").match
    assert not decide(0.40, cal, "1e-3").match


def test_calibration_roundtrip(tmp_path):
    cal = calibrate(np.array([0.9, 0.8, 0.7]), np.linspace(-0.2, 0.4, 1000))
    p = tmp_path / "cal.json"
    cal.save(p)
    loaded = Calibration.load(p)
    assert loaded.thresholds == cal.thresholds
    assert loaded.n_impostor == 1000
