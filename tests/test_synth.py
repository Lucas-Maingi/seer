"""Synthetic engine invariants: geometry ground truth must be exact."""

import random

import cv2
import numpy as np

from seer.synth.compose import compose_scene
from seer.synth.face import ProceduralFaces
from seer.synth.identity import sample_persona
from seer.synth.tamper import apply_random_attack
from seer.synth.template import render_national_id, render_passport


def _sample(seed=0, passport=False):
    rng = random.Random(seed)
    persona = sample_persona(rng)
    portrait = ProceduralFaces().sample(rng)
    render = (render_passport if passport else render_national_id)(persona, portrait, rng)
    return rng, render


def test_corner_ground_truth_matches_homography():
    rng, render = _sample(1)
    scene = compose_scene(render, rng)
    w, h = render.image.size
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, scene.homography).reshape(4, 2)
    assert np.allclose(projected, scene.corners, atol=1e-3)


def test_corners_inside_scene():
    for seed in range(5):
        rng, render = _sample(seed, passport=seed % 2 == 0)
        scene = compose_scene(render, rng)
        h, w = scene.image.shape[:2]
        assert (scene.corners[:, 0] >= 0).all() and (scene.corners[:, 0] <= w).all()
        assert (scene.corners[:, 1] >= 0).all() and (scene.corners[:, 1] <= h).all()


def test_field_quads_are_inside_canonical_image():
    _, render = _sample(2, passport=True)
    w, h = render.image.size
    for f in render.fields:
        q = np.array(f.quad)
        assert (q[:, 0] >= 0).all() and (q[:, 0] <= w).all(), f.name
        assert (q[:, 1] >= 0).all() and (q[:, 1] <= h).all(), f.name


def test_passport_mrz_is_valid():
    from seer.icao9303 import parse_td3
    _, render = _sample(3, passport=True)
    assert render.mrz is not None
    assert parse_td3(*render.mrz).all_valid


def test_tamper_produces_nonempty_mask_and_changed_pixels():
    rng, render = _sample(4)
    impostor = ProceduralFaces().sample(rng)
    before = np.asarray(render.image).copy()
    result = apply_random_attack(render, impostor, rng)
    after = np.asarray(result.render.image)
    assert result.mask.max() == 255
    changed = (before != after).any(axis=2)
    # manipulated pixels must lie inside the declared mask
    assert changed[result.mask == 0].mean() < 0.01
    assert changed[result.mask == 255].mean() > 0.01
