"""API contract tests (no trained weights required)."""

import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from seer.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager triggers lifespan
        yield c


def _jpeg_bytes() -> bytes:
    import cv2
    img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health_reports_stages(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body["stages_available"]) == {"localize", "ocr", "forensics", "face"}
    # without trained weights the service must say so, not pretend
    if not body["ready"]:
        assert body["status"] == "degraded"


def test_verify_rejects_when_not_ready_or_accepts_image(client):
    ready = client.get("/health").json()["ready"]
    r = client.post("/verify", files={"document": ("d.jpg", io.BytesIO(_jpeg_bytes()),
                                                   "image/jpeg")})
    if ready:
        assert r.status_code == 200
        assert r.json()["verdict"] in ("pass", "review", "fail")
    else:
        assert r.status_code == 503


def test_verify_rejects_non_image(client):
    ready = client.get("/health").json()["ready"]
    if not ready:
        pytest.skip("pipeline not loaded; decode path unreachable")
    r = client.post("/verify", files={"document": ("d.jpg", io.BytesIO(b"not an image"),
                                                   "image/jpeg")})
    assert r.status_code == 400


def test_verify_rejects_bad_fmr(client):
    r = client.post("/verify?fmr=0.5",
                    files={"document": ("d.jpg", io.BytesIO(_jpeg_bytes()),
                                        "image/jpeg")})
    assert r.status_code == 422
