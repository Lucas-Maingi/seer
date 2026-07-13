"""Seer verification API.

    uvicorn seer.api.main:app --host 0.0.0.0 --port 8000

POST /verify      multipart: document (required image), selfie (optional
                  image), fmr (query, default 1e-3) -> VerifyResponse
GET  /health      readiness + which stages are loaded
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from seer.api.schemas import (
    DocumentOut, FaceOut, FieldOut, ForensicsOut, HealthResponse, VerifyResponse,
)
from seer.runtime.pipeline import SeerPipeline, VerificationReport

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
_WEIGHTS_DIR = os.environ.get("SEER_WEIGHTS", "weights")

pipeline: SeerPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = SeerPipeline(weights=_WEIGHTS_DIR)
    yield


app = FastAPI(title="Seer KYC verification", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SEER_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"], allow_headers=["*"],
)


async def _decode_upload(upload: UploadFile, name: str) -> np.ndarray:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{name} exceeds {MAX_UPLOAD_BYTES // 2**20} MB limit")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{name} is not a decodable image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_response(r: VerificationReport) -> VerifyResponse:
    doc = None
    if r.document is not None:
        doc = DocumentOut(
            kind=r.document.kind,
            fields={k: FieldOut(text=f.text, confidence=f.confidence,
                                format_valid=f.format_valid)
                    for k, f in r.document.fields.items()},
            mean_confidence=r.document.mean_confidence,
            mrz_valid=r.document.mrz_valid,
            mrz_checks=r.document.mrz_checks,
            viz_mrz_consistency=r.document.viz_mrz_consistency,
        )
    face = None
    if r.face_similarity is not None:
        face = FaceOut(similarity=r.face_similarity, threshold=r.face_threshold,
                       fmr_level="1e-3", match=bool(r.face_match))
    forensics = None
    if r.tamper_probability is not None:
        forensics = ForensicsOut(tamper_probability=r.tamper_probability,
                                 heatmap=r.tamper_heatmap)
    return VerifyResponse(
        verdict=r.verdict, reasons=r.reasons, corners=r.corners,
        document=doc, face=face, forensics=forensics,
        timings_ms={k: round(v, 2) for k, v in r.timings_ms.items()},
        stages_available=r.stages_available,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    stages = ({} if pipeline is None else {
        "localize": pipeline.corner is not None,
        "ocr": pipeline.crnn is not None,
        "forensics": pipeline.tamper is not None,
        "face": pipeline.face is not None and pipeline.calibration is not None,
    })
    ready = bool(stages.get("localize")) and bool(stages.get("ocr"))
    return HealthResponse(status="ok" if ready else "degraded",
                          ready=ready, stages_available=stages)


@app.post("/verify", response_model=VerifyResponse)
async def verify(
    document: UploadFile = File(..., description="photo of the ID/passport"),
    selfie: UploadFile | None = File(None, description="live selfie"),
    fmr: str = Query("1e-3", pattern=r"^1e-[234]$"),
) -> VerifyResponse:
    if pipeline is None or pipeline.corner is None or pipeline.crnn is None:
        raise HTTPException(503, "pipeline not ready: run training/export first "
                                 "(see /health for loaded stages)")
    doc_img = await _decode_upload(document, "document")
    selfie_img = await _decode_upload(selfie, "selfie") if selfie else None
    pipeline.fmr_level = fmr
    report = pipeline.verify(doc_img, selfie_img)
    return _to_response(report)
