"""End-to-end verification pipeline over ONNX Runtime.

Loads whatever models exist under weights/ (INT8 variants preferred when
present), degrades gracefully when a stage is missing, and returns a
structured report: per-stage results, per-stage wall time, and a fused
verdict with human-readable reasons.

Fusion policy (transparent rules, not a black box):
- FAIL    face similarity below the calibrated threshold, or an MRZ check
          digit fails (numbers that disagree with their own checksums are
          not OCR noise at high confidence — they are edits)
- REVIEW  forensic tamper probability above threshold, invalid field
          formats, VIZ/MRZ disagreement, or a missing face
- PASS    everything above holds together
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from seer.forensics.model import INPUT_SIZE as TAMPER_WH
from seer.forensics.signals import forensic_stack
from seer.localize.model import INPUT_SIZE as CORNER_SIZE
from seer.localize.rectify import rectify
from seer.ocr.charset import BLANK, decode_greedy
from seer.ocr.crnn import prepare_line
from seer.ocr.reader import DocumentReader, DocumentReadResult
from seer.face.embed import ArcFaceEmbedder, FacePipeline, YuNetDetector
from seer.face.verify import Calibration, cosine, decide

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
TAMPER_REVIEW_THRESHOLD = 0.5


def _session(path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0  # let ORT pick physical-core parallelism
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


# Stages whose INT8 variant is rejected at serve time, based on measured
# regressions on real CPU hardware (see runs/quantization_notes.md):
#
# - crnn:   the LSTM is dynamically quantized, which on CPUs without
#           AVX512-VNNI (Skylake and most laptops) runs the int8 matmuls
#           through a slow emulated path (~14x slower than fp32) and garbles
#           batched output.
# - corner: static quantization of the DSNT heatmap head collapses accuracy
#           (measured mean corner error ~260px vs ~2px fp32) — the spatial
#           softmax is too sensitive to activation quantization, and dead
#           heatmaps decode to the image center.
#
# Only the tamper conv net quantizes cleanly (AUC 0.62 int8 vs 0.63 fp32),
# so it keeps int8. This is a per-model precision decision, not a blanket one.
_NO_INT8 = {"crnn", "corner"}


def _pick(weights: Path, name: str, prefer_int8: bool) -> Path | None:
    int8, fp32 = weights / f"{name}.int8.onnx", weights / f"{name}.onnx"
    if prefer_int8 and name not in _NO_INT8 and int8.exists():
        return int8
    return fp32 if fp32.exists() else (int8 if int8.exists() else None)


@dataclass
class VerificationReport:
    verdict: str                      # "pass" | "review" | "fail"
    reasons: list[str]
    kind: str | None = None
    corners: list[list[float]] | None = None
    document: DocumentReadResult | None = None
    face_similarity: float | None = None
    face_match: bool | None = None
    face_threshold: float | None = None
    tamper_probability: float | None = None
    tamper_heatmap: list[list[float]] | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    stages_available: dict[str, bool] = field(default_factory=dict)


class SeerPipeline:
    def __init__(self, weights: str | Path = "weights", prefer_int8: bool = True,
                 fmr_level: str = "1e-3"):
        weights = Path(weights)
        self.fmr_level = fmr_level

        corner_path = _pick(weights, "corner", prefer_int8)
        self.corner = _session(corner_path) if corner_path else None

        crnn_path = _pick(weights, "crnn", prefer_int8)
        self.crnn = _session(crnn_path) if crnn_path else None
        self.reader = DocumentReader(self._recognize) if self.crnn else None

        tamper_path = _pick(weights, "tamper", prefer_int8)
        self.tamper = _session(tamper_path) if tamper_path else None

        yunet = weights / "face_detection_yunet_2023mar.onnx"
        arcface = weights / "arcface_w600k_r50.onnx"
        self.face = (FacePipeline(YuNetDetector(yunet), ArcFaceEmbedder(arcface))
                     if yunet.exists() and arcface.exists() else None)
        cal_path = weights / "face_calibration.json"
        self.calibration = Calibration.load(cal_path) if cal_path.exists() else None

    # ------------------------------------------------------------- backends

    def _recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        batch = np.stack([prepare_line(c) for c in crops])[:, None].astype(np.float32)
        (logp,) = self.crnn.run(None, {"input": batch})
        probs = np.exp(logp).transpose(1, 0, 2)  # (B, T, C)
        out = []
        for p in probs:
            idx = p.argmax(axis=-1)
            text = decode_greedy(idx.tolist())
            nonblank = idx != BLANK
            conf = float(p.max(axis=-1)[nonblank].mean()) if nonblank.any() else 0.0
            out.append((text, conf))
        return out

    def locate(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (CORNER_SIZE, CORNER_SIZE)).astype(np.float32) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        coords, _ = self.corner.run(None, {"input": x})
        c = coords[0]  # (4,2) in [-1,1]
        px = np.empty_like(c)
        px[:, 0] = ((c[:, 0] + 1) * w - 1) / 2
        px[:, 1] = ((c[:, 1] + 1) * h - 1) / 2
        return px

    def inspect(self, canonical_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        x = forensic_stack(canonical_rgb, TAMPER_WH)[None]
        doc_logit, hm = self.tamper.run(None, {"input": x})
        prob = float(1 / (1 + np.exp(-doc_logit[0])))
        heat = 1 / (1 + np.exp(-hm[0, 0]))
        return prob, heat

    # ----------------------------------------------------------------- main

    def verify(self, document_rgb: np.ndarray,
               selfie_rgb: np.ndarray | None = None) -> VerificationReport:
        report = VerificationReport(verdict="review", reasons=[])
        report.stages_available = {
            "localize": self.corner is not None,
            "ocr": self.crnn is not None,
            "forensics": self.tamper is not None,
            "face": self.face is not None and self.calibration is not None,
        }
        reasons = report.reasons
        canonical = None

        if self.corner is not None:
            t0 = time.perf_counter()
            corners = self.locate(document_rgb)
            canonical, kind, _ = rectify(document_rgb, corners)
            report.timings_ms["localize"] = (time.perf_counter() - t0) * 1000
            report.corners = corners.tolist()
            report.kind = kind
        else:
            reasons.append("localization model unavailable")

        if canonical is not None and self.reader is not None:
            t0 = time.perf_counter()
            gray = cv2.cvtColor(canonical, cv2.COLOR_RGB2GRAY)
            doc = self.reader.read(gray, report.kind)
            report.timings_ms["ocr"] = (time.perf_counter() - t0) * 1000
            report.document = doc
            invalid = [f.name for f in doc.fields.values()
                       if not f.format_valid and not f.name.startswith("mrz_")]
            if invalid:
                reasons.append(f"invalid field formats: {', '.join(sorted(invalid))}")
            if doc.mrz_valid is False:
                failed = [k for k, ok in doc.mrz_checks.items() if not ok]
                reasons.append("MRZ check digits failed: "
                               + (", ".join(failed) or "unparseable"))
            if any(not ok for ok in doc.viz_mrz_consistency.values()):
                bad = [k for k, ok in doc.viz_mrz_consistency.items() if not ok]
                reasons.append(f"VIZ/MRZ disagreement: {', '.join(bad)}")

        if canonical is not None and self.tamper is not None:
            t0 = time.perf_counter()
            prob, heat = self.inspect(canonical)
            report.timings_ms["forensics"] = (time.perf_counter() - t0) * 1000
            report.tamper_probability = prob
            report.tamper_heatmap = np.round(heat, 3).tolist()
            if prob >= TAMPER_REVIEW_THRESHOLD:
                reasons.append(f"forensic tamper probability {prob:.2f}")

        face_fail = False
        if selfie_rgb is not None and report.stages_available["face"]:
            t0 = time.perf_counter()
            doc_face = self.face.embed(canonical if canonical is not None
                                       else document_rgb)
            selfie_face = self.face.embed(selfie_rgb)
            report.timings_ms["face"] = (time.perf_counter() - t0) * 1000
            if doc_face is None:
                reasons.append("no face found on document")
            elif selfie_face is None:
                reasons.append("no face found in selfie")
            else:
                sim = cosine(doc_face[0], selfie_face[0])
                m = decide(sim, self.calibration, self.fmr_level)
                report.face_similarity = sim
                report.face_match = m.match
                report.face_threshold = m.threshold
                if not m.match:
                    face_fail = True
                    reasons.append(
                        f"face similarity {sim:.3f} below threshold "
                        f"{m.threshold:.3f} (FMR {self.fmr_level})")

        mrz_hard_fail = (report.document is not None
                         and report.document.mrz_valid is False)
        if face_fail or mrz_hard_fail:
            report.verdict = "fail"
        elif reasons:
            report.verdict = "review"
        else:
            report.verdict = "pass"
            reasons.append("all checks passed")
        report.timings_ms["total"] = sum(
            v for k, v in report.timings_ms.items() if k != "total")
        return report
