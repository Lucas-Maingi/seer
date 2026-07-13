"""Verification decision logic with FMR-calibrated thresholds.

A face-match threshold is a business decision expressed in math: fixing an
acceptable false-match rate (an impostor passing) determines the threshold
and *implies* a false-non-match rate (a genuine user bounced). We calibrate
on labeled pairs and persist the whole operating curve, so the service can
expose "match at FMR 1e-3" rather than an arbitrary 0.5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-10))


@dataclass
class Calibration:
    thresholds: dict[str, float]  # fmr label ("1e-2", "1e-3", ...) -> threshold
    tpr_at_fmr: dict[str, float]
    n_genuine: int
    n_impostor: int

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text()))


def calibrate(genuine_scores: np.ndarray, impostor_scores: np.ndarray,
              fmr_targets: tuple[float, ...] = (1e-2, 1e-3, 1e-4)) -> Calibration:
    """Pick thresholds achieving each target FMR on the impostor scores.

    The threshold for FMR target f is the (1-f) quantile of impostor
    similarities; TPR is then measured on the genuine scores.
    """
    genuine = np.asarray(genuine_scores, np.float64)
    impostor = np.asarray(impostor_scores, np.float64)
    thresholds, tprs = {}, {}
    for f in fmr_targets:
        key = f"{f:.0e}".replace("e-0", "e-")
        thr = float(np.quantile(impostor, 1 - f))
        thresholds[key] = thr
        tprs[key] = float((genuine >= thr).mean())
    return Calibration(thresholds=thresholds, tpr_at_fmr=tprs,
                       n_genuine=len(genuine), n_impostor=len(impostor))


@dataclass
class MatchResult:
    similarity: float
    threshold: float
    fmr_level: str
    match: bool


def decide(similarity: float, calibration: Calibration,
           fmr_level: str = "1e-3") -> MatchResult:
    thr = calibration.thresholds[fmr_level]
    return MatchResult(similarity=similarity, threshold=thr,
                       fmr_level=fmr_level, match=similarity >= thr)
