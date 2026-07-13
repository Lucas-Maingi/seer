"""Calibrate face-match thresholds on LFW pairs.

    python -m seer.face.calibrate --lfw data/lfw --pairs data/lfw/pairs.txt \
        --out weights/face_calibration.json

LFW (Labeled Faces in the Wild) is public and standard; using it makes the
reported operating points comparable to published systems. pairs.txt is the
official protocol file: genuine lines "name i j", impostor lines
"name1 i name2 j".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from seer.face.embed import ArcFaceEmbedder, FacePipeline, YuNetDetector
from seer.face.verify import calibrate, cosine

DEFAULT_DETECTOR = Path("weights/face_detection_yunet_2023mar.onnx")
DEFAULT_ARCFACE = Path("weights/arcface_w600k_r50.onnx")


def _lfw_path(root: Path, name: str, idx: int) -> Path:
    return root / name / f"{name}_{idx:04d}.jpg"


def parse_pairs(pairs_file: Path) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    genuine, impostor = [], []
    root = pairs_file.parent
    lines = pairs_file.read_text().splitlines()
    for line in lines[1:]:  # first line is fold header
        parts = line.split()
        if len(parts) == 3:
            name, i, j = parts
            genuine.append((_lfw_path(root, name, int(i)), _lfw_path(root, name, int(j))))
        elif len(parts) == 4:
            n1, i, n2, j = parts
            impostor.append((_lfw_path(root, n1, int(i)), _lfw_path(root, n2, int(j))))
    return genuine, impostor


def _score_pairs(pipeline: FacePipeline, pairs: list[tuple[Path, Path]],
                 desc: str) -> np.ndarray:
    cache: dict[Path, np.ndarray | None] = {}

    def emb(p: Path) -> np.ndarray | None:
        if p not in cache:
            img = cv2.imread(str(p))
            if img is None:
                cache[p] = None
            else:
                r = pipeline.embed(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                cache[p] = None if r is None else r[0]
        return cache[p]

    scores = []
    for a, b in tqdm(pairs, desc=desc):
        ea, eb = emb(a), emb(b)
        if ea is not None and eb is not None:
            scores.append(cosine(ea, eb))
    return np.array(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    ap.add_argument("--arcface", type=Path, default=DEFAULT_ARCFACE)
    ap.add_argument("--out", type=Path, default=Path("weights/face_calibration.json"))
    args = ap.parse_args()

    pipeline = FacePipeline(YuNetDetector(args.detector), ArcFaceEmbedder(args.arcface))
    genuine_pairs, impostor_pairs = parse_pairs(args.pairs)
    genuine = _score_pairs(pipeline, genuine_pairs, "genuine")
    impostor = _score_pairs(pipeline, impostor_pairs, "impostor")

    cal = calibrate(genuine, impostor)
    cal.save(args.out)
    print(f"genuine pairs scored: {cal.n_genuine}, impostor: {cal.n_impostor}")
    for k, thr in cal.thresholds.items():
        print(f"  FMR {k}: threshold {thr:.4f}, TPR {cal.tpr_at_fmr[k]:.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
