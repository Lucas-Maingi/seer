"""INT8 quantization with per-architecture policy.

    python -m seer.runtime.quantize --all --calib data/synth

- corner & tamper (pure conv): **static QDQ** quantization. Activation
  ranges are calibrated on real preprocessed samples from the synth set —
  calibrating on random noise gives ranges that clip real activations.
- crnn (LSTM): **dynamic** quantization of matmul weights. Static
  activation quantization through a recurrence compounds error per time
  step; dynamic keeps the recurrent state in float while still shrinking
  the dominant weight matrices.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path("weights")


class _SynthCalibrationReader:
    """Feeds N preprocessed synth samples to the static quantizer."""

    def __init__(self, model: str, data_root: Path, n: int = 64):
        self.samples = self._load(model, data_root, n)
        self._it = iter(self.samples)

    @staticmethod
    def _load(model: str, root: Path, n: int) -> list[dict[str, np.ndarray]]:
        ids = [json.loads(line)["id"]
               for line in (root / "index.jsonl").read_text().splitlines()]
        random.Random(0).shuffle(ids)
        feeds = []
        for sid in ids[:n]:
            if model == "corner":
                from seer.localize.dataset import normalize_image
                from seer.localize.model import INPUT_SIZE
                img = cv2.cvtColor(cv2.imread(str(root / "scenes" / f"{sid}.jpg")),
                                   cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
                feeds.append({"input": normalize_image(img).numpy()[None]})
            elif model == "tamper":
                from seer.forensics.model import INPUT_SIZE as WH
                from seer.forensics.signals import forensic_stack
                from seer.datautil import canonical_path
                img = cv2.cvtColor(cv2.imread(str(canonical_path(root, sid))),
                                   cv2.COLOR_BGR2RGB)
                feeds.append({"input": forensic_stack(img, WH)[None]})
        return feeds

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.samples)


def quantize_static(onnx_in: Path, onnx_out: Path, model: str, calib_root: Path) -> None:
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.inner = _SynthCalibrationReader(model, calib_root)

        def get_next(self):
            return self.inner.get_next()

    quantize_static(str(onnx_in), str(onnx_out), Reader(),
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    print(f"[ ok ] static int8: {onnx_in.name} -> {onnx_out.name}")


def quantize_dynamic_(onnx_in: Path, onnx_out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(onnx_in), str(onnx_out), weight_type=QuantType.QInt8)
    print(f"[ ok ] dynamic int8: {onnx_in.name} -> {onnx_out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", choices=["corner", "crnn", "tamper"])
    ap.add_argument("--calib", type=Path, help="synth dataset root (static quant)")
    args = ap.parse_args()

    targets = ["corner", "crnn", "tamper"] if args.all else [args.model]
    for name in targets:
        src = WEIGHTS / f"{name}.onnx"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        dst = WEIGHTS / f"{name}.int8.onnx"
        if name == "crnn":
            quantize_dynamic_(src, dst)
        else:
            if not args.calib:
                print(f"[skip] {name}: static quantization needs --calib")
                continue
            quantize_static(src, dst, name, args.calib)


if __name__ == "__main__":
    main()
