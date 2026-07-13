"""Export trained checkpoints to ONNX.

    python -m seer.runtime.export --all
    python -m seer.runtime.export --model corner --ckpt weights/corner.pt

Every export is validated by comparing ONNX Runtime output against the
PyTorch model on the same random input — an export that silently reorders
outputs or bakes a wrong shape fails here, not in production.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from seer.forensics.model import INPUT_SIZE as TAMPER_WH
from seer.forensics.model import TamperNet
from seer.localize.model import INPUT_SIZE as CORNER_SIZE
from seer.localize.model import CornerNet
from seer.ocr.crnn import CRNN, LINE_HEIGHT, LINE_WIDTH

WEIGHTS = Path("weights")

SPECS = {
    "corner": {
        "build": lambda: CornerNet(pretrained=False),
        "dummy": lambda: torch.randn(1, 3, CORNER_SIZE, CORNER_SIZE),
        "outputs": ["coords", "heatmaps"],
    },
    "crnn": {
        "build": CRNN,
        "dummy": lambda: torch.randn(1, 1, LINE_HEIGHT, LINE_WIDTH),
        "outputs": ["logprobs"],
    },
    "tamper": {
        "build": TamperNet,
        "dummy": lambda: torch.randn(1, 3, TAMPER_WH[1], TAMPER_WH[0]),
        "outputs": ["doc_logit", "heatmap"],
    },
}


def export_model(name: str, ckpt: Path, out: Path) -> None:
    spec = SPECS[name]
    model = spec["build"]()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    dummy = spec["dummy"]()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=spec["outputs"],
        dynamic_axes={"input": {0: "batch"},
                      **{o: {0: "batch"} for o in spec["outputs"]}},
        opset_version=17,
        # legacy TorchScript exporter: deterministic, no onnxscript
        # dependency, and handles the CRNN's LSTM cleanly
        dynamo=False,
    )
    _validate(model, dummy, out)
    print(f"[ ok ] {name}: {ckpt} -> {out}")


def _validate(model: torch.nn.Module, dummy: torch.Tensor, onnx_path: Path,
              atol: float = 1e-3) -> None:
    import onnxruntime as ort
    with torch.no_grad():
        ref = model(dummy)
    refs = [r.numpy() for r in (ref if isinstance(ref, tuple) else (ref,))]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"input": dummy.numpy()})
    for i, (r, o) in enumerate(zip(refs, outs)):
        if not np.allclose(r, o, atol=atol):
            raise RuntimeError(
                f"{onnx_path.name} output {i} diverges from PyTorch "
                f"(max abs err {np.abs(r - o).max():.5f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", choices=sorted(SPECS))
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.all:
        for name in SPECS:
            ckpt = WEIGHTS / f"{name}.pt"
            if ckpt.exists():
                export_model(name, ckpt, WEIGHTS / f"{name}.onnx")
            else:
                print(f"[skip] {name}: {ckpt} not found")
    elif args.model:
        ckpt = args.ckpt or WEIGHTS / f"{args.model}.pt"
        out = args.out or WEIGHTS / f"{args.model}.onnx"
        export_model(args.model, ckpt, out)
    else:
        ap.error("pass --all or --model")


if __name__ == "__main__":
    main()
