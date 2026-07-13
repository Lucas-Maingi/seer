"""CPU latency benchmark.

    python -m seer.runtime.bench --report runs/latency.md [--runs 200]

Measures every ONNX model found in weights/ (fp32 and int8 side by side)
with warmup, reporting p50/p95/mean per inference — percentiles, because a
KYC endpoint's SLA lives at the tail, not the average. Writes a markdown
report suitable for committing alongside the code that produced it.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from seer.forensics.model import INPUT_SIZE as TAMPER_WH
from seer.localize.model import INPUT_SIZE as CORNER_SIZE
from seer.ocr.crnn import LINE_HEIGHT, LINE_WIDTH

WEIGHTS = Path("weights")

INPUTS = {
    "corner": lambda: np.random.randn(1, 3, CORNER_SIZE, CORNER_SIZE).astype(np.float32),
    "crnn": lambda: np.random.randn(1, 1, LINE_HEIGHT, LINE_WIDTH).astype(np.float32),
    "tamper": lambda: np.random.randn(1, 3, TAMPER_WH[1], TAMPER_WH[0]).astype(np.float32),
}


def bench_model(path: Path, make_input, runs: int, warmup: int = 15) -> dict[str, float]:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = make_input()
    for _ in range(warmup):
        sess.run(None, {"input": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"p50": float(np.percentile(t, 50)),
            "p95": float(np.percentile(t, 95)),
            "mean": float(t.mean()),
            "size_mb": path.stat().st_size / 1e6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--report", type=Path, default=Path("runs/latency.md"))
    args = ap.parse_args()

    rows = []
    for name, make_input in INPUTS.items():
        for suffix, label in ((".onnx", "fp32"), (".int8.onnx", "int8")):
            path = WEIGHTS / f"{name}{suffix}"
            if not path.exists():
                continue
            m = bench_model(path, make_input, args.runs)
            rows.append((name, label, m))
            print(f"{name:8s} {label:5s} p50 {m['p50']:7.2f}ms  "
                  f"p95 {m['p95']:7.2f}ms  {m['size_mb']:.1f} MB")

    if not rows:
        print("no ONNX models found in weights/ — run seer.runtime.export first")
        return

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Seer CPU latency report", "",
        f"- host: {platform.processor() or platform.machine()}",
        f"- runs per model: {args.runs} (after 15 warmup)", "",
        "| stage | precision | p50 (ms) | p95 (ms) | mean (ms) | size (MB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, label, m in rows:
        lines.append(f"| {name} | {label} | {m['p50']:.2f} | {m['p95']:.2f} "
                     f"| {m['mean']:.2f} | {m['size_mb']:.1f} |")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
