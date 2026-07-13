"""Runtime invariants: ONNX export parity, quantization, bench harness."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from seer.runtime.bench import bench_model  # noqa: E402
from seer.runtime.export import SPECS, export_model  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export every architecture with random weights; parity is validated
    inside export_model (raises if ORT and PyTorch outputs diverge)."""
    tmp = tmp_path_factory.mktemp("onnx")
    paths = {}
    for name, spec in SPECS.items():
        ckpt = tmp / f"{name}.pt"
        torch.save({"model": spec["build"]().state_dict()}, ckpt)
        out = tmp / f"{name}.onnx"
        export_model(name, ckpt, out)
        paths[name] = out
    return paths


def test_export_all_architectures(exported):
    assert set(exported) == {"corner", "crnn", "tamper"}
    for p in exported.values():
        assert p.exists() and p.stat().st_size > 10_000


def test_exported_corner_batch_dynamic(exported):
    sess = ort.InferenceSession(str(exported["corner"]),
                                providers=["CPUExecutionProvider"])
    for b in (1, 3):
        x = np.random.randn(b, 3, 256, 256).astype(np.float32)
        coords, _ = sess.run(None, {"input": x})
        assert coords.shape == (b, 4, 2)
        assert np.abs(coords).max() <= 1.0


def test_dynamic_quantization_shrinks_crnn(exported, tmp_path):
    from seer.runtime.quantize import quantize_dynamic_
    out = tmp_path / "crnn.int8.onnx"
    quantize_dynamic_(exported["crnn"], out)
    assert out.stat().st_size < 0.5 * exported["crnn"].stat().st_size
    # quantized model still runs and normalizes
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 1, 32, 384).astype(np.float32)
    (logp,) = sess.run(None, {"input": x})
    assert np.allclose(np.exp(logp).sum(-1), 1.0, atol=1e-3)


def test_bench_harness(exported):
    m = bench_model(exported["tamper"],
                    lambda: np.random.randn(1, 3, 256, 384).astype(np.float32),
                    runs=5, warmup=2)
    assert m["p50"] > 0 and m["p95"] >= m["p50"]
