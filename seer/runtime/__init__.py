"""CPU inference runtime.

PyTorch is a training dependency, not a serving one: every trained model is
exported to ONNX, quantized to INT8 where it pays, and served through ONNX
Runtime with a benchmark harness that holds the pipeline to a measured
latency budget. Quantization policy is per-architecture: static QDQ for the
conv nets (corner, tamper), dynamic for the CRNN (LSTM weights quantize
well dynamically; static activation ranges through a recurrence are
brittle).
"""
