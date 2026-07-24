# INT8 quantization: measured results and per-model policy

Quantization was applied and then **validated for accuracy, not just size and
speed** — the standard mistake is to quantize, confirm the file shrank, and
ship. Measured on an Intel i7-6600U (Skylake, no AVX512-VNNI) against fresh
held-out synthetic documents:

| model  | quant       | accuracy fp32 → int8            | latency fp32 → int8 | verdict |
|--------|-------------|--------------------------------|---------------------|---------|
| corner | static QDQ  | 2.0px → **264px** (collapsed)  | 11.6ms → 12.8ms     | **reject int8** |
| crnn   | dynamic     | reads → **garbled in batch**   | 51ms → **731ms**    | **reject int8** |
| tamper | static QDQ  | AUC 0.63 → 0.62 (fine)         | 54ms → 33ms         | keep int8 |

## Why corner static-INT8 collapses

The localization head is a DSNT (differentiable spatial-to-numerical)
soft-argmax over predicted heatmaps. Static activation quantization crushes
the pre-softmax dynamic range; several heatmaps flatten to near-uniform and
their expectation decodes to the image center, so two of four corners snap to
(W/2, H/2) and rectification is destroyed. Spatial-softmax heads are known to
be quantization-sensitive; fixing this would need QAT or per-channel
quantization of the head, not post-training static quant.

## Why crnn dynamic-INT8 regresses

The recognizer is a CRNN whose bulk is a 2-layer BiLSTM. ONNX Runtime's
dynamic quantization emulates int8 LSTM matmuls on CPUs lacking AVX512-VNNI,
which is *slower* than the fp32 MKL path, and the emulation degrades the
batched (24-ROI) forward enough to garble decoding.

## Policy

`seer.runtime.pipeline._NO_INT8 = {"corner", "crnn"}` — these serve fp32; only
`tamper` serves int8. fp32 corner+crnn+tamper is ~120ms/doc of model compute
on this CPU, comfortably within budget, so rejecting int8 for two stages costs
nothing that matters while avoiding a silent accuracy cliff.

The lesson worth keeping: **quantization is a per-tensor-distribution
decision, verified against task accuracy on the target hardware** — a blanket
"quantize everything to int8" would have shipped a pipeline that returns the
image center for every document.
