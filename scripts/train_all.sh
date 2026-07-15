#!/usr/bin/env bash
# One-command reproduction of the full training + export sequence.
# Designed for a free-tier cloud GPU (Colab T4, Kaggle P100) but runs
# anywhere. Tune via env vars:
#
#   COUNT=8000 EPOCHS_CORNER=30 bash scripts/train_all.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

COUNT="${COUNT:-8000}"
DATA="${DATA:-data/synth}"
FACES="${FACES:-}"   # optional dir of synthetic faces (SFHQ etc.)

echo "== 1/6 dataset ($COUNT samples -> $DATA) =="
if [ -f "$DATA/index.jsonl" ] && [ "$(wc -l < "$DATA/index.jsonl")" -ge "$COUNT" ]; then
    echo "dataset already present, skipping generation"
else
    python -m seer.synth.generate --out "$DATA" --count "$COUNT" \
        ${FACES:+--faces "$FACES"}
fi

echo "== 2/6 corner localization =="
python -m seer.localize.train --data "$DATA" \
    --epochs "${EPOCHS_CORNER:-30}" --out weights/corner.pt

echo "== 3/6 field OCR =="
python -m seer.ocr.train --data "$DATA" \
    --epochs "${EPOCHS_CRNN:-40}" --out weights/crnn.pt

echo "== 4/6 tamper forensics =="
python -m seer.forensics.train --data "$DATA" \
    --epochs "${EPOCHS_TAMPER:-25}" --out weights/tamper.pt

echo "== 5/6 ONNX export + INT8 quantization =="
python -m seer.runtime.export --all
python -m seer.runtime.quantize --all --calib "$DATA"

echo "== 6/6 latency benchmark =="
python -m seer.runtime.bench --report runs/latency.md

echo
echo "done. deployable artifacts:"
ls -lh weights/*.onnx
echo "copy weights/*.onnx (+ face models via scripts/fetch_models.py) to the serving machine."
