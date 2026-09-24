#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/ablation_16ep}"
for SEED in 2027 2028 2029; do
  "$PYTHON" train.py --epochs 16 --seed "$SEED" --output "$OUT/seed$SEED" "$@"
  "$PYTHON" eval.py --checkpoint "$OUT/seed$SEED/final.pth"
done
