#!/usr/bin/env bash
# Step 4 (optional second weekend): beta sweep over {0.05, 0.1, 0.2}.
# Each run trains into results/runs/beta_<b>/, then runs the full eval suite.
# Finish with: python scripts/05_compare_betas.py
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-configs/dpo_qwen25_15b_qlora.yaml}"
BETAS="${BETAS:-0.05 0.1 0.2}"

for b in $BETAS; do
  RUN_NAME="beta_${b}"
  echo "================================================================"
  echo " beta sweep: beta=${b} -> results/runs/${RUN_NAME}"
  echo "================================================================"
  BETA="$b" OUT_DIR="results/runs/${RUN_NAME}" \
    bash scripts/02_train.sh "$CONFIG"
  bash scripts/03_evaluate.sh "results/runs/${RUN_NAME}"
done

echo ""
echo "Sweep done. Compare the runs:"
echo "  python scripts/05_compare_betas.py"
