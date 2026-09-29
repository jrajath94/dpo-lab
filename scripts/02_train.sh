#!/usr/bin/env bash
# Step 2: launch DPO training. Needs the GPU box and the splits from step 1.
#
# Optional env overrides (used by 04_beta_sweep.sh):
#   BETA     - override dpo.beta, e.g. BETA=0.05
#   OUT_DIR  - override the run output dir, e.g. OUT_DIR=results/runs/beta_0.05
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

CONFIG="${1:-configs/dpo_qwen25_15b_qlora.yaml}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Qwen weights are gated; accept the license at"
  echo "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct and export HF_TOKEN."
  exit 1
fi

if [ ! -f "results/splits/train.jsonl" ]; then
  echo "results/splits/train.jsonl is missing. Run scripts/01_prepare_data.sh first."
  exit 1
fi

SET_ARGS=()
if [ -n "${BETA:-}" ]; then
  SET_ARGS+=(--set "dpo.beta=${BETA}")
  echo "beta override: ${BETA}"
fi
if [ -n "${OUT_DIR:-}" ]; then
  SET_ARGS+=(--output-dir "${OUT_DIR}")
  echo "output dir override: ${OUT_DIR}"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -m dpo_weekend.train --config "$CONFIG" "${SET_ARGS[@]}"

echo ""
echo "Training finished. Check the first 50 steps of the log before trusting it:"
echo "  python - <<'EOF'"
echo "  import json"
echo "  rows=[json.loads(l) for l in open('<run_dir>/training_log.jsonl')]"
echo "  steps=[r for r in rows if 'loss' in r][:60]"
echo "  print('steps:', len(steps), 'loss first/last:', steps[0]['loss'], steps[-1]['loss'])"
echo "  EOF"
