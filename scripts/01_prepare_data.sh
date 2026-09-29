#!/usr/bin/env bash
# Step 1: build the seeded data splits. CPU-only, safe to run anywhere.
# Downloads ultrafeedback_binarized from HF, normalizes, filters, splits,
# and writes results/splits/ (shared by every beta-sweep run).
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

CONFIG="${1:-configs/dpo_qwen25_15b_qlora.yaml}"
OUT_DIR="${OUT_DIR:-results/splits}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set."
else
  echo "note: HF_TOKEN is not set. The dataset is public, but the Qwen"
  echo "tokenizer used for length filtering is gated, so accept the license at"
  echo "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct and export HF_TOKEN."
fi

python -m dpo_weekend.data --config "$CONFIG" --out-dir "$OUT_DIR"

echo ""
echo "Splits written to $OUT_DIR."
echo "Next: read 20 random pairs by eye before training:"
echo "  python - <<'EOF'"
echo "  import json, random"
echo "  rows = [json.loads(l) for l in open('$OUT_DIR/train.jsonl')]"
echo "  for r in random.sample(rows, 20):"
echo "      print('PROMPT:', r['prompt'][0]['content'][:200])"
echo "      print('CHOSEN:', r['chosen'][-1]['content'][:200])"
echo "      print('REJECTED:', r['rejected'][-1]['content'][:200])"
echo "      print('---')"
echo "  EOF"
