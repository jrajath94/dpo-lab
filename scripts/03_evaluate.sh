#!/usr/bin/env bash
# Step 3: run all evals for one trained run. Needs the GPU box.
#
# Usage: 03_evaluate.sh <run-dir> [splits-dir]
#   <run-dir>    e.g. results/runs/beta_0.1 (must contain the adapter)
#   [splits-dir] defaults to results/splits
#
# Runs, in order: margin eval (base, then dpo), generation (base, then dpo),
# blinded judge (needs OPENROUTER_API_KEY), MMLU regression guard
# (base, then dpo), diagnostic plots. Everything lands in <run-dir>/evals/.
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

RUN_DIR="${1:?usage: 03_evaluate.sh <run-dir> [splits-dir]}"
SPLITS_DIR="${2:-results/splits}"
EVALS_DIR="${RUN_DIR}/evals"
CONFIG="${RUN_DIR}/config.yaml"
mkdir -p "$EVALS_DIR"

# One failing eval step must not nuke the rest (run 5, 2026-09-29: the
# generation crash killed the MMLU guard and plots too, and would have
# killed already-finished margin results' downstream consumers).
# Record failures, run everything, exit nonzero at the end if any failed.
FAILED=()
run_step() {
  local name="$1"; shift
  echo "== $name =="
  if "$@"; then
    echo "-- $name OK"
  else
    echo "-- $name FAILED (continuing)"
    FAILED+=("$name")
  fi
}

if [ ! -f "$CONFIG" ]; then
  echo "$CONFIG is missing. Train first (scripts/02_train.sh copies it there)."
  exit 1
fi
if [ ! -f "${SPLITS_DIR}/margin_eval.jsonl" ]; then
  echo "${SPLITS_DIR}/margin_eval.jsonl is missing. Run 01_prepare_data.sh first."
  exit 1
fi

BETA="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['dpo']['beta'])")"
echo "run: $RUN_DIR  beta: $BETA"

run_step "[1/5] margin eval: base" \
  python -m dpo_weekend.evaluate --eval margin --config "$CONFIG" \
  --splits-dir "$SPLITS_DIR" --beta "$BETA" \
  --out "$EVALS_DIR/margin_base.json"

run_step "[2/5] margin eval: dpo" \
  python -m dpo_weekend.evaluate --eval margin --config "$CONFIG" \
  --adapter-dir "$RUN_DIR" --splits-dir "$SPLITS_DIR" --beta "$BETA" \
  --out "$EVALS_DIR/margin_dpo.json"

run_step "[3/5] generation: base" \
  python -m dpo_weekend.evaluate --eval generation --config "$CONFIG" \
  --splits-dir "$SPLITS_DIR" --out "$EVALS_DIR/gens_base.jsonl"

run_step "[4/5] generation: dpo" \
  python -m dpo_weekend.evaluate --eval generation --config "$CONFIG" \
  --adapter-dir "$RUN_DIR" --splits-dir "$SPLITS_DIR" \
  --out "$EVALS_DIR/gens_dpo.jsonl"

echo "== [5/5] blinded judge (order-swapped) =="
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is not set; skipping the judge step."
  echo "Everything else is done. Re-run just the judge later with:"
  echo "  python -m dpo_weekend.evaluate --eval judge --config $CONFIG \\"
  echo "    --base-gens $EVALS_DIR/gens_base.jsonl \\"
  echo "    --dpo-gens $EVALS_DIR/gens_dpo.jsonl \\"
  echo "    --splits-dir $SPLITS_DIR --out $EVALS_DIR/judge_labels.json"
else
  run_step "[5/5] blinded judge" \
    python -m dpo_weekend.evaluate --eval judge --config "$CONFIG" \
    --splits-dir "$SPLITS_DIR" \
    --base-gens "$EVALS_DIR/gens_base.jsonl" \
    --dpo-gens "$EVALS_DIR/gens_dpo.jsonl" \
    --out "$EVALS_DIR/judge_labels.json"
fi

run_step "[6/6] MMLU regression guard: base" \
  python -m dpo_weekend.evaluate --eval regression --config "$CONFIG" \
  --splits-dir "$SPLITS_DIR" --out "$EVALS_DIR/mmlu_base.json"
run_step "[6/6] MMLU regression guard: dpo" \
  python -m dpo_weekend.evaluate --eval regression --config "$CONFIG" \
  --adapter-dir "$RUN_DIR" --splits-dir "$SPLITS_DIR" \
  --out "$EVALS_DIR/mmlu_dpo.json"

run_step "[plots]" \
  python -m dpo_weekend.evaluate --eval plots --config "$CONFIG" \
  --adapter-dir "$RUN_DIR" --splits-dir "$SPLITS_DIR" \
  --beta "$BETA" --out "$RUN_DIR/plots"

echo ""
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "EVAL FAILURES: ${FAILED[*]}"
  echo "Partial results are in $EVALS_DIR and $RUN_DIR/plots."
  exit 1
fi
echo "All evals done. Results in $EVALS_DIR and $RUN_DIR/plots."
echo "See EVAL-PLAN.md for how to read every number."
