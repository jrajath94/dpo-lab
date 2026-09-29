#!/usr/bin/env bash
# Step 0: CPU-only smoke test. Safe to run anywhere, no GPU, no network.
# Validates: package imports, pure-function unit tests, YAML configs parse
# with all required keys, and no unfinished stubs left in the run path.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

echo "== python =="
python -c "import dpo_weekend; print('dpo_weekend', dpo_weekend.__version__)"

echo "== unit tests =="
python -m pytest tests/ -q

echo "== config + run-path checks (tests/test_smoke.py) =="
python -m pytest tests/test_smoke.py -q

echo "SMOKE OK"
