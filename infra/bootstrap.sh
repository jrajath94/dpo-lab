#!/bin/bash
# In-pod bootstrap for the dpo-lab weekend run (RTX 4090, RunPod secure cloud).
# Runs as the container's start command (no SSH needed).
# Steps: unpack code -> venv install -> smoke test -> HF gated-access check ->
# data prep -> training (monitor.py watchdog) -> eval (judge step skipped,
# no OpenRouter key on pod) -> package results -> serve on :8080 with token
# auth -> pod stays alive until the VM fetches results, then the VM terminates it.
#
# Secrets: HF_TOKEN and SERVE_TOKEN arrive as pod env vars (set by
# infra/launch_pod.py). They are never written to disk or the tarball.
set -u
export PYTHONUNBUFFERED=1
mkdir -p /workspace/serve
LOG=/workspace/serve/bootstrap.log
exec > >(tee -a "$LOG") 2>&1

echo "=== bootstrap start $(date -u +%FT%TZ) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo UNKNOWN)"

cd /workspace
echo "$CODE_GZ_B64" | base64 -d | tar xzf -
# monitor.py/serve.py ride inside the code tarball (.ship/) to keep the pod
# create-request under RunPod's ~100KB body limit; env-var delivery is kept
# as a fallback for older launch scripts.
if [ -n "${MONITOR_B64:-}" ]; then
  echo "$MONITOR_B64" | base64 -d > /workspace/monitor.py
else
  cp /workspace/dpo-lab/.ship/monitor.py /workspace/monitor.py
fi
if [ -n "${SERVE_B64:-}" ]; then
  echo "$SERVE_B64" | base64 -d > /workspace/serve.py
else
  cp /workspace/dpo-lab/.ship/serve.py /workspace/serve.py
fi

# Start the results server in the background NOW so the VM can tail
# bootstrap.log and status.json live via
# https://<pod-id>-8080.proxy.runpod.net/<file>?token=<SERVE_TOKEN>
python3 /workspace/serve.py &
echo "results server started in background (pid $!)"
cd /workspace/dpo-lab
echo "code unpacked: $(du -sh /workspace/dpo-lab | cut -f1)"

echo "=== step 0: install ==="
export TMPDIR=/workspace/pip-tmp
mkdir -p "$TMPDIR"
# The RunPod PyTorch image already ships torch + the CUDA stack system-wide.
# Inherit it so pip does NOT re-download ~4GB of torch/nvidia wheels.
python3 -m venv --system-site-packages .venv
export PATH="/workspace/dpo-lab/.venv/bin:$PATH"
hash -r
.venv/bin/python -c "import torch; print('system torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
(while true; do echo "[heartbeat $(date -u +%H:%M:%S)] install in progress"; sleep 60; done) &
HEARTBEAT_PID=$!
timeout 1800 .venv/bin/pip install -U pip | tail -2
INSTALL_OK=1
timeout 1800 .venv/bin/pip install -e ".[eval,test]" || INSTALL_OK=0
kill "$HEARTBEAT_PID" 2>/dev/null || true
if [ "$INSTALL_OK" != "1" ]; then
  echo "PIP INSTALL FAILED OR TIMED OUT - stopping"
  python3 -c "import json; json.dump({'status':'failed','reason':'install'}, open('/workspace/serve/status.json','w'))"
  wait
fi
.venv/bin/pip freeze > requirements.lock
echo "INSTALL DONE"

echo "=== step 1: smoke test ==="
if ! bash scripts/00_smoke_test.sh; then
  echo "SMOKE FAILED - stopping before any GPU spend"
  python3 -c "import json; json.dump({'status':'failed','reason':'smoke_test'}, open('/workspace/serve/status.json','w'))"
  echo "results server (background) keeps the container alive for log inspection"
  wait
fi

echo "=== HF gated-access check (Qwen weights need HF_TOKEN) ==="
if ! .venv/bin/python - <<'EOF'
import os, sys
from huggingface_hub import HfApi
tok = os.environ.get("HF_TOKEN")
if not tok:
    print("HF_TOKEN is empty - cannot download gated Qwen weights")
    sys.exit(1)
api = HfApi(token=tok)
info = api.model_info("Qwen/Qwen2.5-1.5B-Instruct")
print("model reachable:", info.id, "| gated:", info.gated)
u = api.whoami()
print("authenticated as:", u.get("name"))
EOF
then
  echo "HF GATED ACCESS CHECK FAILED - stopping (accept the Qwen license / check HF_TOKEN)"
  python3 -c "import json; json.dump({'status':'failed','reason':'hf_access'}, open('/workspace/serve/status.json','w'))"
  wait
fi

echo "=== step 2: data prep ==="
# Fail fast: a broken data stage must never slide into a training phase
# that would immediately refuse to start (learned 2026-09-28, first pod
# run died here on a NameError and the pipeline kept walking).
if ! bash scripts/01_prepare_data.sh; then
  echo "DATA PREP FAILED - stopping before any training spend"
  python3 -c "import json; json.dump({'status':'failed','reason':'data_prep'}, open('/workspace/serve/status.json','w'))"
  echo "results server (background) keeps the container alive for log inspection"
  wait
fi
echo "--- sample pairs for the log (eye-check happens on the VM) ---"
.venv/bin/python - <<'EOF'
import json, random
rows = [json.loads(l) for l in open('results/splits/train.jsonl')]
print("train pairs:", len(rows))
lc = sum(len(r['chosen'][-1]['content']) for r in rows) / len(rows)
lr = sum(len(r['rejected'][-1]['content']) for r in rows) / len(rows)
print(f"mean chosen chars: {lc:.0f} | mean rejected chars: {lr:.0f}")
for r in random.Random(7).sample(rows, 3):
    print('PROMPT:', r['prompt'][0]['content'][:150].replace('\n', ' '))
    print('CHOSEN:', r['chosen'][-1]['content'][:150].replace('\n', ' '))
    print('REJECTED:', r['rejected'][-1]['content'][:150].replace('\n', ' '))
    print('---')
EOF

echo "=== step 2.5: pre-flight dry run (10-step DPO smoke on GPU) ==="
# Proves the full GPU stack (bitsandbytes + trl + peft + CUDA) before the
# multi-hour run. 32 real pairs, 10 steps, then the smoke adapter and slice
# are deleted so the results package stays lean. Any failure stops here
# with status failed/dry_run. (Operating procedure 2026-09-28: no long-pole
# step without a small-scale dry run completed end to end.)
export PYTHONPATH="/workspace/dpo-lab/src:${PYTHONPATH:-}"
.venv/bin/python - <<'EOF'
import json, os
os.makedirs('results/splits_dryrun', exist_ok=True)
rows = [json.loads(l) for l in open('results/splits/train.jsonl')][:32]
with open('results/splits_dryrun/train.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
print('dry-run pairs:', len(rows))
EOF
if ! .venv/bin/python -m dpo_weekend.train \
    --config configs/dpo_qwen25_15b_qlora.yaml \
    --set data.splits_dir=results/splits_dryrun \
    --set dpo.num_train_epochs=10 \
    --set dpo.logging_steps=1 \
    --output-dir results/runs/dryrun; then
  echo "DRY RUN FAILED - GPU stack broken, stopping before the full run"
  python3 -c "import json; json.dump({'status':'failed','reason':'dry_run_train'}, open('/workspace/serve/status.json','w'))"
  echo "results server (background) keeps the container alive for log inspection"
  wait
fi
.venv/bin/python - <<'EOF'
import json, math
rows = []
for line in open('results/runs/dryrun/training_log.jsonl'):
    r = json.loads(line)
    if 'loss' in r:
        rows.append(r)
losses = [r['loss'] for r in rows]
assert len(rows) >= 5, f"dry run logged only {len(rows)} steps"
assert all(math.isfinite(x) for x in losses), "non-finite loss in dry run"
print(f"dry run OK: {len(rows)} steps, "
      f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
EOF
if [ $? -ne 0 ]; then
  echo "DRY RUN LOG CHECK FAILED - stopping before the full run"
  python3 -c "import json; json.dump({'status':'failed','reason':'dry_run_logcheck'}, open('/workspace/serve/status.json','w'))"
  wait
fi
rm -rf results/runs/dryrun results/splits_dryrun
echo "dry-run artifacts cleaned; proceeding to full training"

echo "=== step 3: training (monitor.py watchdog active) ==="
/workspace/dpo-lab/.venv/bin/python /workspace/monitor.py &
MONITOR_PID=$!
# Hard in-pod cap: 8 hours for the whole training phase.
if ! timeout 28800 bash scripts/02_train.sh; then
  echo "TRAINING PHASE EXITED NONZERO (killed by monitor or timeout - see status.json)"
fi
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

ADAPTER=/workspace/dpo-lab/results/runs/beta_0.1/adapter_config.json
if [ -f "$ADAPTER" ]; then
  echo "=== step 4: eval (judge step will skip: no OpenRouter key on pod) ==="
  bash scripts/03_evaluate.sh results/runs/beta_0.1 || echo "EVAL EXITED NONZERO"
else
  echo "no adapter found - skipping eval"
fi

echo "=== packaging results ==="
cd /workspace/dpo-lab
# Never declare "done" on an empty package (learned 2026-09-28: the first
# run shipped a 2KB tarball with only requirements.lock and a "done"
# status). A real run leaves an adapter and a training log.
if [ -f results/runs/beta_0.1/adapter_config.json ] && \
   [ -f results/runs/beta_0.1/training_log.jsonl ]; then
  tar czf /workspace/serve/results.tar.gz results/ requirements.lock
  FINAL_STATUS="done"
  FINAL_NOTE="results.tar.gz ready"
else
  echo "PACKAGING REFUSED: no adapter or training log in results/runs/beta_0.1/"
  ls -la results/ 2>/dev/null || echo "(no results/ dir at all)"
  FINAL_STATUS="failed"
  FINAL_NOTE="packaging_refused_empty_results"
fi
ls -la /workspace/serve/
export FINAL_STATUS FINAL_NOTE
python3 - <<'EOF'
import json, os
sp = '/workspace/serve/status.json'
cur = {}
if os.path.exists(sp):
    cur = json.load(open(sp))
if cur.get('status') not in ('failed',):
    cur.update({'status': os.environ['FINAL_STATUS'],
                'note': os.environ['FINAL_NOTE']})
    json.dump(cur, open(sp, 'w'))
print(open(sp).read())
EOF

echo "=== pipeline complete, keeping results server alive ==="
# The background serve.py (started at bootstrap) keeps the container alive.
# The VM fetches results.tar.gz via the proxy URL, then terminates this pod.
wait
