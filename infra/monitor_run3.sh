#!/bin/bash
# Run 3 background monitor: poll status.json + bootstrap.log tail every 15 min.
# Exits 0 with "TERMINAL:<status>" when status.json is done/failed,
# or after MAX_POLLS (~11h) with TIMEOUT.
LAB="$HOME/workspace/dpo-lab"
LOG="$LAB/infra/monitor_run3.log"
SLEEP_S=900
MAX_POLLS=44  # 44*15min = 11h
i=0
echo "$(date -u +%FT%TZ) monitor started" >> "$LOG"
while [ $i -lt $MAX_POLLS ]; do
  i=$((i+1))
  TS="$(date -u +%FT%TZ)"
  ST="$(python3 "$LAB/infra/pod_status.py" status.json 2>&1 | head -c 1500)"
  echo "[$TS poll $i] status.json: $ST" >> "$LOG"
  # terminal check: look for "done" or "failed" as the status value
  if echo "$ST" | grep -qE '"status"[[:space:]]*:[[:space:]]*"(done|failed)"'; then
    FINAL="$(echo "$ST" | grep -oE '"status"[[:space:]]*:[[:space:]]*"[a-z_]+"' | head -1)"
    echo "[$TS] TERMINAL: $FINAL" >> "$LOG"
    echo "TERMINAL:$FINAL after $i polls"
    exit 0
  fi
  # capture log tail (last 3000 chars) for phase tracking
  TAIL="$(python3 "$LAB/infra/pod_status.py" bootstrap.log 2>&1 | tail -c 3000)"
  echo "[$TS poll $i] bootstrap.log tail:" >> "$LOG"
  echo "$TAIL" >> "$LOG"
  echo "[$TS] --- end poll $i ---" >> "$LOG"
  sleep $SLEEP_S
done
echo "TIMEOUT after $MAX_POLLS polls" | tee -a "$LOG"
exit 2
