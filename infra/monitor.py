#!/usr/bin/env python3
"""In-pod training watchdog for the dpo-lab run.

Watches results/runs/beta_0.1/training_log.jsonl and enforces the two stop
conditions:
  1. NaN loss -> stop training immediately.
  2. Margin flat at zero while the loss falls -> stop training.
Also writes /workspace/serve/status.json every poll so the VM can watch too.
On a stop condition it kills the training process and records the failure.
"""
import json
import math
import os
import subprocess
import time

LOG_PATH = "/workspace/dpo-lab/results/runs/beta_0.1/training_log.jsonl"
STATUS_PATH = "/workspace/serve/status.json"
POLL_SECONDS = 60
FLAT_WINDOW = 20          # consecutive margin readings to judge flatness
MARGIN_EPS = 0.02         # |margin| below this counts as "at zero"
MIN_LOSS_DROP = 0.10      # loss must have fallen this much for flat-margin stop


def write_status(payload):
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, STATUS_PATH)


def kill_training():
    try:
        subprocess.run(["pkill", "-f", "dpo_weekend.train"],
                       timeout=10, check=False)
    except Exception:
        pass
    time.sleep(10)
    try:
        subprocess.run(["pkill", "-9", "-f", "dpo_weekend.train"],
                       timeout=10, check=False)
    except Exception:
        pass


def read_rows():
    rows = []
    try:
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "loss" in r:
                    rows.append(r)
    except FileNotFoundError:
        pass
    return rows


def main():
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    write_status({"status": "starting", "steps": 0})
    while True:
        time.sleep(POLL_SECONDS)
        rows = read_rows()
        if not rows:
            write_status({"status": "waiting_for_log", "steps": 0})
            continue
        losses = [r["loss"] for r in rows]
        margins = [r.get("rewards/margins") for r in rows]
        status = {
            "status": "training",
            "steps": len(rows),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "margin_last": margins[-1],
        }
        if any(isinstance(x, float) and math.isnan(x) for x in losses):
            write_status({"status": "failed", "reason": "nan_loss",
                          "steps": len(rows), "loss_last": "NaN"})
            kill_training()
            return
        recent_margins = [m for m in margins[-FLAT_WINDOW:]
                          if isinstance(m, (int, float))]
        loss_drop = losses[0] - losses[-1]
        if (len(recent_margins) >= FLAT_WINDOW
                and all(abs(m) < MARGIN_EPS for m in recent_margins)
                and loss_drop > MIN_LOSS_DROP):
            write_status({"status": "failed",
                          "reason": "flat_margin_while_loss_falls",
                          "steps": len(rows),
                          "loss_first": losses[0], "loss_last": losses[-1],
                          "margin_last": margins[-1]})
            kill_training()
            return
        write_status(status)


if __name__ == "__main__":
    main()
