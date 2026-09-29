#!/usr/bin/env python3
"""Blinded A/B judge harness, run HERE on the VM.

The GPU pod trains the model and generates answers (gens_base.jsonl,
gens_dpo.jsonl); this script judges them with a fixed external judge.
The OpenRouter credential never leaves this box: auth goes through the
vault surrogate in /opt/hatch/skills/skill-creator/bin/dynamic_credentials.py.
No raw key is ever requested, printed, logged, or persisted.

Protocol (identical to dpo_weekend.evaluate.run_judge):
  - for each prompt, two order-swapped blinded A/B calls
    (A=base/B=dpo, then A=dpo/B=base), temperature 0.0, max_tokens 200
  - A/B winners mapped back to base/dpo/tie
  - each label records dpo_longer, parse_error, judge_model, reason
  - judge_labels.json uses the same summary schema as run_judge:
    overall / when_dpo_longer / when_base_longer (ties = half wins),
    n, n_ties, n_parse_errors, judge_model, judge_template, n_prompts,
    labels
  - polite 0.5 s sleep between prompts; resume-friendly: prompt_ids
    already fully labeled in an existing --out file are skipped, and a
    judge-model/template mismatch against existing labels is a loud error.

Usage:
  python3 scripts/judge_on_vm.py \
      --base-gens results/runs/beta_0.1/gens_base.jsonl \
      --dpo-gens  results/runs/beta_0.1/gens_dpo.jsonl \
      --config    configs/dpo_qwen25_15b_qlora.yaml \
      --out       results/runs/beta_0.1/evals/judge_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

# Make the repo package importable (src layout, not installed).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpo_weekend.evaluate import (  # noqa: E402
    JUDGE_SYSTEM,
    JUDGE_TEMPLATE,
    length_controlled_win_rate,
    parse_judge_content,  # single source of truth; no local copy
)
from dpo_weekend.utils import load_jsonl, save_json  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 200
POLITE_SLEEP_S = 0.5

ORDERS = (("base_first", "base", "dpo"), ("dpo_first", "dpo", "base"))


# ---------------------------------------------------------------------------
# Pure parts (covered by tests/test_judge_vm.py).
# ---------------------------------------------------------------------------
# NOTE: parse_judge_content is imported from dpo_weekend.evaluate (single
# source of truth) so the VM harness can never drift from the pod-side
# parser. The name stays available as judge_on_vm.parse_judge_content.


def map_winner(winner_ab: str, a_is: str, b_is: str) -> str:
    """Map an A/B winner back to base/dpo/tie.

    a_is / b_is name which model was shown as A and B
    ("base" and "dpo" in some order). Same mapping as run_judge.
    """
    if winner_ab == "TIE":
        return "tie"
    shown = {"A": a_is, "B": b_is}.get(winner_ab)
    return "dpo" if shown == "dpo" else "base"


# ---------------------------------------------------------------------------
# Network part: one blinded A/B call through OpenRouter, auth via the
# vault surrogate (never a raw key).
# ---------------------------------------------------------------------------

def judge_call(judge_model: str, judge_template: str, prompt: str,
               a: str, b: str, timeout: int = 120) -> dict:
    """One blinded A/B call. Returns the parsed label."""
    from dynamic_credentials import add_surrogate_to_request

    body = json.dumps({
        "model": judge_model,
        "messages": [
            {"role": "system",
             "content": JUDGE_SYSTEM},
            {"role": "user",
             "content": judge_template.format(prompt=prompt, a=a, b=b)},
        ],
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
    }).encode("utf-8")

    req = urllib.request.Request(OPENROUTER_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    # The helper attaches the surrogate as a bearer header; no raw key
    # is ever visible here.
    add_surrogate_to_request(req, "custom.openrouter",
                             allowed_hosts=("openrouter.ai",))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    return parse_judge_content(content)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def _already_done(out_path: Path) -> dict[str, set[str]]:
    """prompt_id -> set of order names already labeled in an existing file."""
    done: dict[str, set[str]] = {}
    if not out_path.exists():
        return done
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return done
    for lab in existing.get("labels", []):
        done.setdefault(str(lab.get("prompt_id")), set()).add(
            str(lab.get("order")))
    return done


def _check_judge_consistency(existing_path: Path, judge_model: str,
                             judge_template: str) -> None:
    """A judge-model or template change invalidates old numbers: fail
    loudly instead of mixing labels judged under different protocols."""
    if not existing_path.exists():
        return
    try:
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return
    old_model = existing.get("judge_model")
    if old_model and old_model != judge_model:
        raise RuntimeError(
            f"{existing_path} was judged with {old_model!r}, but this run "
            f"uses {judge_model!r}. Judge change invalidates old numbers; "
            "delete the file or match the model.")
    old_template = existing.get("judge_template")
    if old_template and old_template != judge_template:
        raise RuntimeError(
            f"{existing_path} was judged with a different judge template. "
            "Judge change invalidates old numbers; delete the file or "
            "match the template.")


def run_judge(base_gens: str, dpo_gens: str, out_path: str,
              judge_model: str, judge_template: str = JUDGE_TEMPLATE,
              sleep_s: float = POLITE_SLEEP_S) -> dict:
    """Blinded A/B judging with order swapping. Same output schema as
    dpo_weekend.evaluate.run_judge."""
    out = Path(out_path)
    _check_judge_consistency(out, judge_model, judge_template)
    done = _already_done(out)

    base = {r["prompt_id"]: r for r in load_jsonl(base_gens)}
    dpo = {r["prompt_id"]: r for r in load_jsonl(dpo_gens)}
    common = [pid for pid in base if pid in dpo]
    if not common:
        raise ValueError("no shared prompt_ids between generation files")

    labels = []
    for pid in common:
        seen = done.get(pid, set())
        if len(seen) >= 2:
            continue  # resume: both orders already labeled
        b, d = base[pid], dpo[pid]
        for order_name, a_is, b_is in ORDERS:
            if order_name in seen:
                continue  # resume: this order already labeled
            a_text = b["text"] if a_is == "base" else d["text"]
            b_text = d["text"] if b_is == "dpo" else b["text"]
            lab = judge_call(judge_model, judge_template,
                             b["prompt"], a_text, b_text)
            lab.update({"prompt_id": pid, "order": order_name,
                        "a_is": a_is, "b_is": b_is})
            labels.append(lab)
        time.sleep(sleep_s)  # be polite to the API

    mapped = []
    for lab in labels:
        winner = map_winner(lab["winner_ab"], lab["a_is"], lab["b_is"])
        b_row = base[lab["prompt_id"]]
        d_row = dpo[lab["prompt_id"]]
        mapped.append({
            "prompt_id": lab["prompt_id"],
            "order": lab["order"],
            "winner": winner,
            "dpo_longer": len(d_row["text"]) > len(b_row["text"]),
            "parse_error": lab["parse_error"],
            "judge_model": judge_model,
            "reason": lab.get("reason", ""),
        })

    summary = length_controlled_win_rate(mapped)
    summary["judge_model"] = judge_model
    summary["judge_template"] = judge_template
    summary["n_prompts"] = len(common)
    summary["labels"] = mapped
    save_json(summary, out)
    print(f"judge [{judge_model}]: overall win rate "
          f"{summary['overall']:.3f} over {summary['n']} comparisons "
          f"({summary['n_prompts']} prompts), {summary['n_ties']} ties, "
          f"{summary['n_parse_errors']} parse errors")
    return summary


def _resolve_judge_settings(args, config: dict) -> tuple[str, str]:
    """judge_model: --judge-model > JUDGE_MODEL env > config eval.judge_model.
    judge_template: --judge-template-file > config eval.judge_template >
    the canonical template in dpo_weekend.evaluate."""
    import os
    eval_cfg = config.get("eval", {})
    judge_model = (args.judge_model
                   or os.environ.get("JUDGE_MODEL")
                   or eval_cfg.get("judge_model")
                   or "openai/gpt-4o-mini")
    if args.judge_template_file:
        judge_template = Path(args.judge_template_file).read_text(
            encoding="utf-8")
    else:
        judge_template = eval_cfg.get("judge_template") or JUDGE_TEMPLATE
    return judge_model, judge_template


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Blinded A/B judge for dpo-lab generations (VM side).")
    parser.add_argument("--base-gens", required=True,
                        help="Base-model generations JSONL.")
    parser.add_argument("--dpo-gens", required=True,
                        help="DPO-model generations JSONL.")
    parser.add_argument("--out", required=True,
                        help="Output judge_labels.json (resume-friendly).")
    parser.add_argument("--config", required=True,
                        help="Training YAML config (provides eval.judge_model "
                             "and optionally eval.judge_template).")
    parser.add_argument("--judge-model", default=None,
                        help="Override the judge model id.")
    parser.add_argument("--judge-template-file", default=None,
                        help="Override the judge template (file path).")
    parser.add_argument("--sleep", type=float, default=POLITE_SLEEP_S,
                        help="Seconds between API calls (default 0.5).")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    judge_model, judge_template = _resolve_judge_settings(args, config)

    # The surrogate helper must be importable before any network happens.
    sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
    run_judge(args.base_gens, args.dpo_gens, args.out, judge_model,
              judge_template, sleep_s=args.sleep)


if __name__ == "__main__":
    main()
