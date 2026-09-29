#!/usr/bin/env python3
"""Independent verification gate for a dpo-lab run. The "nothing ships on
a single unverified pass" gate.

Re-derives every headline number from the raw artifacts:
  (a) split integrity: zero prompt-text overlap between train and each
      eval split; prompt_ids unique and correctly prefixed
  (b) number re-derivation: margin accuracies from per-pair margins,
      judge win rate + length splits from raw labels, Wilson CI from
      integer wins; all must match metrics.json / the stored summaries
      within 1e-9
  (c) training-log sanity: loss finite, decreasing (first vs last), no
      NaN, margins logged
  (d) config-hash check: config.yaml hashes to the recorded config_hash

Exit 0 prints VERIFY OK. ANY mismatch prints VERIFY FAIL lines and
exits 1.

CPU-only. No GPU, no network.

Usage:
  python3 scripts/verify_run.py --run-dir results/runs/beta_0.1 \
      [--splits-dir results/splits]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make the repo package importable (src layout, not installed).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpo_weekend.data import messages_to_text  # noqa: E402
from dpo_weekend.evaluate import length_controlled_win_rate  # noqa: E402
from dpo_weekend.utils import (  # noqa: E402
    load_jsonl,
    sha256_of_file,
    wilson_interval,
)

TOL = 1e-9

SPLITS = [
    ("train", "train.jsonl", "train-"),
    ("margin_eval", "margin_eval.jsonl", "margin_eval-"),
    ("generation", "generation.jsonl", "generation-"),
]

JUDGE_ORDERS = {"base_first", "dpo_first"}


class Verifier:
    """Collects check results. Every failure is recorded; nothing is
    raised early, so one run reports ALL problems."""

    def __init__(self):
        self.failures: list[str] = []
        self.passes = 0

    def check(self, cond: bool, msg: str) -> None:
        if cond:
            self.passes += 1
        else:
            self.failures.append(msg)

    def check_close(self, actual, expected, msg: str,
                    tol: float = TOL) -> None:
        """Float compare within tol; None only equals None."""
        if actual is None or expected is None:
            self.check(actual is None and expected is None,
                       f"{msg}: actual={actual}, expected={expected}")
            return
        self.check(abs(actual - expected) <= tol,
                   f"{msg}: actual={actual}, expected={expected}, "
                   f"diff={abs(actual - expected)}, tol={tol}")


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# (a) split integrity
# ---------------------------------------------------------------------------

def check_split_integrity(v: Verifier, splits_dir: Path) -> None:
    id_counts: Counter = Counter()
    texts: dict[str, set[str]] = {}
    for name, fname, prefix in SPLITS:
        path = splits_dir / fname
        if not path.is_file():
            v.check(False, f"splits/{fname} is missing")
            texts[name] = set()
            continue
        rows = load_jsonl(path)
        v.check(len(rows) > 0, f"splits/{fname} is empty")
        name_texts = set()
        for row in rows:
            pid = row.get("prompt_id")
            v.check(isinstance(pid, str) and pid,
                    f"splits/{fname}: row missing prompt_id")
            if isinstance(pid, str) and pid:
                v.check(pid.startswith(prefix),
                        f"splits/{fname}: prompt_id {pid!r} does not start "
                        f"with expected prefix {prefix!r}")
                id_counts[pid] += 1
            prompt = row.get("prompt")
            v.check(prompt is not None,
                    f"splits/{fname}: row {pid!r} missing prompt")
            if prompt is not None:
                name_texts.add(messages_to_text(prompt).strip())
        texts[name] = name_texts

    for pid, count in sorted(id_counts.items()):
        v.check(count == 1,
                f"duplicate prompt_id {pid!r} appears {count} times "
                f"across splits")

    # train must share zero prompt text with either eval split (and the
    # two eval splits must be disjoint too, or the evals double-count).
    for a, b in (("train", "margin_eval"),
                 ("train", "generation"),
                 ("margin_eval", "generation")):
        inter = texts[a] & texts[b]
        example = sorted(inter)[0][:80] if inter else ""
        v.check(not inter,
                f"prompt-text overlap between {a} and {b}: "
                f"{len(inter)} shared prompt(s), e.g. {example!r}")


# ---------------------------------------------------------------------------
# (b) number re-derivation
# ---------------------------------------------------------------------------

def _rederived_margin_accuracy(margins: list[float]) -> float:
    return sum(1 for m in margins if m > 0) / len(margins)


def check_margin_rederivation(v: Verifier, run_dir: Path,
                             metrics: dict) -> None:
    for fname, key in (("evals/margin_base.json", "margin_base_acc"),
                       ("evals/margin_dpo.json", "margin_dpo_acc")):
        path = run_dir / fname
        if not path.is_file():
            v.check(False, f"{fname} is missing")
            continue
        doc = _read_json(path)
        margins = doc.get("margins") or []
        v.check(len(margins) > 0, f"{fname} has no margins")
        if not margins:
            continue
        acc = _rederived_margin_accuracy(margins)
        v.check_close(acc, doc.get("accuracy"),
                      f"{fname}: stored accuracy != re-derived "
                      f"from per-pair margins")
        v.check(doc.get("n") == len(margins),
                f"{fname}: n={doc.get('n')} != "
                f"len(margins)={len(margins)}")
        # The base margin has a length-normalized twin that the assembler
        # prefers (raw logprob sums confound length). Its per-pair
        # normalized margins are not persisted, so this is a consistency
        # check between the two files, not a full re-derivation: the raw
        # accuracy above IS fully re-derived from per-pair margins.
        if "accuracy_len_norm" in doc:
            v.check_close(doc["accuracy_len_norm"], metrics.get(key),
                          f"{fname}: metrics.json {key} != stored "
                          f"accuracy_len_norm")
            v.check(metrics.get("margin_base_metric") == "accuracy_len_norm",
                    f"{fname}: metrics.json margin_base_metric should be "
                    f"'accuracy_len_norm', got "
                    f"{metrics.get('margin_base_metric')!r}")
        else:
            v.check_close(acc, metrics.get(key),
                          f"{fname}: metrics.json {key} != re-derived "
                          f"accuracy")


def check_judge_rederivation(v: Verifier, run_dir: Path,
                             metrics: dict) -> None:
    path = run_dir / "evals" / "judge_labels.json"
    if not path.is_file():
        v.check(False, "judge_labels.json is missing")
        return
    judge = _read_json(path)
    labels = judge.get("labels") or []
    v.check(len(labels) > 0, "judge_labels.json has no labels")
    if not labels:
        return

    re = length_controlled_win_rate(labels)
    for k in ("overall", "when_dpo_longer", "when_base_longer"):
        v.check_close(re[k], judge.get(k),
                      f"judge_labels.json: stored {k} != re-derived "
                      f"from labels")
    for k in ("n", "n_ties", "n_parse_errors"):
        v.check(re[k] == judge.get(k),
                f"judge_labels.json: stored {k}={judge.get(k)} != "
                f"re-derived {re[k]}")

    # completeness: every prompt judged under both orders
    orders: dict[str, set[str]] = defaultdict(set)
    for lab in labels:
        orders[str(lab.get("prompt_id"))].add(str(lab.get("order")))
    for pid in sorted(orders):
        v.check(orders[pid] == JUDGE_ORDERS,
                f"judge_labels.json: prompt {pid!r} has orders "
                f"{sorted(orders[pid])} (expected both "
                f"{sorted(JUDGE_ORDERS)})")
    v.check(len(orders) == judge.get("n_prompts"),
            f"judge_labels.json: n_prompts={judge.get('n_prompts')} != "
            f"distinct prompt_ids={len(orders)}")

    # metrics.json must mirror the stored summaries exactly. The headline
    # win rate is STRICT (integer dpo-wins / n labels), re-derived here
    # from the raw labels so it cannot drift from the Wilson CI.
    strict_rate = (sum(1 for lab in labels
                       if lab.get("winner") == "dpo") / len(labels))
    v.check_close(strict_rate, metrics.get("judge_win_rate"),
                  "metrics.json judge_win_rate != strict dpo-wins / n "
                  "re-derived from judge labels")
    v.check(judge.get("n") == metrics.get("judge_n"),
            f"metrics.json judge_n={metrics.get('judge_n')} != "
            f"judge_labels.json n={judge.get('n')}")
    v.check(judge.get("n_prompts") == metrics.get("judge_n_prompts"),
            f"metrics.json judge_n_prompts={metrics.get('judge_n_prompts')} "
            f"!= judge_labels.json n_prompts={judge.get('n_prompts')}")
    for k in ("overall", "when_dpo_longer", "when_base_longer"):
        v.check_close(metrics.get("judge_length_splits", {}).get(k),
                      re[k],
                      f"metrics.json judge_length_splits.{k} != "
                      f"re-derived from labels")

    wins = sum(1 for lab in labels if lab.get("winner") == "dpo")
    lo, hi = wilson_interval(wins, len(labels))
    stored = metrics.get("judge_wilson_95")
    v.check(isinstance(stored, (list, tuple)) and len(stored) == 2,
            f"metrics.json judge_wilson_95 must be [lo, hi], "
            f"got {stored!r}")
    if isinstance(stored, (list, tuple)) and len(stored) == 2:
        v.check_close(lo, stored[0],
                      "metrics.json judge_wilson_95[0] != re-derived "
                      "Wilson lower bound")
        v.check_close(hi, stored[1],
                      "metrics.json judge_wilson_95[1] != re-derived "
                      "Wilson upper bound")

    # Decisive-only rate (ties excluded), re-derived the same way.
    decisive = [lab for lab in labels if lab.get("winner") != "tie"]
    d_wins = sum(1 for lab in decisive if lab.get("winner") == "dpo")
    d_lo, d_hi = wilson_interval(d_wins, len(decisive))
    v.check_close(d_wins / len(decisive),
                  metrics.get("judge_decisive_win_rate"),
                  "metrics.json judge_decisive_win_rate != re-derived "
                  "dpo-wins / decisive labels")
    v.check(metrics.get("judge_decisive_n") == len(decisive),
            "metrics.json judge_decisive_n != re-derived decisive count")
    d_stored = metrics.get("judge_decisive_wilson_95")
    if isinstance(d_stored, (list, tuple)) and len(d_stored) == 2:
        v.check_close(d_lo, d_stored[0],
                      "metrics.json judge_decisive_wilson_95[0] mismatch")
        v.check_close(d_hi, d_stored[1],
                      "metrics.json judge_decisive_wilson_95[1] mismatch")


# ---------------------------------------------------------------------------
# (c) training-log sanity
# ---------------------------------------------------------------------------

def check_training_log(v: Verifier, run_dir: Path) -> None:
    path = run_dir / "training_log.jsonl"
    if not path.is_file():
        v.check(False, "training_log.jsonl is missing")
        return
    rows = load_jsonl(path)
    loss_rows = [r for r in rows if "loss" in r]
    v.check(len(loss_rows) > 0,
            "training_log.jsonl: no entries with loss")
    for r in loss_rows:
        loss = r["loss"]
        v.check(isinstance(loss, (int, float)) and math.isfinite(loss),
                f"training_log.jsonl: non-finite loss {loss!r} "
                f"at step {r.get('step')}")
    margin_rows = [r for r in rows if "rewards/margins" in r]
    v.check(len(margin_rows) > 0,
            "training_log.jsonl: no rewards/margins entries logged")
    for r in margin_rows:
        m = r["rewards/margins"]
        v.check(isinstance(m, (int, float)) and math.isfinite(m),
                f"training_log.jsonl: non-finite rewards/margins {m!r} "
                f"at step {r.get('step')}")
    if loss_rows:
        first, last = loss_rows[0]["loss"], loss_rows[-1]["loss"]
        v.check(last < first,
                f"training_log.jsonl: loss not decreasing "
                f"(first={first}, last={last})")


# ---------------------------------------------------------------------------
# (d) config-hash check
# ---------------------------------------------------------------------------

def check_config_hash(v: Verifier, run_dir: Path, metrics: dict) -> None:
    path = run_dir / "config.yaml"
    if not path.is_file():
        v.check(False, "config.yaml is missing")
        return
    actual = sha256_of_file(path)
    recorded = metrics.get("config_hash")
    v.check(actual == recorded,
            f"config_hash mismatch: config.yaml hashes to "
            f"{actual[:16]}... but metrics.json records "
            f"{str(recorded)[:16]}...")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def verify_run(run_dir: Path, splits_dir: Path) -> Verifier:
    v = Verifier()

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        v.check(False, "metrics.json is missing: run "
                       "scripts/assemble_results.py first")
        return v
    metrics = _read_json(metrics_path)

    check_split_integrity(v, splits_dir)
    check_margin_rederivation(v, run_dir, metrics)
    check_judge_rederivation(v, run_dir, metrics)
    check_training_log(v, run_dir)
    check_config_hash(v, run_dir, metrics)
    return v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independent verification gate for a dpo-lab run.")
    parser.add_argument("--run-dir", required=True,
                        help="Run dir, e.g. results/runs/beta_0.1.")
    parser.add_argument("--splits-dir", default=None,
                        help="Splits dir (default: <run-dir>/../../splits).")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    splits_dir = (Path(args.splits_dir) if args.splits_dir
                  else run_dir.parent.parent / "splits")

    v = verify_run(run_dir, splits_dir)
    if v.failures:
        print("VERIFY FAIL:", file=sys.stderr)
        for msg in v.failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    print(f"VERIFY OK: {v.passes} checks passed for {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
