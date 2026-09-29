#!/usr/bin/env python3
"""Assemble a dpo-lab run dir into metrics.json + the honest one-line claim.

Reads results/runs/<run>/ (config.yaml, training_log.jsonl,
judge_labels.json, evals/*) plus results/splits/, validates that every
expected artifact exists, and writes metrics.json. Fails loudly, listing
every missing file, instead of writing a partial metrics file.

CPU-only. No GPU, no network.

Usage:
  python3 scripts/assemble_results.py \
      --run-dir results/runs/beta_0.1 \
      [--splits-dir results/splits] [--out results/runs/beta_0.1/metrics.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Make the repo package importable (src layout, not installed).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpo_weekend.evaluate import length_controlled_win_rate  # noqa: E402
from dpo_weekend.utils import (  # noqa: E402
    load_jsonl,
    save_json,
    sha256_of_file,
    wilson_interval,
)

RUN_FILES = [
    "config.yaml",
    "training_log.jsonl",
]
EVAL_FILES = [
    "evals/margin_base.json",
    "evals/margin_dpo.json",
    "evals/gens_base.jsonl",
    "evals/gens_dpo.jsonl",
    "evals/judge_labels.json",
]
# The MMLU regression guard is optional: when the pod cannot run it
# (run 6: lm-eval 0.4.13 needs Python >= 3.13, pod has 3.11), assembly
# proceeds and records the absence honestly instead of failing.
MMLU_FILES = [
    "evals/mmlu_base.json",
    "evals/mmlu_dpo.json",
]
SPLIT_FILES = [
    "train.jsonl",
    "margin_eval.jsonl",
    "generation.jsonl",
    "manifest.json",
]
PLOTS_DIR = "plots"  # TRD: plots live in <run-dir>/plots/, not evals/


class MissingArtifactsError(Exception):
    """Raised when the run dir is missing expected artifacts."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__("missing expected artifacts:\n  - "
                         + "\n  - ".join(missing))


def find_missing(run_dir: Path, splits_dir: Path) -> list[str]:
    """Every expected file that does not exist, as display paths."""
    missing = []
    for rel in RUN_FILES + EVAL_FILES:
        if not (run_dir / rel).is_file():
            missing.append(rel)
    for rel in SPLIT_FILES:
        if not (splits_dir / rel).is_file():
            missing.append(f"splits/{rel}")
    plots = run_dir / PLOTS_DIR
    if not (plots.is_dir() and any(plots.glob("*.png"))):
        missing.append(f"{PLOTS_DIR}/*.png (no .png plots found)")
    return missing


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _judge_metrics(judge: dict) -> dict:
    """Judge numbers from judge_labels.json.

    The headline win rate is the STRICT fraction: integer dpo-wins over
    n labels, ties counting as non-wins. This is the only definition
    under which the reported 95% Wilson CI is exactly valid, and it is
    the plain-English reading of "won X of N comparisons". The
    length_controlled_win_rate splits keep the AlpacaEval convention
    (ties as half wins) as a diagnostic; they are labeled as splits,
    not the headline.
    """
    labels = judge.get("labels") or []
    if not labels:
        raise ValueError("judge_labels.json has no labels")
    splits = length_controlled_win_rate(labels)
    wins = sum(1 for lab in labels if lab.get("winner") == "dpo")
    n = len(labels)
    lo, hi = wilson_interval(wins, n)
    decisive = [lab for lab in labels if lab.get("winner") != "tie"]
    d_wins = sum(1 for lab in decisive if lab.get("winner") == "dpo")
    d_lo, d_hi = wilson_interval(d_wins, len(decisive))
    return {
        "judge_win_rate": wins / n,
        "judge_n": judge.get("n"),
        "judge_n_prompts": judge.get("n_prompts"),
        "judge_ties": judge.get("n_ties"),
        "judge_parse_errors": judge.get("n_parse_errors"),
        "judge_wilson_95": [lo, hi],
        "judge_wilson_note": ("Wilson CI on integer dpo-wins / n labels; "
                              "ties count as non-wins, matching "
                              "judge_win_rate exactly"),
        # Decisive comparisons only (ties excluded): the natural framing
        # for "is DPO better than base".
        "judge_decisive_win_rate": d_wins / len(decisive),
        "judge_decisive_n": len(decisive),
        "judge_decisive_wilson_95": [d_lo, d_hi],
        "judge_length_splits": {
            "overall": splits["overall"],
            "when_dpo_longer": splits["when_dpo_longer"],
            "when_base_longer": splits["when_base_longer"],
        },
        "judge_model": judge.get("judge_model"),
    }


def _one_line_claim(m: dict, mmlu_limit) -> str:
    lo, hi = m["judge_wilson_95"]
    d_lo, d_hi = m["judge_decisive_wilson_95"]
    if m.get("mmlu_status") == "measured":
        limit = f"-{mmlu_limit}" if mmlu_limit else ""
        mmlu_part = (f"MMLU{limit}: {m['mmlu_base_acc']:.3f} -> "
                     f"{m['mmlu_dpo_acc']:.3f} "
                     f"(delta {m['mmlu_delta']:+.3f})")
    else:
        mmlu_part = "MMLU: not measured (lm-eval 0.4.13 requires Python>=3.13; pod runs 3.11)"
    return (
        f"DPO fine-tune {m['model_name']} "
        f"(beta={m['beta']}, seed={m['seed']}): "
        f"held-out margin accuracy {m['margin_base_acc']:.3f} "
        f"({m.get('margin_base_metric', 'accuracy')} base) -> "
        f"{m['margin_dpo_acc']:.3f} (delta {m['margin_delta']:+.3f}, "
        f"n={m['margin_n']}); "
        f"blinded A/B judge {m['judge_model']}: dpo win rate "
        f"{m['judge_win_rate']:.3f} (strict: ties count as non-wins), "
        f"95% Wilson CI [{lo:.3f}, {hi:.3f}] "
        f"({m['judge_n_prompts']} prompts x 2 orders, {m['judge_ties']} ties); "
        f"excluding ties: {m['judge_decisive_win_rate']:.3f} "
        f"[{d_lo:.3f}, {d_hi:.3f}] (n={m['judge_decisive_n']}); "
        f"{mmlu_part}. "
        f"Numbers re-derived from raw artifacts by scripts/verify_run.py."
    )


def assemble_run(run_dir: Path, splits_dir: Path,
                 out_path: Path) -> dict:
    """Validate, assemble, write metrics.json, print the honest claim."""
    missing = find_missing(run_dir, splits_dir)
    if missing:
        raise MissingArtifactsError(missing)

    # Start from train-time metrics.json when present so train_pairs,
    # global_step, and friends survive assembly.
    metrics: dict = {}
    existing = run_dir / "metrics.json"
    if existing.is_file():
        metrics = _read_json(existing)

    with open(run_dir / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    margin_base = _read_json(run_dir / "evals" / "margin_base.json")
    margin_dpo = _read_json(run_dir / "evals" / "margin_dpo.json")
    mmlu_base = _read_json(run_dir / "evals" / "mmlu_base.json") \
        if (run_dir / "evals" / "mmlu_base.json").is_file() else None
    mmlu_dpo = _read_json(run_dir / "evals" / "mmlu_dpo.json") \
        if (run_dir / "evals" / "mmlu_dpo.json").is_file() else None
    judge = _read_json(run_dir / "evals" / "judge_labels.json")

    # Sanity: the headline numbers must exist and be non-degenerate.
    for doc, name in ((margin_base, "margin_base.json"),
                      (margin_dpo, "margin_dpo.json")):
        if not doc.get("margins"):
            raise ValueError(f"{name} has no margins")

    # Base margin accuracy: prefer the length-normalized twin when the
    # eval wrote it (raw sums confound length: chosen responses skew
    # longer, so the raw base number understates agreement).
    margin_base_acc = margin_base.get("accuracy_len_norm",
                                      margin_base["accuracy"])
    margin_base_metric = ("accuracy_len_norm"
                          if "accuracy_len_norm" in margin_base
                          else "accuracy")
    mmlu_measured = bool(mmlu_base and mmlu_dpo)

    metrics.update({
        "margin_base_acc": margin_base_acc,
        "margin_base_metric": margin_base_metric,
        "margin_dpo_acc": margin_dpo["accuracy"],
        "margin_delta": margin_dpo["accuracy"] - margin_base_acc,
        "margin_n": margin_base["n"],
        # MMLU is an optional regression guard. When the pod could not run
        # it, the fields are null and the reason is recorded; the claim
        # line says "not measured" instead of printing numbers.
        "mmlu_base_acc": mmlu_base["acc"] if mmlu_measured else None,
        "mmlu_dpo_acc": mmlu_dpo["acc"] if mmlu_measured else None,
        "mmlu_delta": ((mmlu_dpo["acc"] - mmlu_base["acc"])
                       if mmlu_measured else None),
        "mmlu_status": "measured" if mmlu_measured else "not_measured",
        "mmlu_note": (None if mmlu_measured
                      else "lm-eval 0.4.13 requires Python >= 3.13 "
                           "(PEP 728 TypedDict extra_items); pod runs "
                           "Python 3.11, so the guard could not import"),
        "config_hash": sha256_of_file(run_dir / "config.yaml"),
        "model_name": config["model"]["name"],
        "model_revision": config["model"].get("revision"),
        "seed": config.get("seed"),
        "beta": float(config["dpo"]["beta"]),
    })
    metrics.update(_judge_metrics(judge))

    save_json(metrics, out_path)
    mmlu_limit = mmlu_base.get("limit") if mmlu_base else None
    print(_one_line_claim(metrics, mmlu_limit))
    print(f"wrote {out_path}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a dpo-lab run dir into metrics.json.")
    parser.add_argument("--run-dir", required=True,
                        help="Run dir, e.g. results/runs/beta_0.1.")
    parser.add_argument("--splits-dir", default=None,
                        help="Splits dir (default: <run-dir>/../../splits).")
    parser.add_argument("--out", default=None,
                        help="metrics.json path "
                             "(default: <run-dir>/metrics.json).")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    splits_dir = (Path(args.splits_dir) if args.splits_dir
                  else run_dir.parent.parent / "splits")
    out_path = Path(args.out) if args.out else run_dir / "metrics.json"

    try:
        assemble_run(run_dir, splits_dir, out_path)
    except MissingArtifactsError as e:
        print(f"ERROR: {run_dir} is missing "
              f"{len(e.missing)} expected artifact(s):", file=sys.stderr)
        for rel in e.missing:
            print(f"  - {rel}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
