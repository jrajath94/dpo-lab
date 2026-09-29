"""Training diagnostics plots. Reads the raw logs, draws the curves.

Inputs:
  - <run_dir>/training_log.jsonl : TRL trainer.state.log_history
  - evals/margin_base.json, evals/margin_dpo.json : margin lists
Outputs PNGs into the given plots dir. matplotlib Agg backend, no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read_log(log_path: Path) -> list[dict]:
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in rows:
        if key in r and "step" in r:
            try:
                ys.append(float(r[key]))
                xs.append(float(r["step"]))
            except (TypeError, ValueError):
                continue
    return xs, ys


def plot_training_curves(run_dir: Path, plots_dir: Path) -> list[str]:
    """Loss, chosen/rejected implicit rewards, KL to reference. From TRL logs."""
    rows = _read_log(run_dir / "training_log.jsonl")
    plots_dir.mkdir(parents=True, exist_ok=True)
    made = []

    curves = [
        ("loss", "DPO loss vs step", "loss.png", "loss"),
        ("rewards/chosen", "Chosen implicit reward vs step",
         "rewards_chosen.png", "reward"),
        ("rewards/rejected", "Rejected implicit reward vs step",
         "rewards_rejected.png", "reward"),
        ("rewards/margins", "Reward margin vs step", "margins_train.png",
         "margin"),
        ("rewards/kl", "Mean KL to reference vs step", "kl.png", "KL"),
        ("grad_norm", "Gradient norm vs step", "grad_norm.png", "norm"),
    ]
    for key, title, fname, ylabel in curves:
        xs, ys = _series(rows, key)
        if not xs:
            continue
        fig, ax = plt.subplots()
        ax.plot(xs, ys)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=120)
        plt.close(fig)
        made.append(fname)

    # Combined chosen vs rejected rewards on one axis: the gap between the
    # two lines is the margin the loss is pushing apart.
    xc, yc = _series(rows, "rewards/chosen")
    xr, yr = _series(rows, "rewards/rejected")
    if xc and xr:
        fig, ax = plt.subplots()
        ax.plot(xc, yc, label="chosen")
        ax.plot(xr, yr, label="rejected")
        ax.set_title("Chosen vs rejected implicit reward")
        ax.set_xlabel("step")
        ax.set_ylabel("reward")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "rewards_chosen_vs_rejected.png", dpi=120)
        plt.close(fig)
        made.append("rewards_chosen_vs_rejected.png")
    return made


def plot_margin_histograms(evals_dir: Path, plots_dir: Path) -> list[str]:
    """Before/after margin histograms, one subplot each.

    The two metrics live on different scales by construction (base is a raw
    logprob difference, DPO is a beta-scaled implicit margin), so they are
    drawn on separate axes. The comparable numbers are the two accuracies
    in the subplot titles.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    made = []
    series = []
    for fname, label in (("margin_base.json", "base"),
                         ("margin_dpo.json", "dpo")):
        path = evals_dir / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        margins = data.get("margins", [])
        if margins:
            series.append((label, margins, data.get("accuracy")))
    if len(series) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (label, margins, acc) in zip(axes, series):
            ax.hist(margins, bins=40, alpha=0.7)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
            acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else "?"
            ax.set_title(f"{label}: acc={acc_str} (right of 0 = correct)")
            ax.set_xlabel("margin")
            ax.set_ylabel("pairs")
        fig.suptitle("Held-out preference agreement: base vs DPO")
        fig.tight_layout()
        fig.savefig(plots_dir / "margin_hist_base_vs_dpo.png", dpi=120)
        plt.close(fig)
        made.append("margin_hist_base_vs_dpo.png")
    return made


def plot_run_diagnostics(splits_dir: str, adapter_dir: str | None,
                         plots_out: str, config: dict,
                         beta: float = 0.1) -> list[str]:
    """Entry point used by evaluate.py --eval plots and 03_evaluate.sh.

    run_dir is derived from the adapter dir (plots sit next to the run).
    """
    run_dir = Path(adapter_dir) if adapter_dir else Path(".")
    evals_dir = run_dir / "evals"
    plots_dir = Path(plots_out)
    made = plot_training_curves(run_dir, plots_dir)
    made += plot_margin_histograms(evals_dir, plots_dir)
    print(f"plots written to {plots_dir}: {made}")
    return made
