"""Plots smoke tests: the diagnostics must run headless on CPU and
produce real PNG files from synthetic logs. No network, no GPU."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from dpo_weekend.plots import (plot_margin_histograms,  # noqa: E402
                               plot_training_curves)


def _write_log(path: Path, n: int = 60) -> None:
    rows = [{"step": i, "loss": 0.7 - 0.005 * i,
             "rewards/chosen": -0.1 + 0.01 * i,
             "rewards/rejected": -0.2 + 0.005 * i,
             "rewards/margins": 0.1 + 0.005 * i,
             "rewards/kl": 0.02 + 0.001 * i,
             "grad_norm": 1.0}
            for i in range(n)]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_plot_training_curves_writes_pngs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_log(run_dir / "training_log.jsonl")
    plots_dir = tmp_path / "plots"
    made = plot_training_curves(run_dir, plots_dir)
    assert "loss.png" in made
    assert "kl.png" in made
    assert "rewards_chosen_vs_rejected.png" in made
    for fname in made:
        p = plots_dir / fname
        assert p.exists() and p.stat().st_size > 1000  # a real image


def test_plot_training_curves_skips_missing_keys(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "training_log.jsonl", "w",
              encoding="utf-8") as f:
        f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")
    made = plot_training_curves(run_dir, tmp_path / "plots")
    assert made == ["loss.png"]


def test_plot_margin_histograms_writes_comparison(tmp_path):
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "margin_base.json").write_text(
        json.dumps({"margins": [0.1, -0.2, 0.3] * 20, "accuracy": 0.66}),
        encoding="utf-8")
    (evals / "margin_dpo.json").write_text(
        json.dumps({"margins": [0.2, 0.4, -0.1] * 20, "accuracy": 0.8}),
        encoding="utf-8")
    plots_dir = tmp_path / "plots"
    made = plot_margin_histograms(evals, plots_dir)
    assert made == ["margin_hist_base_vs_dpo.png"]
    p = plots_dir / made[0]
    assert p.exists() and p.stat().st_size > 1000


def test_plot_margin_histograms_missing_files_is_empty(tmp_path):
    made = plot_margin_histograms(tmp_path / "nope", tmp_path / "plots")
    assert made == []
