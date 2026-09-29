"""Tests for scripts/assemble_results.py.

Hand-built synthetic run dirs in tmp_path; no GPU, no network.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

from dpo_weekend.evaluate import length_controlled_win_rate
from dpo_weekend.utils import sha256_of_file, wilson_interval

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REV = "abc123def456"
JUDGE_MODEL = "openai/gpt-4o-mini"


def _msgs(text):
    return [{"role": "user", "content": text}]


def _judge_summary():
    labels = [
        {"prompt_id": "generation-0000", "order": "base_first",
         "winner": "dpo", "dpo_longer": True, "parse_error": False,
         "judge_model": JUDGE_MODEL, "reason": "clearer"},
        {"prompt_id": "generation-0000", "order": "dpo_first",
         "winner": "dpo", "dpo_longer": True, "parse_error": False,
         "judge_model": JUDGE_MODEL, "reason": "clearer"},
        {"prompt_id": "generation-0001", "order": "base_first",
         "winner": "tie", "dpo_longer": False, "parse_error": False,
         "judge_model": JUDGE_MODEL, "reason": "equal"},
        {"prompt_id": "generation-0001", "order": "dpo_first",
         "winner": "base", "dpo_longer": False, "parse_error": False,
         "judge_model": JUDGE_MODEL, "reason": "base better"},
        {"prompt_id": "generation-0002", "order": "base_first",
         "winner": "tie", "dpo_longer": True, "parse_error": False,
         "judge_model": JUDGE_MODEL, "reason": "equal"},
        {"prompt_id": "generation-0002", "order": "dpo_first",
         "winner": "tie", "dpo_longer": False, "parse_error": False,
         "judge_model": JUDGE_MODEL, "reason": "equal"},
    ]
    summary = length_controlled_win_rate(labels)
    summary.update({
        "judge_model": JUDGE_MODEL,
        "judge_template": "CANONICAL TEMPLATE",
        "n_prompts": 3,
        "labels": labels,
    })
    return summary


def build_run(tmp_path):
    """Create a complete synthetic run dir + splits dir. Returns paths."""
    run_dir = tmp_path / "results" / "runs" / "beta_0.1"
    splits_dir = tmp_path / "results" / "splits"
    evals_dir = run_dir / "evals"
    evals_dir.mkdir(parents=True)
    (run_dir / "plots").mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    config = {
        "model": {"name": MODEL_NAME, "revision": MODEL_REV,
                  "trust_remote_code": False},
        "data": {"dataset": "synthetic", "train_pairs": 12,
                 "margin_eval_pairs": 4, "generation_prompts": 3,
                 "max_seq_length": 1024, "seed": 42},
        "dpo": {"beta": 0.1, "learning_rate": 1e-4, "per_device_batch": 2,
                "gradient_accumulation_steps": 16, "num_train_epochs": 1},
        "lora": {"r": 64},
        "output": {"dir": str(run_dir)},
        "eval": {"judge_model": JUDGE_MODEL, "mmlu_limit": 200},
        "seed": 42,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))

    log = [
        {"step": 1, "loss": 0.70, "rewards/margins": 0.10, "epoch": 0.2},
        {"step": 2, "loss": 0.66, "rewards/margins": 0.20, "epoch": 0.4},
        {"step": 3, "loss": 0.60, "rewards/margins": 0.30, "epoch": 0.6},
        {"step": 4, "loss": 0.57, "rewards/margins": 0.40, "epoch": 0.8},
        {"step": 5, "loss": 0.55, "rewards/margins": 0.50, "epoch": 1.0},
        {"train_runtime": 12.3, "epoch": 1.0},
    ]
    with open(run_dir / "training_log.jsonl", "w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")

    def split_rows(name, n, kind):
        rows = []
        for i in range(n):
            row = {"prompt_id": f"{name}-{i:04d}",
                   "prompt": _msgs(f"{name} question {i}")}
            if kind == "pref":
                row["chosen"] = _msgs("the good answer")
                row["rejected"] = _msgs("the bad answer")
            rows.append(row)
        return rows

    def write_jsonl(path, rows):
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    write_jsonl(splits_dir / "train.jsonl", split_rows("train", 12, "pref"))
    write_jsonl(splits_dir / "margin_eval.jsonl",
                split_rows("margin_eval", 4, "pref"))
    write_jsonl(splits_dir / "generation.jsonl",
                split_rows("generation", 3, "gen"))
    (splits_dir / "manifest.json").write_text(json.dumps(
        {"seed": 42, "counts": {"train": 12, "margin_eval": 4,
                               "generation": 3}}))

    base_margins = [0.5, -0.2, 0.1, -0.3]
    dpo_margins = [1.2, 0.8, -0.1, 0.4]
    (evals_dir / "margin_base.json").write_text(json.dumps({
        "adapter": "base", "metric": "base_logprob_diff",
        "n": 4, "accuracy": 0.5,
        "mean_margin": sum(base_margins) / 4,
        "margins": base_margins,
        "per_pair": [{"prompt_id": f"margin_eval-{i:04d}", "margin": m}
                     for i, m in enumerate(base_margins)]}))
    (evals_dir / "margin_dpo.json").write_text(json.dumps({
        "adapter": "adapter", "metric": "dpo_implicit_margin", "beta": 0.1,
        "n": 4, "accuracy": 0.75,
        "mean_margin": sum(dpo_margins) / 4,
        "margins": dpo_margins,
        "per_pair": [{"prompt_id": f"margin_eval-{i:04d}", "margin": m}
                     for i, m in enumerate(dpo_margins)]}))

    def gen_rows():
        return [{"prompt_id": f"generation-{i:04d}",
                 "prompt": f"generation question {i}",
                 "text": f"answer text {i} from the model"}
                for i in range(3)]

    write_jsonl(evals_dir / "gens_base.jsonl", gen_rows())
    write_jsonl(evals_dir / "gens_dpo.jsonl", gen_rows())
    (evals_dir / "mmlu_base.json").write_text(json.dumps(
        {"adapter": "base", "task": "mmlu", "limit": 200, "acc": 0.55}))
    (evals_dir / "mmlu_dpo.json").write_text(json.dumps(
        {"adapter": "adapter", "task": "mmlu", "limit": 200, "acc": 0.53}))
    (run_dir / "plots" / "margin_hist.png").write_bytes(
        b"\x89PNG\r\n\x1a\n")

    (run_dir / "evals" / "judge_labels.json").write_text(
        json.dumps(_judge_summary(), indent=2))

    return {"run_dir": run_dir, "splits_dir": splits_dir,
            "evals_dir": evals_dir}


def run_assemble(run_dir, splits_dir, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "assemble_results.py"),
         "--run-dir", str(run_dir), "--splits-dir", str(splits_dir),
         *extra],
        capture_output=True, text=True, cwd=REPO_ROOT)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_assemble_happy_path_writes_metrics(tmp_path):
    paths = build_run(tmp_path)
    proc = run_assemble(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode == 0, proc.stderr

    metrics = json.loads(
        (paths["run_dir"] / "metrics.json").read_text())
    assert metrics["margin_base_acc"] == pytest.approx(0.5)
    assert metrics["margin_dpo_acc"] == pytest.approx(0.75)
    assert metrics["margin_delta"] == pytest.approx(0.25)
    assert metrics["mmlu_base_acc"] == pytest.approx(0.55)
    assert metrics["mmlu_dpo_acc"] == pytest.approx(0.53)
    assert metrics["mmlu_delta"] == pytest.approx(-0.02)
    assert metrics["config_hash"] == sha256_of_file(
        paths["run_dir"] / "config.yaml")
    assert metrics["model_name"] == MODEL_NAME
    assert metrics["model_revision"] == MODEL_REV
    assert metrics["seed"] == 42
    assert metrics["beta"] == pytest.approx(0.1)


def test_assemble_judge_numbers_with_ties(tmp_path):
    # Headline win rate is STRICT (integer dpo-wins / n, ties as
    # non-wins) so the Wilson CI is exactly valid for it. The length
    # splits keep the AlpacaEval half-win convention as a diagnostic.
    paths = build_run(tmp_path)
    proc = run_assemble(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode == 0, proc.stderr

    metrics = json.loads(
        (paths["run_dir"] / "metrics.json").read_text())
    assert metrics["judge_win_rate"] == pytest.approx(2 / 6)
    assert metrics["judge_n"] == 6
    assert metrics["judge_n_prompts"] == 3
    assert metrics["judge_ties"] == 3
    assert metrics["judge_parse_errors"] == 0
    assert metrics["judge_decisive_win_rate"] == pytest.approx(2 / 3)
    assert metrics["judge_decisive_n"] == 3
    splits = metrics["judge_length_splits"]
    assert splits["overall"] == pytest.approx(3.5 / 6)
    assert splits["when_dpo_longer"] == pytest.approx(2.5 / 3)
    assert splits["when_base_longer"] == pytest.approx(1 / 3)

    lo, hi = metrics["judge_wilson_95"]
    assert lo == pytest.approx(0.0968, abs=1e-3)
    assert hi == pytest.approx(0.7000, abs=1e-3)


def test_assemble_wilson_matches_utils_on_fixture(tmp_path):
    # independent guard: the stored CI must equal utils.wilson_interval
    # computed straight from the fixture labels.
    paths = build_run(tmp_path)
    proc = run_assemble(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode == 0, proc.stderr

    labels = json.loads(
        (paths["run_dir"] / "evals" / "judge_labels.json").read_text())["labels"]
    wins = sum(1 for lab in labels if lab["winner"] == "dpo")
    expected = wilson_interval(wins, len(labels))
    metrics = json.loads(
        (paths["run_dir"] / "metrics.json").read_text())
    assert metrics["judge_wilson_95"][0] == pytest.approx(expected[0])
    assert metrics["judge_wilson_95"][1] == pytest.approx(expected[1])


def test_assemble_prints_one_line_claim(tmp_path):
    paths = build_run(tmp_path)
    proc = run_assemble(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "0.500" in out and "0.750" in out
    assert "0.333" in out            # strict judge win rate (2/6)
    assert "0.667" in out            # decisive win rate (2/3)
    assert "0.097" in out and "0.700" in out  # Wilson CI
    assert "0.550" in out and "0.530" in out  # MMLU
    assert MODEL_NAME in out
    assert "beta=0.1" in out


def test_assemble_merges_train_time_metrics(tmp_path):
    paths = build_run(tmp_path)
    # train.py already wrote a metrics.json with train-time keys.
    (paths["run_dir"] / "metrics.json").write_text(json.dumps({
        "config_hash": sha256_of_file(paths["run_dir"] / "config.yaml"),
        "model": MODEL_NAME, "beta": 0.1,
        "train_pairs": 12, "global_step": 625}))
    proc = run_assemble(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode == 0, proc.stderr
    metrics = json.loads(
        (paths["run_dir"] / "metrics.json").read_text())
    assert metrics["train_pairs"] == 12
    assert metrics["global_step"] == 625
    assert metrics["margin_dpo_acc"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Missing-file loud errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("relpath", [
    "config.yaml",
    "training_log.jsonl",
    "evals/judge_labels.json",
    "evals/margin_base.json",
    "evals/margin_dpo.json",
    "evals/gens_base.jsonl",
    "evals/gens_dpo.jsonl",
])
def test_assemble_missing_file_loud_error(tmp_path, relpath):
    paths = build_run(tmp_path)
    (paths["run_dir"] / relpath).unlink()
    proc = run_assemble(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert relpath in combined
    assert "missing" in combined.lower()
    assert not (paths["run_dir"] / "metrics.json").exists()


def test_assemble_without_mmlu_records_not_measured(tmp_path):
    paths = build_run(tmp_path)
    (paths["run_dir"] / "evals" / "mmlu_base.json").unlink()
    (paths["run_dir"] / "evals" / "mmlu_dpo.json").unlink()
    proc = run_assemble(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode == 0, proc.stderr
    metrics = json.loads(
        (paths["run_dir"] / "metrics.json").read_text())
    assert metrics["mmlu_status"] == "not_measured"
    assert metrics["mmlu_base_acc"] is None
    assert metrics["mmlu_dpo_acc"] is None
    assert "not measured" in proc.stdout.lower()


@pytest.mark.parametrize("relpath", [
    "train.jsonl",
    "margin_eval.jsonl",
    "generation.jsonl",
    "manifest.json",
])
def test_assemble_missing_split_loud_error(tmp_path, relpath):
    paths = build_run(tmp_path)
    (paths["splits_dir"] / relpath).unlink()
    proc = run_assemble(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert relpath in (proc.stdout + proc.stderr)


def test_assemble_missing_plots_loud_error(tmp_path):
    paths = build_run(tmp_path)
    (paths["run_dir"] / "plots" / "margin_hist.png").unlink()
    proc = run_assemble(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "plots" in (proc.stdout + proc.stderr).lower()


def test_assemble_lists_every_missing_file(tmp_path):
    paths = build_run(tmp_path)
    (paths["run_dir"] / "evals" / "margin_dpo.json").unlink()
    (paths["run_dir"] / "evals" / "mmlu_dpo.json").unlink()
    proc = run_assemble(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "margin_dpo.json" in combined
    # MMLU is optional: its absence must not fail assembly or be listed.
    assert "mmlu_dpo.json" not in combined
