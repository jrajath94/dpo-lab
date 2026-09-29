"""Tests for scripts/verify_run.py, the independent verification gate.

Builds a synthetic run dir, assembles metrics.json with the real
assemble_results.py, then checks that verify_run.py passes on the honest
artifacts and fails loudly on every tampered variant. No GPU, no network.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

from dpo_weekend.evaluate import length_controlled_win_rate

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


def assemble(paths):
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "assemble_results.py"),
         "--run-dir", str(paths["run_dir"]),
         "--splits-dir", str(paths["splits_dir"])],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr


def run_verify(run_dir, splits_dir):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "verify_run.py"),
         "--run-dir", str(run_dir), "--splits-dir", str(splits_dir)],
        capture_output=True, text=True, cwd=REPO_ROOT)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_verify_happy_path(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERIFY OK" in proc.stdout


# ---------------------------------------------------------------------------
# (a) split integrity
# ---------------------------------------------------------------------------

def test_verify_catches_leaked_train_prompt_in_margin_eval(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    rows = read_jsonl(paths["splits_dir"] / "margin_eval.jsonl")
    # leak: first margin-eval prompt text is now a verbatim train prompt
    rows[0]["prompt"] = _msgs("train question 0")
    write_jsonl(paths["splits_dir"] / "margin_eval.jsonl", rows)
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "overlap" in combined.lower()
    assert "train" in combined and "margin_eval" in combined


def test_verify_catches_leaked_train_prompt_in_generation(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    rows = read_jsonl(paths["splits_dir"] / "generation.jsonl")
    rows[1]["prompt"] = _msgs("train question 5")
    write_jsonl(paths["splits_dir"] / "generation.jsonl", rows)
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "overlap" in (proc.stdout + proc.stderr).lower()


def test_verify_catches_wrong_prompt_id_prefix(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    rows = read_jsonl(paths["splits_dir"] / "margin_eval.jsonl")
    rows[0]["prompt_id"] = "eval-0000"  # wrong prefix
    write_jsonl(paths["splits_dir"] / "margin_eval.jsonl", rows)
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "eval-0000" in combined
    assert "prefix" in combined.lower()


def test_verify_catches_duplicate_prompt_ids(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    rows = read_jsonl(paths["splits_dir"] / "train.jsonl")
    rows[1]["prompt_id"] = rows[0]["prompt_id"]  # duplicate
    write_jsonl(paths["splits_dir"] / "train.jsonl", rows)
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "duplicate" in (proc.stdout + proc.stderr).lower()


# ---------------------------------------------------------------------------
# (b) number re-derivation
# ---------------------------------------------------------------------------

def test_verify_catches_tampered_margin_accuracy(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    p = paths["run_dir"] / "evals" / "margin_dpo.json"
    doc = json.loads(p.read_text())
    doc["accuracy"] = doc["accuracy"] + 0.01  # tamper: margins unchanged
    p.write_text(json.dumps(doc))
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "margin_dpo" in (proc.stdout + proc.stderr)


def test_verify_catches_tampered_margin_margins(tmp_path):
    # margins changed but stored accuracy left stale: re-derivation from
    # the per-pair margins must disagree with the summary.
    paths = build_run(tmp_path)
    assemble(paths)
    p = paths["run_dir"] / "evals" / "margin_base.json"
    doc = json.loads(p.read_text())
    doc["margins"][0] = -5.0  # flip one margin, accuracy still says 0.5
    p.write_text(json.dumps(doc))
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "margin_base" in (proc.stdout + proc.stderr)


def test_verify_accepts_len_norm_base_twin(tmp_path):
    # When margin_base.json carries accuracy_len_norm, metrics.json must
    # carry the twin (not the raw accuracy) and say so.
    paths = build_run(tmp_path)
    p = paths["run_dir"] / "evals" / "margin_base.json"
    doc = json.loads(p.read_text())
    doc["accuracy_len_norm"] = 0.75
    doc["mean_margin_len_norm"] = 0.1
    p.write_text(json.dumps(doc))
    assemble(paths)
    metrics = json.loads(
        (paths["run_dir"] / "metrics.json").read_text())
    assert metrics["margin_base_acc"] == pytest.approx(0.75)
    assert metrics["margin_base_metric"] == "accuracy_len_norm"
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_verify_catches_len_norm_mismatch(tmp_path):
    # metrics.json claiming the twin while the eval file disagrees.
    paths = build_run(tmp_path)
    p = paths["run_dir"] / "evals" / "margin_base.json"
    doc = json.loads(p.read_text())
    doc["accuracy_len_norm"] = 0.75
    p.write_text(json.dumps(doc))
    assemble(paths)
    m = paths["run_dir"] / "metrics.json"
    metrics = json.loads(m.read_text())
    metrics["margin_base_acc"] = 0.5  # tamper: raw acc, not the twin
    m.write_text(json.dumps(metrics))
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "margin_base" in (proc.stdout + proc.stderr)


def test_verify_catches_tampered_judge_summary(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    p = paths["run_dir"] / "evals" / "judge_labels.json"
    doc = json.loads(p.read_text())
    # flip one raw label winner: the re-derived strict win rate must
    # stop matching metrics.json
    for lab in doc["labels"]:
        if lab["winner"] == "dpo":
            lab["winner"] = "base"
            break
    p.write_text(json.dumps(doc))
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "judge" in (proc.stdout + proc.stderr).lower()


def test_verify_catches_tampered_wilson_ci(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    p = paths["run_dir"] / "metrics.json"
    doc = json.loads(p.read_text())
    doc["judge_wilson_95"][0] += 0.01  # tamper the stored CI
    p.write_text(json.dumps(doc))
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "wilson" in (proc.stdout + proc.stderr).lower()


def test_verify_catches_incomplete_judge_labels(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    p = paths["run_dir"] / "evals" / "judge_labels.json"
    doc = json.loads(p.read_text())
    # drop one order: generation-0002 now has a single label
    doc["labels"] = [lab for lab in doc["labels"]
                     if not (lab["prompt_id"] == "generation-0002"
                             and lab["order"] == "dpo_first")]
    p.write_text(json.dumps(doc))
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "generation-0002" in combined


# ---------------------------------------------------------------------------
# (c) training-log sanity
# ---------------------------------------------------------------------------

def test_verify_catches_nan_loss(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    rows = read_jsonl(paths["run_dir"] / "training_log.jsonl")
    rows[2]["loss"] = float("nan")
    write_jsonl(paths["run_dir"] / "training_log.jsonl", rows)
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "nan" in (proc.stdout + proc.stderr).lower()


def test_verify_catches_increasing_loss(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    rows = read_jsonl(paths["run_dir"] / "training_log.jsonl")
    loss_rows = [r for r in rows if "loss" in r]
    for i, r in enumerate(loss_rows):
        r["loss"] = 0.5 + 0.05 * i  # strictly increasing
    write_jsonl(paths["run_dir"] / "training_log.jsonl", rows)
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "decreasing" in (proc.stdout + proc.stderr).lower()


def test_verify_catches_missing_logged_margins(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    rows = read_jsonl(paths["run_dir"] / "training_log.jsonl")
    for r in rows:
        r.pop("rewards/margins", None)
    write_jsonl(paths["run_dir"] / "training_log.jsonl", rows)
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "margin" in (proc.stdout + proc.stderr).lower()


# ---------------------------------------------------------------------------
# (d) config hash
# ---------------------------------------------------------------------------

def test_verify_catches_config_hash_mismatch(tmp_path):
    paths = build_run(tmp_path)
    assemble(paths)
    cfg = paths["run_dir"] / "config.yaml"
    cfg.write_text(cfg.read_text() + "# sneaky edit\n")
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "config_hash" in (proc.stdout + proc.stderr)


def test_verify_catches_missing_metrics(tmp_path):
    paths = build_run(tmp_path)
    # assemble not run: no metrics.json at all
    proc = run_verify(paths["run_dir"], paths["splits_dir"])
    assert proc.returncode != 0
    assert "metrics.json" in (proc.stdout + proc.stderr)
