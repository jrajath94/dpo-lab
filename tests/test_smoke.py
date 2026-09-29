"""Smoke checks that run on CPU with no GPU, no network, no heavy deps.

Run: pytest tests/   (or bash scripts/00_smoke_test.sh)
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_TOP_KEYS = {"model", "data", "dpo", "output", "seed"}
REQUIRED_DATA_KEYS = {"dataset", "splits_dir", "train_pairs",
                      "margin_eval_pairs", "generation_prompts", "seed",
                      "max_seq_length"}
REQUIRED_EVAL_KEYS = {"temperature", "top_p", "max_new_tokens", "judge_model",
                      "mmlu_limit", "mmlu_batch_size"}


def _load(name: str) -> dict:
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("name", ["dpo_qwen25_15b_qlora.yaml",
                                  "dpo_qwen25_3b.yaml"])
def test_configs_parse_and_have_required_keys(name):
    cfg = _load(name)
    assert REQUIRED_TOP_KEYS <= set(cfg), f"{name} missing top-level keys"
    assert REQUIRED_DATA_KEYS <= set(cfg["data"]), f"{name} missing data keys"
    assert REQUIRED_EVAL_KEYS <= set(cfg.get("eval", {})), \
        f"{name} missing eval keys"


@pytest.mark.parametrize("name", ["dpo_qwen25_15b_qlora.yaml",
                                  "dpo_qwen25_3b.yaml"])
def test_beta_in_sweep_set(name):
    cfg = _load(name)
    assert float(cfg["dpo"]["beta"]) in (0.05, 0.1, 0.2)
    assert cfg["dpo"]["loss_type"] == "sigmoid"
    assert cfg["data"]["train_pairs"] >= 1000


def test_qlora_config_consistency():
    cfg = _load("dpo_qwen25_15b_qlora.yaml")
    assert cfg["quantization"]["load_in_4bit"] is True
    assert not cfg.get("full_finetune", False)
    assert cfg["lora"]["r"] > 0
    assert cfg["dpo"]["gradient_accumulation_steps"] * \
        cfg["dpo"]["per_device_batch"] == 32  # effective batch


def test_full_finetune_config_consistency():
    cfg = _load("dpo_qwen25_3b.yaml")
    assert cfg.get("full_finetune", False) is True
    assert cfg["quantization"]["load_in_4bit"] is False


def test_scripts_have_no_todos():
    for script in (ROOT / "scripts").glob("*.sh"):
        text = script.read_text(encoding="utf-8")
        assert "TODO" not in text, f"{script.name} still has a TODO"


def test_run_path_has_no_notimplemented():
    for src in (ROOT / "src" / "dpo_weekend").glob("*.py"):
        text = src.read_text(encoding="utf-8")
        assert "NotImplementedError" not in text, \
            f"{src.name} still raises NotImplementedError"


def test_pyproject_has_all_deps():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for dep in ["torch", "transformers", "datasets", "trl", "peft",
                "accelerate", "bitsandbytes", "pyyaml", "requests",
                "matplotlib"]:
        assert dep in text, f"pyproject missing {dep}"


def test_evaluate_dispatch_covers_all_evals():
    text = (ROOT / "src" / "dpo_weekend" / "evaluate.py").read_text(
        encoding="utf-8")
    for choice in ["margin", "generation", "judge", "regression", "plots",
                   "all"]:
        assert f'"{choice}"' in text


def test_judge_records_template_and_model():
    text = (ROOT / "src" / "dpo_weekend" / "evaluate.py").read_text(
        encoding="utf-8")
    assert "JUDGE_TEMPLATE" in text
    assert "judge_model" in text
    assert "order" in text  # order-swapped judging
