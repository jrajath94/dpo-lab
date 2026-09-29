"""Hardening tests for dpo_weekend.utils: every pure function.

RED phase: written before wilson_interval / validate_config exist.
"""

import copy
import json
import math
from pathlib import Path

import pytest
import yaml

from dpo_weekend import utils
from dpo_weekend.utils import (
    apply_overrides,
    dpo_loss_from_margin,
    implicit_reward_margin,
    load_jsonl,
    mean_kl_to_reference,
    save_json,
    sha256_of_file,
    validate_config,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parent.parent


def _load_cfg(name: str) -> dict:
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# implicit_reward_margin
# ---------------------------------------------------------------------------

def test_implicit_reward_margin_sign():
    m = implicit_reward_margin(beta=0.1, logp_chosen=-2.0,
                               logp_rejected=-5.0, logp_ref_chosen=-3.0,
                               logp_ref_rejected=-3.0)
    assert m > 0


def test_implicit_reward_margin_exact_value():
    # beta * ((lc - lrc) - (lr - lrr)) = 0.5 * ((1-2) - (3-4)) = 0.5 * 0 = 0
    m = implicit_reward_margin(beta=0.5, logp_chosen=1.0,
                               logp_rejected=3.0, logp_ref_chosen=2.0,
                               logp_ref_rejected=4.0)
    assert m == pytest.approx(0.0)
    # beta scales linearly: doubling beta doubles the margin
    m1 = implicit_reward_margin(0.1, -2.0, -5.0, -3.0, -3.0)
    m2 = implicit_reward_margin(0.2, -2.0, -5.0, -3.0, -3.0)
    assert m2 == pytest.approx(2 * m1)


def test_implicit_reward_margin_zero_when_policy_matches_ref():
    m = implicit_reward_margin(0.1, -2.0, -2.0, -2.0, -2.0)
    assert m == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# dpo_loss_from_margin
# ---------------------------------------------------------------------------

def test_dpo_loss_at_zero_margin_is_log2():
    assert dpo_loss_from_margin(0.0) == pytest.approx(math.log(2))


def test_dpo_loss_decreases_with_margin():
    assert dpo_loss_from_margin(2.0) < dpo_loss_from_margin(0.0)
    assert dpo_loss_from_margin(10.0) < dpo_loss_from_margin(2.0)
    assert dpo_loss_from_margin(-2.0) > dpo_loss_from_margin(0.0)


def test_dpo_loss_matches_sigmoid_formula():
    for margin in (-3.0, -0.5, 0.0, 1.5, 4.0):
        expected = -math.log(1.0 / (1.0 + math.exp(-margin)))
        assert dpo_loss_from_margin(margin) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# mean_kl_to_reference
# ---------------------------------------------------------------------------

def test_mean_kl_zero_for_identical_models():
    logp = [-1.0, -2.0, -3.0]
    assert mean_kl_to_reference(logp, logp) == 0.0


def test_mean_kl_exact_value():
    # mean of (1-0, 2-0, 3-0) = 2.0
    assert mean_kl_to_reference([1.0, 2.0, 3.0],
                                [0.0, 0.0, 0.0]) == pytest.approx(2.0)


def test_mean_kl_rejects_empty_or_misaligned():
    with pytest.raises(ValueError):
        mean_kl_to_reference([], [])
    with pytest.raises(ValueError):
        mean_kl_to_reference([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------

def _cfg():
    return {"dpo": {"beta": 0.1, "num_train_epochs": 1,
                    "gradient_checkpointing": True},
            "model": {"name": "x"}}


def test_apply_overrides_parses_types():
    cfg = apply_overrides(_cfg(),
                          ["dpo.beta=0.2", "dpo.num_train_epochs=3",
                           "dpo.gradient_checkpointing=false",
                           "model.name=qwen"])
    assert cfg["dpo"]["beta"] == pytest.approx(0.2)
    assert isinstance(cfg["dpo"]["beta"], float)
    assert cfg["dpo"]["num_train_epochs"] == 3
    assert isinstance(cfg["dpo"]["num_train_epochs"], int)
    assert cfg["dpo"]["gradient_checkpointing"] is False
    assert cfg["model"]["name"] == "qwen"


def test_apply_overrides_bool_true():
    cfg = apply_overrides(_cfg(), ["dpo.gradient_checkpointing=True"])
    assert cfg["dpo"]["gradient_checkpointing"] is True


def test_apply_overrides_rejects_unknown_paths():
    with pytest.raises(KeyError):
        apply_overrides(_cfg(), ["dpo.nope=1"])
    with pytest.raises(KeyError):
        apply_overrides(_cfg(), ["missing.beta=1"])
    with pytest.raises(ValueError):
        apply_overrides(_cfg(), ["dpo.beta"])  # no '='


def test_apply_overrides_scientific_notation_is_float():
    cfg = apply_overrides(_cfg(), ["dpo.beta=1e-4"])
    assert cfg["dpo"]["beta"] == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# sha256_of_file
# ---------------------------------------------------------------------------

def test_sha256_of_file_known_digest(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    # well-known SHA-256 of b"abc"
    assert sha256_of_file(p) == \
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_of_file_changes_with_content(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert sha256_of_file(a) != sha256_of_file(b)


# ---------------------------------------------------------------------------
# load_jsonl / save_json
# ---------------------------------------------------------------------------

def test_save_json_roundtrip(tmp_path):
    obj = {"a": 1, "b": [1, 2, {"c": "x \u00e9"}]}
    p = tmp_path / "sub" / "out.json"
    save_json(obj, p)  # creates parent dirs
    assert json.loads(p.read_text(encoding="utf-8")) == obj


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n   \n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"a": 2}]


# ---------------------------------------------------------------------------
# wilson_interval
# ---------------------------------------------------------------------------

def test_wilson_interval_typical_value():
    # hand-computed via the Wilson formula (z=1.96):
    # p=0.6, n=200 -> center=0.59811, half=0.06728
    lo, hi = wilson_interval(120, 200)
    assert lo == pytest.approx(0.531, abs=0.001)
    assert hi == pytest.approx(0.665, abs=0.001)
    # tighter: values I computed directly from the formula
    assert lo == pytest.approx(0.530835, abs=1e-4)
    assert hi == pytest.approx(0.665395, abs=1e-4)


def test_wilson_interval_zero_wins_has_zero_lower_bound():
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0
    assert hi == pytest.approx(0.07135, abs=1e-4)
    assert hi > 0  # not degenerate


def test_wilson_interval_all_wins_has_one_upper_bound():
    lo, hi = wilson_interval(50, 50)
    assert hi == 1.0
    assert lo == pytest.approx(0.92865, abs=1e-4)
    assert lo < 1.0


def test_wilson_interval_half_wins():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=0.01)
    assert hi == pytest.approx(0.7634, abs=0.01)


def test_wilson_interval_bounds_and_ordering():
    for wins, n in [(0, 1), (1, 1), (3, 10), (99, 100), (1, 1000)]:
        lo, hi = wilson_interval(wins, n)
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= wins / n <= hi  # interval covers the point estimate


def test_wilson_interval_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_interval(5, 0)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["dpo_qwen25_15b_qlora.yaml",
                                  "dpo_qwen25_3b.yaml"])
def test_validate_config_accepts_real_configs(name):
    validate_config(_load_cfg(name))  # must not raise


def test_validate_config_lists_every_missing_key():
    cfg = _load_cfg("dpo_qwen25_15b_qlora.yaml")
    broken = copy.deepcopy(cfg)
    del broken["model"]["name"]
    del broken["dpo"]["beta"]
    del broken["dpo"]["learning_rate"]
    del broken["output"]["dir"]
    del broken["eval"]["judge_model"]
    with pytest.raises(ValueError) as exc:
        validate_config(broken)
    msg = str(exc.value)
    for key in ("model.name", "dpo.beta", "dpo.learning_rate",
                "output.dir", "eval.judge_model"):
        assert key in msg, f"{key} not listed in: {msg}"


def test_validate_config_lora_required_unless_full_finetune():
    cfg = _load_cfg("dpo_qwen25_15b_qlora.yaml")
    broken = copy.deepcopy(cfg)
    del broken["lora"]["r"]
    with pytest.raises(ValueError) as exc:
        validate_config(broken)
    assert "lora.r" in str(exc.value)
    # the 3B config is full-finetune with no lora section at all: passes
    validate_config(_load_cfg("dpo_qwen25_3b.yaml"))


def test_validate_config_rejects_non_dict_sections():
    cfg = _load_cfg("dpo_qwen25_15b_qlora.yaml")
    broken = copy.deepcopy(cfg)
    broken["data"] = "oops"
    with pytest.raises(ValueError) as exc:
        validate_config(broken)
    assert "data.dataset" in str(exc.value)
