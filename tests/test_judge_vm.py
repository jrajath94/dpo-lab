"""Unit tests for the VM-side judge harness (scripts/judge_on_vm.py).

Covers the pure parts: A/B -> base/dpo winner mapping (both orders),
judge-JSON parsing (valid, chatter-wrapped, lowercase, malformed),
summary math (ties as half wins, length splits), and the Wilson interval
on the final win rate.

The Wilson-interval test imports dpo_weekend.utils.wilson_interval, which
does not exist yet -- a sibling agent is adding it. The test skips cleanly
until it lands (noted as a dependency); everything else must pass now.

Run: pytest tests/test_judge_vm.py
No network, no GPU, no credentials touched here.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import judge_on_vm
from dpo_weekend import evaluate
from dpo_weekend.evaluate import length_controlled_win_rate


# ---------------------------------------------------------------------------
# Protocol parity: the harness must use the exact same template/system.
# ---------------------------------------------------------------------------

def test_judge_template_and_system_match_evaluate():
    assert judge_on_vm.JUDGE_TEMPLATE == evaluate.JUDGE_TEMPLATE
    assert judge_on_vm.JUDGE_SYSTEM == evaluate.JUDGE_SYSTEM


# ---------------------------------------------------------------------------
# Winner mapping: A/B back to base/dpo/tie, both orders.
# ---------------------------------------------------------------------------

def test_mapping_base_first_order():
    # order 1 in evaluate.run_judge: A=base, B=dpo
    assert judge_on_vm.map_winner("A", "base", "dpo") == "base"
    assert judge_on_vm.map_winner("B", "base", "dpo") == "dpo"
    assert judge_on_vm.map_winner("TIE", "base", "dpo") == "tie"


def test_mapping_dpo_first_order():
    # order 2 in evaluate.run_judge: A=dpo, B=base
    assert judge_on_vm.map_winner("A", "dpo", "base") == "dpo"
    assert judge_on_vm.map_winner("B", "dpo", "base") == "base"
    assert judge_on_vm.map_winner("TIE", "dpo", "base") == "tie"


# ---------------------------------------------------------------------------
# Parse logic for the judge's raw content.
# ---------------------------------------------------------------------------

def _mk(winner, reason="clear"):
    return '{"winner": "%s", "reason": "%s"}' % (winner, reason)


def test_parse_valid_json():
    lab = judge_on_vm.parse_judge_content(_mk("A", "a is more helpful"))
    assert lab["winner_ab"] == "A"
    assert lab["parse_error"] is False
    assert lab["reason"] == "a is more helpful"


def test_parse_chatter_wrapped_json():
    content = ("Sure, here is my judgment:\n"
               + _mk("B", "b wins") + "\nHope this helps.")
    lab = judge_on_vm.parse_judge_content(content)
    assert lab["winner_ab"] == "B"
    assert lab["parse_error"] is False


def test_parse_lowercase_winner_normalized():
    lab = judge_on_vm.parse_judge_content(_mk("a", "lower"))
    assert lab["winner_ab"] == "A"
    assert lab["parse_error"] is False
    lab2 = judge_on_vm.parse_judge_content(_mk("tie", "equal"))
    assert lab2["winner_ab"] == "TIE"
    assert lab2["parse_error"] is False


def test_parse_unknown_winner_is_tie_with_error():
    lab = judge_on_vm.parse_judge_content(_mk("C", "huh"))
    assert lab["winner_ab"] == "TIE"
    assert lab["parse_error"] is True


def test_parse_malformed_content_is_tie_with_error():
    lab = judge_on_vm.parse_judge_content("I cannot decide between them")
    assert lab["winner_ab"] == "TIE"
    assert lab["parse_error"] is True
    assert lab["raw"] == "I cannot decide between them"


def test_parse_reason_truncated_to_300_chars():
    long_reason = "x" * 500
    lab = judge_on_vm.parse_judge_content(_mk("A", long_reason))
    assert len(lab["reason"]) == 300


# ---------------------------------------------------------------------------
# Summary math: ties count as half wins; length splits; parse-error count.
# ---------------------------------------------------------------------------

def _label(winner, dpo_longer, parse_error=False):
    return {"winner": winner, "dpo_longer": dpo_longer,
            "parse_error": parse_error}


def test_summary_ties_count_as_half_wins():
    labels = [_label("dpo", True), _label("dpo", False),
              _label("tie", True), _label("base", False)]
    s = length_controlled_win_rate(labels)
    assert s["overall"] == pytest.approx((2 + 0.5) / 4)
    assert s["n"] == 4
    assert s["n_ties"] == 1
    assert s["n_parse_errors"] == 0


def test_summary_length_splits():
    labels = [
        _label("dpo", True),   # dpo longer, dpo wins
        _label("tie", True),   # dpo longer, tie
        _label("dpo", False),  # base longer, dpo wins
        _label("base", False),  # base longer, base wins
    ]
    s = length_controlled_win_rate(labels)
    assert s["when_dpo_longer"] == pytest.approx((1 + 0.5) / 2)
    assert s["when_base_longer"] == pytest.approx(1 / 2)


def test_summary_length_split_empty_side_is_none():
    labels = [_label("dpo", True), _label("base", True)]
    s = length_controlled_win_rate(labels)
    assert s["when_dpo_longer"] == pytest.approx(0.5)
    assert s["when_base_longer"] is None


def test_summary_empty_labels():
    s = length_controlled_win_rate([])
    assert s["overall"] is None
    assert s["n"] == 0
    assert s["n_ties"] == 0
    assert s["n_parse_errors"] == 0


def test_summary_counts_parse_errors():
    labels = [_label("tie", True, parse_error=True),
              _label("dpo", False),
              _label("tie", False, parse_error=True)]
    s = length_controlled_win_rate(labels)
    assert s["n_parse_errors"] == 2
    assert s["n_ties"] == 2


# ---------------------------------------------------------------------------
# Wilson interval on the final win rate.
#
# DEPENDENCY: dpo_weekend.utils.wilson_interval does not exist yet; a
# sibling agent is adding it. This test skips until it lands.
# ---------------------------------------------------------------------------

def test_wilson_interval():
    try:
        from dpo_weekend.utils import wilson_interval
    except ImportError:
        pytest.skip(
            "dpo_weekend.utils.wilson_interval not added yet "
            "(sibling agent owns it)")
    lo, hi = wilson_interval(5, 10)
    assert 0 <= lo <= hi <= 1
    # 95% Wilson CI for 5/10 is approx (0.2366, 0.7634).
    assert lo == pytest.approx(0.2366, abs=0.01)
    assert hi == pytest.approx(0.7634, abs=0.01)

    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == pytest.approx(0.0, abs=1e-6)
    assert hi0 > 0.25  # not degenerate: approx (0, 0.2776)

    lo10, hi10 = wilson_interval(10, 10)
    assert lo10 < 0.75
    assert hi10 == pytest.approx(1.0, abs=1e-6)
