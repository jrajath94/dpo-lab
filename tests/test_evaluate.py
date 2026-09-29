"""Hardening tests for dpo_weekend.evaluate pure functions.

Covers margin_accuracy, length_controlled_win_rate splits, and the judge
content parser extracted from _judge_call.
"""

import pytest

from dpo_weekend.evaluate import (length_controlled_win_rate,
                                  margin_accuracy, parse_judge_content)


# ---------------------------------------------------------------------------
# margin_accuracy
# ---------------------------------------------------------------------------

def test_margin_accuracy_exact_fraction():
    assert margin_accuracy([1.0, -1.0, 2.0, -0.5]) == pytest.approx(0.5)


def test_margin_accuracy_all_positive_is_one():
    assert margin_accuracy([0.1, 5.0, 0.001]) == pytest.approx(1.0)


def test_margin_accuracy_all_negative_is_zero():
    assert margin_accuracy([-1.0, -0.1]) == pytest.approx(0.0)


def test_margin_accuracy_zero_margin_counts_as_wrong():
    # strictly positive: a zero margin is not agreement
    assert margin_accuracy([0.0]) == pytest.approx(0.0)


def test_margin_accuracy_rejects_empty():
    with pytest.raises(ValueError):
        margin_accuracy([])


# ---------------------------------------------------------------------------
# length_controlled_win_rate
# ---------------------------------------------------------------------------

def _label(winner, dpo_longer, parse_error=False):
    return {"winner": winner, "dpo_longer": dpo_longer,
            "parse_error": parse_error}


def test_win_rate_overall_with_ties_as_half_wins():
    labels = [_label("dpo", True), _label("dpo", False),
              _label("tie", True), _label("base", False)]
    s = length_controlled_win_rate(labels)
    assert s["overall"] == pytest.approx((2 + 0.5) / 4)
    assert s["n"] == 4
    assert s["n_ties"] == 1
    assert s["n_parse_errors"] == 0


def test_win_rate_length_splits():
    labels = [
        _label("dpo", True),   # dpo longer, dpo wins
        _label("tie", True),   # dpo longer, tie -> half win
        _label("dpo", False),  # base longer, dpo wins
        _label("base", False),  # base longer, base wins
    ]
    s = length_controlled_win_rate(labels)
    assert s["when_dpo_longer"] == pytest.approx((1 + 0.5) / 2)
    assert s["when_base_longer"] == pytest.approx(1 / 2)


def test_win_rate_empty_side_is_none():
    labels = [_label("dpo", True), _label("base", True)]
    s = length_controlled_win_rate(labels)
    assert s["when_dpo_longer"] == pytest.approx(0.5)
    assert s["when_base_longer"] is None


def test_win_rate_empty_labels():
    s = length_controlled_win_rate([])
    assert s["overall"] is None
    assert s["n"] == 0
    assert s["n_ties"] == 0
    assert s["n_parse_errors"] == 0


def test_win_rate_counts_parse_errors():
    labels = [_label("tie", True, parse_error=True),
              _label("dpo", False),
              _label("tie", False, parse_error=True)]
    s = length_controlled_win_rate(labels)
    assert s["n_parse_errors"] == 2
    assert s["n_ties"] == 2


# ---------------------------------------------------------------------------
# parse_judge_content
# ---------------------------------------------------------------------------

def _mk(winner, reason="clear"):
    return '{"winner": "%s", "reason": "%s"}' % (winner, reason)


def test_parse_valid_json():
    lab = parse_judge_content(_mk("A", "a is more helpful"))
    assert lab["winner_ab"] == "A"
    assert lab["parse_error"] is False
    assert lab["reason"] == "a is more helpful"
    assert lab["raw"] == _mk("A", "a is more helpful")


def test_parse_json_embedded_in_chatter():
    content = ("Sure, here is my judgment:\n"
               + _mk("B", "b wins") + "\nHope this helps.")
    lab = parse_judge_content(content)
    assert lab["winner_ab"] == "B"
    assert lab["parse_error"] is False


def test_parse_lowercase_winner_normalized():
    lab = parse_judge_content(_mk("a", "lower"))
    assert lab["winner_ab"] == "A"
    assert lab["parse_error"] is False
    lab2 = parse_judge_content(_mk("tie", "equal"))
    assert lab2["winner_ab"] == "TIE"
    assert lab2["parse_error"] is False


def test_parse_whitespace_winner_normalized():
    lab = parse_judge_content('{"winner": " b ", "reason": "x"}')
    assert lab["winner_ab"] == "B"
    assert lab["parse_error"] is False


def test_parse_unknown_winner_is_tie_with_error():
    lab = parse_judge_content(_mk("C", "huh"))
    assert lab["winner_ab"] == "TIE"
    assert lab["parse_error"] is True


def test_parse_missing_winner_is_tie_with_error():
    lab = parse_judge_content('{"reason": "no winner key"}')
    assert lab["winner_ab"] == "TIE"
    assert lab["parse_error"] is True


def test_parse_malformed_content_is_tie_with_error():
    lab = parse_judge_content("I cannot decide between them")
    assert lab["winner_ab"] == "TIE"
    assert lab["parse_error"] is True
    assert lab["raw"] == "I cannot decide between them"


def test_parse_json_array_is_tie_with_error():
    lab = parse_judge_content('["A", "B"]')
    assert lab["winner_ab"] == "TIE"
    assert lab["parse_error"] is True


def test_parse_reason_truncated_to_300_chars():
    lab = parse_judge_content(_mk("A", "x" * 500))
    assert len(lab["reason"]) == 300


def test_parse_empty_content_is_tie_with_error():
    lab = parse_judge_content("")
    assert lab["winner_ab"] == "TIE"
    assert lab["parse_error"] is True


# ---------------------------------------------------------------------------
# chat_template_ids / _response_span: transformers>=5 BatchEncoding drift
# (run 4, 2026-09-28). apply_chat_template(tokenize=True) used to return
# list[int]; on transformers>=5 it returns a BatchEncoding whose bare
# list() is the FIELD NAMES. _response_span then built garbage ids and
# sequence_logprob crashed; data.py's count_tokens silently returned 2.
# ---------------------------------------------------------------------------

from collections import UserDict

from dpo_weekend.evaluate import _response_span
from dpo_weekend.utils import chat_template_ids


class _OldStyleTok:
    """Mimics transformers<5: apply_chat_template returns list[int]."""

    def __init__(self, prompt_ids, full_ids):
        self._prompt = prompt_ids
        self._full = full_ids

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        if kwargs.get("add_generation_prompt"):
            return self._prompt
        return self._full

    def encode(self, text, add_special_tokens=False):
        return [99, 100]


class _NewStyleTok(_OldStyleTok):
    """Mimics transformers>=5: apply_chat_template returns BatchEncoding."""

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        ids = super().apply_chat_template(messages, tokenize=tokenize,
                                          **kwargs)
        return UserDict({"input_ids": ids,
                         "attention_mask": [1] * len(ids)})


def test_chat_template_ids_old_style_list():
    tok = _OldStyleTok([1, 2, 3], [1, 2, 3, 4])
    assert chat_template_ids(tok, []) == [1, 2, 3, 4]


def test_chat_template_ids_new_style_batch_encoding():
    # The run-4 regression: bare list(BatchEncoding) is ['input_ids',
    # 'attention_mask'] (field names). The helper must return real ids.
    tok = _NewStyleTok([1, 2, 3], [1, 2, 3, 4])
    out = chat_template_ids(tok, [])
    assert out == [1, 2, 3, 4]
    assert all(isinstance(i, int) for i in out)


def test_chat_template_ids_rejects_non_int_ids():
    class _BadTok:
        def apply_chat_template(self, messages, tokenize=True, **kwargs):
            return ["input_ids", "attention_mask"]  # the run-4 garbage

    with pytest.raises(TypeError):
        chat_template_ids(_BadTok(), [])


def test_response_span_prefix_path_new_style():
    tok = _NewStyleTok([1, 2, 3], [1, 2, 3, 4, 5])
    ids, start = _response_span(tok, [{"role": "user"}],
                                [{"role": "assistant"}])
    assert ids == [1, 2, 3, 4, 5]
    assert start == 3


def test_response_span_fallback_path_new_style():
    # Template disagreement -> response tokenized on its own.
    tok = _NewStyleTok([1, 2, 3], [7, 8, 9])
    ids, start = _response_span(tok, [{"role": "user"}],
                                [{"role": "assistant"}])
    assert ids == [1, 2, 3, 99, 100]
    assert start == 3


def test_response_span_real_qwen_tokenizer():
    """Pod-gated pre-flight: the real pinned tokenizer through _response_span.

    Skipped on the VM (no transformers); runs on the pod inside the smoke
    test. This is the gate that would have caught the run-4 eval crash
    before any training spend.
    """
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        trust_remote_code=False)
    prompt = [{"role": "user", "content": "Say hi."}]
    resp = [{"role": "assistant", "content": "Hello!"}]
    ids, start = _response_span(tok, prompt, resp)
    assert isinstance(ids, list) and len(ids) > start > 0
    assert all(isinstance(i, int) for i in ids)


# ---------------------------------------------------------------------------
# generate() contract: transformers 5.x rejects the `generator` kwarg
# (run 5, 2026-09-29). generate() must be called without it; per-row
# determinism comes from seeding the global RNGs instead.
# ---------------------------------------------------------------------------

import ast


def _generate_calls():
    src = open("src/dpo_weekend/evaluate.py", encoding="utf-8").read()
    tree = ast.parse(src)
    calls = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node):
            func = node.func
            if (isinstance(func, ast.Attribute)
                    and func.attr == "generate"):
                calls.append(node)
            self.generic_visit(node)

    V().visit(tree)
    return calls


def test_generate_never_passes_generator_kwarg():
    calls = _generate_calls()
    assert calls, "no model.generate() call found; test is stale"
    for call in calls:
        kwarg_names = [kw.arg for kw in call.keywords]
        assert "generator" not in kwarg_names, (
            "model.generate() must not pass generator= "
            "(transformers>=5 raises ValueError)")


def test_generate_still_passes_sampling_params():
    # The sampling behavior itself must not be dropped with the kwarg.
    for call in _generate_calls():
        kwarg_names = {kw.arg for kw in call.keywords}
        assert {"do_sample", "temperature", "top_p",
                "max_new_tokens"} <= kwarg_names


# ---------------------------------------------------------------------------
# _response_token_len: length-debiased base margin twin (run 5, 2026-09-29).
# Raw logprob sums confound length (chosen responses skew longer), so the
# base margin eval also reports per-token-normalized agreement.
# ---------------------------------------------------------------------------

from dpo_weekend.evaluate import _response_token_len


def test_response_token_len_counts_response_tokens_only():
    tok = _NewStyleTok([1, 2, 3], [1, 2, 3, 4, 5])
    assert _response_token_len(tok, [{"role": "user"}],
                               [{"role": "assistant"}]) == 2


def test_response_token_len_old_style():
    tok = _OldStyleTok([1, 2], [1, 2, 3, 4])
    assert _response_token_len(tok, [{"role": "user"}],
                               [{"role": "assistant"}]) == 2
