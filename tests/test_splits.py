"""Split tests. Pure functions only, no torch, no GPU, no network.

Run: pytest tests/
"""

from dpo_weekend.data import filter_pair, make_splits, messages_to_text
from dpo_weekend.utils import (dpo_loss_from_margin,
                               implicit_reward_margin, mean_kl_to_reference)


def _pair(i: int) -> dict:
    return {"prompt": f"prompt {i}", "chosen": "good answer",
            "rejected": "bad answer"}


def test_splits_have_no_prompt_overlap():
    pairs = [_pair(i) for i in range(2000)]
    splits = make_splits(pairs, seed=42, n_train=1000, n_margin_eval=200,
                         n_gen=50)
    prompts = {name: {r["prompt"] for r in rows}
               for name, rows in splits.items()}
    assert prompts["train"].isdisjoint(prompts["margin_eval"])
    assert prompts["train"].isdisjoint(prompts["generation"])
    assert prompts["margin_eval"].isdisjoint(prompts["generation"])


def test_splits_are_deterministic():
    pairs = [_pair(i) for i in range(2000)]
    a = make_splits(pairs, seed=42, n_train=1000, n_margin_eval=200, n_gen=50)
    b = make_splits(pairs, seed=42, n_train=1000, n_margin_eval=200, n_gen=50)
    assert [r["prompt"] for r in a["train"]] == \
        [r["prompt"] for r in b["train"]]


def test_splits_have_expected_sizes():
    pairs = [_pair(i) for i in range(2000)]
    splits = make_splits(pairs, seed=7, n_train=1000, n_margin_eval=200,
                         n_gen=50)
    assert len(splits["train"]) == 1000
    assert len(splits["margin_eval"]) == 200
    assert len(splits["generation"]) == 50


def test_filter_pair_rejects_oversize():
    count = lambda s: len(s.split())  # noqa: E731 - word-count stand-in
    assert filter_pair("p", "short", "short", max_tokens=10,
                       count_tokens=count)
    assert not filter_pair("p", "word " * 50, "short", max_tokens=10,
                           count_tokens=count)
    assert not filter_pair("p", "short", "   ", max_tokens=10,
                           count_tokens=count)


def test_messages_to_text_handles_strings_and_lists():
    assert messages_to_text("already text") == "already text"
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    assert messages_to_text(msgs) == "hi\nhello"


def test_margin_sign_matches_preference():
    # Policy likes chosen more than rejected, relative to reference.
    m = implicit_reward_margin(beta=0.1, logp_chosen=-2.0,
                               logp_rejected=-5.0, logp_ref_chosen=-3.0,
                               logp_ref_rejected=-3.0)
    assert m > 0


def test_dpo_loss_decreases_with_margin():
    assert dpo_loss_from_margin(2.0) < dpo_loss_from_margin(0.0)


def test_mean_kl_zero_for_identical_models():
    logp = [-1.0, -2.0, -3.0]
    assert mean_kl_to_reference(logp, logp) == 0.0
