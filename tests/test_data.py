"""Hardening tests for dpo_weekend.data pure functions."""

import pytest

from dpo_weekend.data import (_as_messages, dedupe_by_prompt,
                              drop_identical_pairs, filter_pair,
                              make_splits, messages_to_text)


def _pair(i: int) -> dict:
    return {"prompt": f"prompt {i}", "chosen": "good answer",
            "rejected": "bad answer"}


# ---------------------------------------------------------------------------
# messages_to_text
# ---------------------------------------------------------------------------

def test_messages_to_text_string_passthrough():
    assert messages_to_text("already text") == "already text"


def test_messages_to_text_joins_contents():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    assert messages_to_text(msgs) == "hi\nhello"


def test_messages_to_text_skips_empty_and_missing_content():
    msgs = [{"role": "user", "content": ""},
            {"role": "assistant"},  # no content key
            "raw string",
            {"role": "user", "content": "kept"}]
    assert messages_to_text(msgs) == "raw string\nkept"


def test_messages_to_text_empty_list_is_empty():
    assert messages_to_text([]) == ""


# ---------------------------------------------------------------------------
# filter_pair (fake count_tokens injected)
# ---------------------------------------------------------------------------

def test_filter_pair_accepts_short_pair():
    count = lambda s: len(s.split())  # noqa: E731
    assert filter_pair("p", "short", "short", max_tokens=10,
                       count_tokens=count)


def test_filter_pair_rejects_oversize_either_side():
    count = lambda s: len(s.split())  # noqa: E731
    assert not filter_pair("p", "word " * 50, "short", max_tokens=10,
                           count_tokens=count)
    assert not filter_pair("p", "short", "word " * 50, max_tokens=10,
                           count_tokens=count)


def test_filter_pair_boundary_is_inclusive():
    count = lambda s: len(s.split())  # noqa: E731
    assert filter_pair("p", "one two", "three four", max_tokens=2,
                       count_tokens=count)


def test_filter_pair_rejects_empty_responses():
    count = lambda s: len(s.split())  # noqa: E731
    assert not filter_pair("p", "", "short", max_tokens=10,
                           count_tokens=count)
    assert not filter_pair("p", "short", "   ", max_tokens=10,
                           count_tokens=count)


# ---------------------------------------------------------------------------
# drop_identical_pairs
# ---------------------------------------------------------------------------

def test_drop_identical_pairs_drops_and_counts():
    rows = [
        {"prompt": "p1", "chosen": "same", "rejected": "same"},
        {"prompt": "p2", "chosen": "a", "rejected": "b"},
        {"prompt": "p3", "chosen": "x", "rejected": "x"},
    ]
    kept, dropped = drop_identical_pairs(rows)
    assert dropped == 2
    assert [r["prompt"] for r in kept] == ["p2"]


def test_drop_identical_pairs_normalizes_message_lists():
    rows = [{"prompt": "p",
             "chosen": [{"role": "assistant", "content": "hi "}],
             "rejected": "hi"}]
    kept, dropped = drop_identical_pairs(rows)
    assert dropped == 1 and kept == []


def test_drop_identical_pairs_keeps_everything_when_all_differ():
    rows = [_pair(i) for i in range(10)]
    kept, dropped = drop_identical_pairs(rows)
    assert dropped == 0 and len(kept) == 10


# ---------------------------------------------------------------------------
# dedupe_by_prompt
# ---------------------------------------------------------------------------

def test_dedupe_by_prompt_drops_case_and_whitespace_dupes():
    rows = [
        {"prompt": "What is AI?", "chosen": "a", "rejected": "b"},
        {"prompt": "what  is   ai?", "chosen": "c", "rejected": "d"},
        {"prompt": "What is AI? ", "chosen": "e", "rejected": "f"},
        {"prompt": "Something else", "chosen": "g", "rejected": "h"},
    ]
    kept, dropped = dedupe_by_prompt(rows)
    assert dropped == 2
    assert [r["chosen"] for r in kept] == ["a", "g"]  # keeps first


def test_dedupe_by_prompt_keeps_distinct_prompts():
    rows = [_pair(i) for i in range(20)]
    kept, dropped = dedupe_by_prompt(rows)
    assert dropped == 0 and len(kept) == 20


# ---------------------------------------------------------------------------
# make_splits
# ---------------------------------------------------------------------------

def test_splits_have_no_prompt_overlap():
    pairs = [_pair(i) for i in range(2000)]
    splits = make_splits(pairs, seed=42, n_train=1000, n_margin_eval=200,
                         n_gen=50)
    prompts = {name: {r["prompt"] for r in rows}
               for name, rows in splits.items()}
    assert prompts["train"].isdisjoint(prompts["margin_eval"])
    assert prompts["train"].isdisjoint(prompts["generation"])
    assert prompts["margin_eval"].isdisjoint(prompts["generation"])


def test_splits_are_deterministic_across_calls():
    pairs = [_pair(i) for i in range(2000)]
    a = make_splits(pairs, seed=42, n_train=1000, n_margin_eval=200, n_gen=50)
    b = make_splits(pairs, seed=42, n_train=1000, n_margin_eval=200, n_gen=50)
    for name in ("train", "margin_eval", "generation"):
        assert [r["prompt"] for r in a[name]] == [r["prompt"] for r in b[name]]


def test_splits_have_exact_sizes():
    pairs = [_pair(i) for i in range(2000)]
    splits = make_splits(pairs, seed=7, n_train=1000, n_margin_eval=200,
                         n_gen=50)
    assert len(splits["train"]) == 1000
    assert len(splits["margin_eval"]) == 200
    assert len(splits["generation"]) == 50


def test_splits_cover_disjoint_pool_portion():
    pairs = [_pair(i) for i in range(300)]
    splits = make_splits(pairs, seed=1, n_train=100, n_margin_eval=50,
                         n_gen=25)
    all_prompts = [r["prompt"] for rows in splits.values() for r in rows]
    assert len(all_prompts) == len(set(all_prompts)) == 175


def test_splits_raise_when_pool_too_small():
    pairs = [_pair(i) for i in range(100)]
    with pytest.raises(ValueError, match="not enough pairs"):
        make_splits(pairs, seed=1, n_train=80, n_margin_eval=15, n_gen=10)


def test_splits_different_seeds_differ():
    pairs = [_pair(i) for i in range(2000)]
    a = make_splits(pairs, seed=1, n_train=1000, n_margin_eval=200, n_gen=50)
    b = make_splits(pairs, seed=2, n_train=1000, n_margin_eval=200, n_gen=50)
    assert [r["prompt"] for r in a["train"]] != \
        [r["prompt"] for r in b["train"]]


def test_splits_do_not_mutate_input_order():
    pairs = [_pair(i) for i in range(500)]
    before = [r["prompt"] for r in pairs]
    make_splits(pairs, seed=3, n_train=200, n_margin_eval=50, n_gen=25)
    assert [r["prompt"] for r in pairs] == before


# ---------------------------------------------------------------------------
# _as_messages
# ---------------------------------------------------------------------------

def test_as_messages_string_becomes_assistant_turn():
    assert _as_messages("hi") == [{"role": "assistant", "content": "hi"}]


def test_as_messages_none_becomes_empty():
    # a malformed dataset row with a missing response must not crash
    # the data build; it is counted as empty downstream
    assert _as_messages(None) == []


def test_as_messages_keeps_role_and_content():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"}]
    assert _as_messages(msgs) == msgs


def test_as_messages_skips_entries_without_content():
    assert _as_messages([{"role": "user"}, {"content": "x"}]) == \
        [{"role": "user", "content": "x"}]
