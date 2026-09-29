"""Preference data: load, filter, split, and persist.

ultrafeedback_binarized stores chosen/rejected as message lists that still
contain the prompt as their first turn ("implicit prompt" format). We run
TRL's extract_prompt to get explicit {"prompt", "chosen", "rejected"} message
lists, which is exactly the conversational format TRL's DPOTrainer expects.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dpo_weekend import utils


def messages_to_text(messages) -> str:
    """Flatten a message list to plain text.

    Only used for length stats and eyeball inspection, never for training.
    Training keeps the message lists so TRL can apply the chat template.
    """
    if isinstance(messages, str):
        return messages
    parts = []
    for m in messages:
        if isinstance(m, dict):
            parts.append(m.get("content", ""))
        else:
            parts.append(str(m))
    return "\n".join(p for p in parts if p).strip()


def filter_pair(prompt: str, chosen: str, rejected: str, max_tokens: int,
                count_tokens) -> bool:
    """Keep a pair only if both responses fit the sequence budget.

    count_tokens is injected so this stays testable without a tokenizer.
    Identical chosen/rejected pairs are dropped in a separate counted step
    (drop_identical_pairs), not here, so this stays a pure length check.
    """
    return (count_tokens(chosen) <= max_tokens
            and count_tokens(rejected) <= max_tokens
            and len(chosen.strip()) > 0
            and len(rejected.strip()) > 0)


def drop_identical_pairs(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop pairs where chosen and rejected are the same after normalize."""
    kept, dropped = [], 0
    for row in rows:
        if messages_to_text(row["chosen"]) == messages_to_text(row["rejected"]):
            dropped += 1
        else:
            kept.append(row)
    return kept, dropped


def dedupe_by_prompt(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop near-duplicate prompts (whitespace/case normalized). Keeps first."""
    seen: set[str] = set()
    kept, dropped = [], 0
    for row in rows:
        key = " ".join(messages_to_text(row["prompt"]).lower().split())
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def make_splits(pairs: list[dict], seed: int, n_train: int,
                n_margin_eval: int, n_gen: int) -> dict[str, list[dict]]:
    """Seeded split with no prompt overlap between the three sets.

    Generation prompts are drawn first so the margin-eval set never leaks
    into what the judge sees, and vice versa.
    """
    rng = random.Random(seed)
    pool = pairs[:]
    rng.shuffle(pool)
    gen = pool[:n_gen]
    margin = pool[n_gen:n_gen + n_margin_eval]
    train = pool[n_gen + n_margin_eval:n_gen + n_margin_eval + n_train]
    if len(train) < n_train or len(margin) < n_margin_eval or len(gen) < n_gen:
        raise ValueError(
            f"not enough pairs after filtering: need "
            f"{n_train + n_margin_eval + n_gen}, have {len(pool)}")
    return {"train": train, "margin_eval": margin, "generation": gen}


def save_splits(splits: dict[str, list[dict]], out_dir: Path, seed: int,
                extra: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"seed": seed}
    if extra:
        manifest.update(extra)
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest[name] = {"file": path.name, "count": len(rows)}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


def _as_messages(value) -> list[dict]:
    """Coerce a dataset value into a list of {"role", "content"} dicts."""
    if value is None:
        return []  # malformed row; counted as empty downstream, not a crash
    if isinstance(value, str):
        return [{"role": "assistant", "content": value}]
    msgs = []
    for m in value:
        if isinstance(m, dict) and "content" in m:
            msgs.append({"role": m.get("role", "user"),
                         "content": m["content"]})
    return msgs


def _pick_split_name(ds_dict) -> str:
    """The prefs split is "train" on the HuggingFaceH4 card, "train_prefs"
    on some mirrors. Take whichever exists."""
    available = set(ds_dict.keys())
    for name in ("train_prefs", "train"):
        if name in available:
            return name
    raise ValueError(f"no usable split in {sorted(available)}")


def build_splits_from_hf(dataset_name: str, out_dir: Path, seed: int,
                         n_train: int, n_margin_eval: int, n_gen: int,
                         max_seq_length: int, model_name: str,
                         trust_remote_code: bool = False
                         ) -> dict[str, list[dict]]:
    """End-to-end: download, normalize, filter, split, persist.

    Returns the splits dict. Writes JSONL splits, manifest.json, and
    data_stats.json to out_dir.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    builder = load_dataset(dataset_name)
    split_name = _pick_split_name(builder)
    ds = builder[split_name]
    print(f"dataset: {dataset_name} split={split_name} rows={len(ds)}")

    tok = AutoTokenizer.from_pretrained(model_name,
                                        trust_remote_code=trust_remote_code)

    # TRL's extract_prompt handles the implicit-prompt format. Import lazily
    # so the CPU smoke test never needs trl installed.
    try:
        from trl import extract_prompt as trl_extract_prompt
        use_trl_extract = True
    except ImportError:
        use_trl_extract = False
        print("warning: trl not installed, using fallback prompt extraction")

    def count_tokens(prompt_msgs: list[dict], resp_msgs: list[dict]) -> int:
        ids = utils.chat_template_ids(tok, prompt_msgs + resp_msgs)
        return len(ids)

    stats = {"raw_rows": len(ds), "empty": 0, "identical": 0,
             "dupe_prompt": 0, "overlong": 0}
    rows: list[dict] = []
    for r in ds:
        chosen = _as_messages(r.get("chosen"))
        rejected = _as_messages(r.get("rejected"))
        if not chosen or not rejected:
            stats["empty"] += 1
            continue
        if use_trl_extract:
            ex = trl_extract_prompt({"chosen": chosen, "rejected": rejected})
            prompt_msgs, chosen, rejected = (ex["prompt"], ex["chosen"],
                                            ex["rejected"])
        else:
            # Fallback: first turn of chosen is the prompt.
            prompt_msgs = [chosen[0]]
            chosen, rejected = chosen[1:], rejected[1:]
        prompt_text = messages_to_text(prompt_msgs)
        if not prompt_text.strip():
            stats["empty"] += 1
            continue
        rows.append({"prompt": prompt_msgs, "chosen": chosen,
                     "rejected": rejected})

    rows, n_identical = drop_identical_pairs(rows)
    stats["identical"] = n_identical

    rows, n_dupes = dedupe_by_prompt(rows)
    stats["dupe_prompt"] = n_dupes

    kept: list[dict] = []
    for row in rows:
        if (not messages_to_text(row["chosen"]).strip()
                or not messages_to_text(row["rejected"]).strip()):
            stats["empty"] += 1
            continue
        if (count_tokens(row["prompt"], row["chosen"]) > max_seq_length
                or count_tokens(row["prompt"], row["rejected"])
                > max_seq_length):
            stats["overlong"] += 1
            continue
        kept.append(row)

    splits = make_splits(kept, seed, n_train, n_margin_eval, n_gen)
    for name, rs in splits.items():
        for i, row in enumerate(rs):
            row["prompt_id"] = f"{name}-{i:04d}"

    stats.update({f"kept_{n}": len(rs) for n, rs in splits.items()})
    stats["seed"] = seed

    # Length-bias check: chosen responses skew longer in most pref datasets.
    def mean_len(rs, key):
        lens = [len(messages_to_text(r[key])) for r in rs]
        return sum(lens) / len(lens) if lens else 0.0
    stats["mean_chars_chosen_train"] = mean_len(splits["train"], "chosen")
    stats["mean_chars_rejected_train"] = mean_len(splits["train"], "rejected")

    save_splits(splits, out_dir, seed, extra={"source_dataset": dataset_name,
                                              "split_used": split_name,
                                              "max_seq_length": max_seq_length,
                                              "model": model_name})
    (out_dir / "data_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"chosen mean chars: {stats['mean_chars_chosen_train']:.0f} | "
          f"rejected mean chars: {stats['mean_chars_rejected_train']:.0f} "
          f"(watch the length bias)")
    return splits


def main() -> None:
    import yaml  # local import: pyyaml is a light CPU dependency

    parser = argparse.ArgumentParser(
        description="Build seeded preference-data splits from HF.")
    parser.add_argument("--config", required=True,
                        help="Path to a YAML config in configs/.")
    parser.add_argument("--out-dir", required=True,
                        help="Where to write splits + manifest + stats.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    utils.validate_config(config)
    data_cfg = config["data"]
    build_splits_from_hf(
        dataset_name=data_cfg["dataset"],
        out_dir=Path(args.out_dir),
        seed=int(data_cfg.get("seed", config.get("seed", 42))),
        n_train=int(data_cfg["train_pairs"]),
        n_margin_eval=int(data_cfg["margin_eval_pairs"]),
        n_gen=int(data_cfg["generation_prompts"]),
        max_seq_length=int(data_cfg["max_seq_length"]),
        model_name=config["model"]["name"],
        trust_remote_code=bool(config["model"].get("trust_remote_code",
                                                   False)),
    )


if __name__ == "__main__":
    main()
