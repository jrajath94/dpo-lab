"""Small helpers. Pure functions, no GPU, unit-testable."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


def set_seed(seed: int) -> None:
    """Seed Python and NumPy. Torch seeding happens in train.py (GPU box)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass  # numpy comes with torch on the GPU box


def implicit_reward_margin(beta: float, logp_chosen: float,
                           logp_rejected: float, logp_ref_chosen: float,
                           logp_ref_rejected: float) -> float:
    """DPO's implicit reward margin for one preference pair.

    r(x, y) = beta * log(pi(y|x) / pi_ref(y|x)).
    margin  = r(chosen) - r(rejected).
    Positive margin means the policy prefers the chosen response.
    """
    return beta * ((logp_chosen - logp_ref_chosen)
                   - (logp_rejected - logp_ref_rejected))


def dpo_loss_from_margin(margin: float) -> float:
    """The per-example DPO loss: -log(sigmoid(margin)).

    Same formula TRL optimizes. Useful for sanity-checking logged losses
    by hand on a few examples.
    """
    return -math.log(1.0 / (1.0 + math.exp(-margin)))


def mean_kl_to_reference(logp: list[float],
                         logp_ref: list[float]) -> float:
    """Mean per-token KL estimate: E[log pi - log pi_ref].

    This is the quantity beta is supposed to keep small. If it explodes,
    DPO is pushing the policy too far from the reference model.
    """
    if not logp or len(logp) != len(logp_ref):
        raise ValueError("logp lists must be non-empty and aligned")
    return sum(a - b for a, b in zip(logp, logp_ref)) / len(logp)


def wilson_interval(wins: int, n: int,
                    z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Use on the final win rate instead of the naive p +- z*sqrt(p(1-p)/n),
    which breaks down near 0/1 and for small n. wins=0 gives lo=0.0
    exactly; wins=n gives hi=1.0 exactly.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= wins <= n:
        raise ValueError(f"wins must satisfy 0 <= wins <= n, got {wins}/{n}")
    if z <= 0:
        raise ValueError("z must be positive")
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# Dotted paths every run config must define. lora.r joins the list unless
# this is a full-finetune config (see validate_config).
REQUIRED_CONFIG_PATHS = [
    "model.name",
    "data.dataset",
    "data.train_pairs",
    "data.margin_eval_pairs",
    "data.generation_prompts",
    "data.max_seq_length",
    "dpo.beta",
    "dpo.learning_rate",
    "dpo.per_device_batch",
    "dpo.gradient_accumulation_steps",
    "dpo.num_train_epochs",
    "output.dir",
    "eval.judge_model",
]


def validate_config(config: dict) -> None:
    """Fail fast on a config that cannot run. Raises ValueError naming
    every missing required dotted path, so one run surfaces all typos."""
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    paths = list(REQUIRED_CONFIG_PATHS)
    if not config.get("full_finetune", False):
        paths.append("lora.r")
    missing = []
    for path in paths:
        node: object = config
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                missing.append(path)
                break
            node = node[part]
    if missing:
        raise ValueError("config is missing required keys: "
                         + ", ".join(missing))


def chat_template_ids(tok, messages: list[dict], **kwargs) -> list[int]:
    """apply_chat_template with tokenize=True, robust across transformers.

    transformers<5 returns list[int]; transformers>=5 returns a
    BatchEncoding (dict-like), whose bare list() is the FIELD NAMES
    (['input_ids', 'attention_mask']) — silently wrong ids that crashed
    run 4's eval and broke run 4's length pre-filter. Unwrap either way,
    and fail loudly if the result is not integer ids.
    """
    out = tok.apply_chat_template(messages, tokenize=True, **kwargs)
    if not isinstance(out, list):
        out = out["input_ids"]  # transformers>=5: BatchEncoding
    ids = list(out)
    if not ids or any(not isinstance(i, int) for i in ids):
        raise TypeError(
            "chat template did not return a non-empty int id list; got "
            f"{type(out).__name__}"
        )
    return ids


def hf_model_kwargs(model_cfg: dict) -> dict:
    """Name, pinned revision, and trust flag for from_pretrained calls.

    Pure helper so the hub-revision pin is tested without transformers.
    revision=None is passed through untouched, which means "hub default";
    the shipped config pins an exact revision so weight updates on the
    repo cannot silently change the base model between runs.
    """
    return {
        "name": model_cfg["name"],
        "revision": model_cfg.get("revision"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }


def sha256_of_file(path: str | Path) -> str:
    """Hex digest of a file's bytes. Used to pin the exact config per run."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(obj: dict, path: str | Path) -> None:
    """Write a dict as pretty JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply dotted-path CLI overrides, e.g. ["dpo.beta=0.2"].

    Values are parsed as int, float, bool, or left as strings.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like a.b=c, got: {item!r}")
        key_path, raw = item.split("=", 1)
        value: object = raw
        lowered = raw.lower()
        if lowered in ("true", "false"):
            value = lowered == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        node = config
        parts = key_path.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise KeyError(f"override path not found: {key_path!r}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"override key not found: {key_path!r}")
        node[parts[-1]] = value
    return config
