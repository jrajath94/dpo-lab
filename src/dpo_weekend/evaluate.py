"""Evals. Every function here must be able to fail loudly.

Three evals plus diagnostics:
  1. margin_eval      - held-out preference agreement, before vs after DPO
  2. generation_eval  - blinded A/B win rate with a fixed external judge
  3. regression_guard - small public-benchmark slice, before vs after DPO
Diagnostics (logprob curves, margin histogram, KL curve) live in plots.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from . import utils
from .data import messages_to_text


# ---------------------------------------------------------------------------
# Pure helpers (CPU-safe, covered by unit tests)
# ---------------------------------------------------------------------------

def margin_accuracy(margins: list[float]) -> float:
    """Fraction of held-out pairs where the model prefers the chosen one."""
    if not margins:
        raise ValueError("empty margin list")
    return sum(1 for m in margins if m > 0) / len(margins)


def length_controlled_win_rate(labels: list[dict]) -> dict:
    """Win rate split by which response was longer.

    Judge bias toward longer answers is the classic confounder in
    generation evals. Reporting the split is more honest than one number.
    Each label: {"winner": "dpo"|"base"|"tie", "dpo_longer": bool}.
    Ties count as half a win, the standard AlpacaEval convention.
    """
    def rate(rows: list[dict]) -> float | None:
        if not rows:
            return None
        wins = sum(1 for r in rows if r["winner"] == "dpo")
        ties = sum(1 for r in rows if r["winner"] == "tie")
        return (wins + 0.5 * ties) / len(rows)

    return {
        "overall": rate(labels),
        "when_dpo_longer": rate([r for r in labels if r["dpo_longer"]]),
        "when_base_longer": rate([r for r in labels if not r["dpo_longer"]]),
        "n": len(labels),
        "n_ties": sum(1 for r in labels if r["winner"] == "tie"),
        "n_parse_errors": sum(1 for r in labels
                              if r.get("parse_error", False)),
    }


# ---------------------------------------------------------------------------
# Model loading (GPU box)
# ---------------------------------------------------------------------------

def load_eval_model(model_name: str, adapter_dir: str | None,
                    trust_remote_code: bool = False,
                    load_in_4bit: bool = True,
                    revision: str | None = None):
    """Load the base model, optionally with a trained LoRA adapter.

    adapter_dir=None means the base model (the "before" number).
    revision pins the exact hub weights; None means hub default.
    """
    import torch
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tok = AutoTokenizer.from_pretrained(model_name,
                                        revision=revision,
                                        trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = (BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
           if load_in_4bit else None)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code, device_map="auto")
    model.eval()
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# 1. Held-out preference margin eval
# ---------------------------------------------------------------------------

def _model_device(model):
    """First parameter's device. Works for device_map="auto" models,
    where model.device is not reliable."""
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _response_span(tok, prompt_msgs: list[dict],
                   response_msgs: list[dict]) -> tuple[list[int], int]:
    """Token ids for prompt+response and the index where response starts.

    Builds the full conversation through the chat template, and the prompt
    with the generation header separately. If the two disagree (some
    templates do odd things), falls back to tokenizing the response text
    on its own. Either way the caller masks the prompt tokens.
    """
    prompt_ids = utils.chat_template_ids(
        tok, prompt_msgs, add_generation_prompt=True)
    full_ids = utils.chat_template_ids(tok, prompt_msgs + response_msgs)
    if full_ids[:len(prompt_ids)] == prompt_ids:
        return full_ids, len(prompt_ids)
    resp_ids = tok.encode(messages_to_text(response_msgs),
                          add_special_tokens=False)
    return prompt_ids + list(resp_ids), len(prompt_ids)


def sequence_logprob(model, tok, prompt_msgs: list[dict],
                     response_msgs: list[dict]) -> float:
    """Sum of log probs of the response tokens given the prompt."""
    import torch

    input_ids, resp_start = _response_span(tok, prompt_msgs, response_msgs)
    ids = torch.tensor([input_ids], device=_model_device(model))
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    logp = torch.log_softmax(logits, dim=-1)
    # logits[i] predicts token i+1, so response token at position j
    # (j >= resp_start) is scored by logp[j - 1].
    total = 0.0
    for j in range(resp_start, len(input_ids)):
        total += float(logp[j - 1, input_ids[j]])
    return total


def _response_token_len(tok, prompt_msgs: list[dict],
                        response_msgs: list[dict]) -> int:
    """Number of response tokens _response_span would score."""
    ids, resp_start = _response_span(tok, prompt_msgs, response_msgs)
    return len(ids) - resp_start


def run_margin_eval(adapter_dir: str | None, splits_dir: str, beta: float,
                    out_path: str, config: dict) -> dict:
    """Held-out preference agreement, before vs after DPO.

    Base model (adapter_dir=None): plain logprob agreement, the fraction of
    pairs where logp_base(chosen) > logp_base(rejected). Raw sums confound
    length (chosen responses skew longer; more tokens = more negative sum),
    so a length-normalized twin (mean logprob per response token) is also
    reported as accuracy_len_norm. The DPO number is already length-debiased
    via the reference model, so accuracy_len_norm is the fair "before".
    DPO model: implicit-reward margin accuracy, the fraction of pairs where
    beta * (log pi(chosen)/pi_ref(chosen) - log pi(rejected)/pi_ref(rejected))
    is positive. Both are agreement rates on the same held-out set; the DPO
    number has to beat the length-normalized base number for the run to
    count.
    """
    model_cfg = config["model"]
    hf_kwargs = utils.hf_model_kwargs(model_cfg)
    trust_rc = hf_kwargs["trust_remote_code"]
    revision = hf_kwargs["revision"]
    policy, tok = load_eval_model(hf_kwargs["name"], adapter_dir,
                                  trust_remote_code=trust_rc,
                                  revision=revision)

    rows = utils.load_jsonl(Path(splits_dir) / "margin_eval.jsonl")
    margins, details = [], []
    margins_len_norm = []  # base only: length-debiased twin margins
    if adapter_dir is None:
        metric = "base_logprob_diff"
        ref = None
    else:
        metric = "dpo_implicit_margin"
        ref, _ = load_eval_model(hf_kwargs["name"], None,
                                 trust_remote_code=trust_rc,
                                 revision=revision)

    for row in rows:
        lp_c = sequence_logprob(policy, tok, row["prompt"], row["chosen"])
        lp_r = sequence_logprob(policy, tok, row["prompt"], row["rejected"])
        if ref is None:
            m = lp_c - lp_r
            # Length-debiased twin: mean logprob per response token.
            n_c = _response_token_len(tok, row["prompt"], row["chosen"])
            n_r = _response_token_len(tok, row["prompt"], row["rejected"])
            m_len_norm = lp_c / n_c - lp_r / n_r
            margins_len_norm.append(m_len_norm)
        else:
            lr_c = sequence_logprob(ref, tok, row["prompt"], row["chosen"])
            lr_r = sequence_logprob(ref, tok, row["prompt"], row["rejected"])
            m = utils.implicit_reward_margin(beta, lp_c, lp_r, lr_c, lr_r)
        margins.append(m)
        details.append({"prompt_id": row["prompt_id"], "margin": m})

    result = {
        "adapter": adapter_dir or "base",
        "metric": metric,
        "beta": beta,
        "n": len(margins),
        "accuracy": margin_accuracy(margins),
        "mean_margin": sum(margins) / len(margins),
        "margins": margins,
        "per_pair": details,
    }
    if margins_len_norm:
        # Base only. accuracy_len_norm is the fair "before" number:
        # raw sums penalize the (longer) chosen responses.
        result["accuracy_len_norm"] = margin_accuracy(margins_len_norm)
        result["mean_margin_len_norm"] = (sum(margins_len_norm)
                                          / len(margins_len_norm))
    utils.save_json(result, out_path)
    base_extra = ""
    if margins_len_norm:
        base_extra = (f" len_norm_accuracy={result['accuracy_len_norm']:.3f}")
    print(f"margin eval [{result['adapter']}] ({metric}): "
          f"accuracy={result['accuracy']:.3f}{base_extra} "
          f"mean_margin={result['mean_margin']:.4f} n={result['n']}")
    return result


# ---------------------------------------------------------------------------
# 2. Generation eval + blinded judging
# ---------------------------------------------------------------------------

def run_generation_eval(adapter_dir: str | None, splits_dir: str,
                        out_path: str, config: dict) -> str:
    """Generate answers to the held-out prompts with fixed decoding params.

    Writes {"prompt_id", "prompt", "text"} JSONL. Judging is a separate step
    so the labels stay auditable.
    """
    import torch

    model_cfg = config["model"]
    eval_cfg = config.get("eval", {})
    hf_kwargs = utils.hf_model_kwargs(model_cfg)
    model, tok = load_eval_model(
        hf_kwargs["name"], adapter_dir,
        trust_remote_code=hf_kwargs["trust_remote_code"],
        revision=hf_kwargs["revision"])

    temperature = float(eval_cfg.get("temperature", 0.7))
    top_p = float(eval_cfg.get("top_p", 0.9))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 512))

    rows = utils.load_jsonl(Path(splits_dir) / "generation.jsonl")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = _model_device(model)
    with open(out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            enc = tok.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True,
                return_tensors="pt")
            # transformers<5 returned a tensor here; >=5 returns a
            # BatchEncoding. Unwrap so generate() and .shape both work.
            prompt_ids = (enc if isinstance(enc, torch.Tensor)
                          else enc["input_ids"]).to(device)
            # transformers<5 accepted generate(..., generator=gen); 5.x
            # rejects the kwarg outright (run 5, 2026-09-29). Seed the
            # global RNGs instead for the same per-generation determinism.
            seed_i = int(config.get("seed", 42)) + i
            torch.manual_seed(seed_i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_i)
            with torch.no_grad():
                out_ids = model.generate(
                    prompt_ids, do_sample=True, temperature=temperature,
                    top_p=top_p, max_new_tokens=max_new_tokens,
                    pad_token_id=tok.pad_token_id)
            text = tok.decode(out_ids[0][prompt_ids.shape[1]:],
                              skip_special_tokens=True).strip()
            f.write(json.dumps({"prompt_id": row["prompt_id"],
                                "prompt": messages_to_text(row["prompt"]),
                                "text": text},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} generations to {out}")
    return str(out)


JUDGE_TEMPLATE = """You are judging two answers to the same question.
Pick the better answer. Judge only on helpfulness, correctness, and clarity.
Do not favor longer answers. Do not favor the answer listed first.

Question:
{prompt}

Answer A:
{a}

Answer B:
{b}

Reply with exactly this JSON and nothing else:
{{"winner": "A", "reason": "one short sentence"}}
Use "tie" as the winner only if the answers are truly equal in quality."""

JUDGE_SYSTEM = ("You are a strict, impartial judge of answer quality. "
                "You always reply with exactly the requested JSON.")


def parse_judge_content(content: str) -> dict:
    """Parse one judge reply into a normalized label. Pure function.

    Returns {"raw", "winner_ab", "parse_error", "reason"}. winner_ab is
    normalized to A/B/TIE; anything unparseable becomes TIE with
    parse_error=True. Reason is truncated to 300 chars.
    """
    label: dict = {"raw": content, "parse_error": False}
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        parsed = json.loads(content[start:end])
        winner = str(parsed.get("winner", "")).strip().upper()
        label["winner_ab"] = winner if winner in ("A", "B", "TIE") else "TIE"
        if winner not in ("A", "B", "TIE"):
            label["parse_error"] = True
        label["reason"] = str(parsed.get("reason", ""))[:300]
    except (ValueError, KeyError, AttributeError):
        label["winner_ab"] = "TIE"
        label["parse_error"] = True
    return label


def _judge_call(api_key: str, judge_model: str, prompt: str,
                a: str, b: str) -> dict:
    """One blinded A/B call through OpenRouter. Returns the parsed label."""
    import requests

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": judge_model,
              "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                           {"role": "user",
                            "content": JUDGE_TEMPLATE.format(prompt=prompt,
                                                             a=a, b=b)}],
              "temperature": 0.0,
              "max_tokens": 200},
        timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return parse_judge_content(content)


def run_judge(base_gens: str, dpo_gens: str, out_path: str,
              config: dict) -> dict:
    """Blinded A/B judging with order swapping to cancel position bias.

    For each prompt, the judge sees (base, dpo) and (dpo, base). Every raw
    label lands in out_path. The judge model name and the exact template
    are recorded there too. A judge change invalidates old numbers.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. The judge step needs it; "
            "run the margin and generation evals without it.")
    eval_cfg = config.get("eval", {})
    judge_model = os.environ.get(
        "JUDGE_MODEL", eval_cfg.get("judge_model", "openai/gpt-4o-mini"))

    base = {r["prompt_id"]: r for r in utils.load_jsonl(base_gens)}
    dpo = {r["prompt_id"]: r for r in utils.load_jsonl(dpo_gens)}
    common = [pid for pid in base if pid in dpo]
    if not common:
        raise ValueError("no shared prompt_ids between generation files")

    labels = []
    for pid in common:
        b, d = base[pid], dpo[pid]
        # Order 1: A=base, B=dpo.
        lab1 = _judge_call(api_key, judge_model, b["prompt"],
                           b["text"], d["text"])
        lab1.update({"prompt_id": pid, "order": "base_first",
                     "a_is": "base", "b_is": "dpo"})
        # Order 2: A=dpo, B=base.
        lab2 = _judge_call(api_key, judge_model, b["prompt"],
                           d["text"], b["text"])
        lab2.update({"prompt_id": pid, "order": "dpo_first",
                     "a_is": "dpo", "b_is": "base"})
        labels.extend([lab1, lab2])
        time.sleep(0.5)  # be polite to the API

    # Map A/B winners back to base/dpo. _judge_call normalizes winner_ab
    # to exactly A/B/TIE, so anything else here is a bug: fail loudly
    # rather than silently crediting one side.
    mapped = []
    for lab in labels:
        w = lab["winner_ab"]
        if w == "TIE":
            winner = "tie"
        else:
            shown = {"A": lab["a_is"], "B": lab["b_is"]}.get(w)
            if shown not in ("base", "dpo"):
                raise ValueError(
                    f"impossible winner_ab={w!r} for prompt "
                    f"{lab['prompt_id']}")
            winner = shown
        b_row = base[lab["prompt_id"]]
        d_row = dpo[lab["prompt_id"]]
        mapped.append({
            "prompt_id": lab["prompt_id"],
            "order": lab["order"],
            "winner": winner,
            "dpo_longer": len(d_row["text"]) > len(b_row["text"]),
            "parse_error": lab["parse_error"],
            "judge_model": judge_model,
            "reason": lab.get("reason", ""),
        })

    summary = length_controlled_win_rate(mapped)
    summary["judge_model"] = judge_model
    summary["judge_template"] = JUDGE_TEMPLATE
    summary["n_prompts"] = len(common)
    summary["labels"] = mapped
    utils.save_json(summary, out_path)
    print(f"judge [{judge_model}]: overall win rate "
          f"{summary['overall']:.3f} over {summary['n']} comparisons, "
          f"{summary['n_ties']} ties, {summary['n_parse_errors']} parse errors")
    return summary


# ---------------------------------------------------------------------------
# 3. Regression guard: small MMLU slice, before vs after
# ---------------------------------------------------------------------------

def run_regression_guard(adapter_dir: str | None, out_path: str,
                         config: dict) -> dict:
    """200 random MMLU questions, 5-shot, fixed seed. Collapse check only.

    Small moves are noise. A large drop invalidates the run.
    """
    try:
        from lm_eval import simple_evaluate
    except ImportError as e:
        raise RuntimeError(
            "lm-eval is not installed. Install the eval extras: "
            "pip install -e '.[eval]'") from e

    model_cfg = config["model"]
    eval_cfg = config.get("eval", {})
    hf_kwargs = utils.hf_model_kwargs(model_cfg)
    model_args = f"pretrained={hf_kwargs['name']}"
    if hf_kwargs["revision"]:
        model_args += f",revision={hf_kwargs['revision']}"
    if adapter_dir:
        model_args += f",peft={adapter_dir}"
    if hf_kwargs["trust_remote_code"]:
        model_args += ",trust_remote_code=True"

    eval_kwargs = dict(
        model="hf",
        model_args=model_args,
        tasks=["mmlu"],
        num_fewshot=5,
        limit=int(eval_cfg.get("mmlu_limit", 200)),
        batch_size=int(eval_cfg.get("mmlu_batch_size", 8)),
    )
    try:
        results = simple_evaluate(**eval_kwargs,
                                  random_seed=int(config.get("seed", 42)))
    except TypeError:
        # Older lm-eval releases have no random_seed kwarg.
        results = simple_evaluate(**eval_kwargs)
    acc = results["results"]["mmlu"].get("acc,none")
    summary = {"adapter": adapter_dir or "base",
               "task": "mmlu", "num_fewshot": 5,
               "limit": int(eval_cfg.get("mmlu_limit", 200)),
               "acc": acc,
               "full_results": results["results"]["mmlu"]}
    utils.save_json(summary, out_path)
    print(f"regression guard [{summary['adapter']}]: mmlu acc={acc}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run one or all evals.")
    parser.add_argument("--eval", required=True,
                        choices=["margin", "generation", "judge",
                                 "regression", "plots", "all"])
    parser.add_argument("--config", required=True,
                        help="Path to the YAML config used for training.")
    parser.add_argument("--adapter-dir", default=None,
                        help="Trained adapter dir. None = base model.")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--out", required=True,
                        help="Output file or dir, depending on --eval.")
    parser.add_argument("--base-gens", default=None,
                        help="Base generations JSONL (for --eval judge).")
    parser.add_argument("--dpo-gens", default=None,
                        help="DPO generations JSONL (for --eval judge).")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    utils.validate_config(config)

    if args.eval == "margin":
        run_margin_eval(args.adapter_dir, args.splits_dir, args.beta,
                        args.out, config)
    elif args.eval == "generation":
        run_generation_eval(args.adapter_dir, args.splits_dir, args.out,
                            config)
    elif args.eval == "judge":
        if not args.base_gens or not args.dpo_gens:
            raise ValueError("--eval judge needs --base-gens and --dpo-gens")
        run_judge(args.base_gens, args.dpo_gens, args.out, config)
    elif args.eval == "regression":
        run_regression_guard(args.adapter_dir, args.out, config)
    elif args.eval == "plots":
        from .plots import plot_run_diagnostics
        plot_run_diagnostics(args.splits_dir, args.adapter_dir, args.out,
                             config, beta=args.beta)
    elif args.eval == "all":
        if not args.adapter_dir:
            raise ValueError("--eval all needs --adapter-dir")
        out = Path(args.out)
        run_margin_eval(None, args.splits_dir, args.beta,
                        str(out / "margin_base.json"), config)
        run_margin_eval(args.adapter_dir, args.splits_dir, args.beta,
                        str(out / "margin_dpo.json"), config)
        base_g = run_generation_eval(None, args.splits_dir,
                                     str(out / "gens_base.jsonl"), config)
        dpo_g = run_generation_eval(args.adapter_dir, args.splits_dir,
                                    str(out / "gens_dpo.jsonl"), config)
        run_judge(base_g, dpo_g, str(out / "judge_labels.json"), config)
        run_regression_guard(None, str(out / "mmlu_base.json"), config)
        run_regression_guard(args.adapter_dir, str(out / "mmlu_dpo.json"),
                             config)
        from .plots import plot_run_diagnostics
        plot_run_diagnostics(args.splits_dir, args.adapter_dir,
                             str(out / "plots"), config, beta=args.beta)


if __name__ == "__main__":
    main()
