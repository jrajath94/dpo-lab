"""Training driver: config YAML in, TRL DPOTrainer out.

Deliberately thin. The point of this project is DPO mechanics and evals,
not re-implementing a solved trainer.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

import yaml

from . import utils


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tokenizer_kwarg(trainer_cls) -> str:
    """TRL renamed DPOTrainer's `tokenizer` arg to `processing_class`
    across the 0.12-0.14 releases. Pass whichever this install accepts."""
    params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in params:
        return "processing_class"
    return "tokenizer"


def _filter_kwargs_for(cls, kwargs: dict, critical: set) -> dict:
    """Drop kwargs the installed TRL version no longer accepts.

    TRL removes/renames DPOConfig args across releases (e.g.
    max_prompt_length is gone in trl 1.14). Passing only accepted kwargs
    keeps this working across versions. Dropping a CRITICAL kwarg is a
    loud TypeError, never a silent default, because a missing beta or
    learning rate would silently change the experiment.
    """
    params = inspect.signature(cls.__init__).parameters
    takes_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    accepted = {p for p in params if p != "self"}
    missing_critical = [
        k for k in critical if k not in accepted and not takes_kwargs
    ]
    if missing_critical:
        raise TypeError(
            f"{cls.__name__} does not accept critical kwargs "
            f"{missing_critical}; TRL version drift needs a code change."
        )
    dropped = [k for k in kwargs if k not in accepted and not takes_kwargs]
    for k in dropped:
        print(
            f"warning: {cls.__name__} does not accept {k!r}; dropping it "
            f"(installed TRL API drift)"
        )
    return {k: v for k, v in kwargs.items() if k in accepted or takes_kwargs}


def build_dpo_config(config: dict, out_dir: Path):
    """Map our YAML onto trl.DPOConfig. GPU import, called from train()."""
    from trl import DPOConfig

    dpo = config["dpo"]
    kwargs = dict(
        output_dir=str(out_dir),
        beta=float(dpo["beta"]),
        loss_type=str(dpo.get("loss_type", "sigmoid")),
        max_length=int(config["data"]["max_seq_length"]),
        max_prompt_length=int(config["data"]["max_seq_length"]) // 2,
        per_device_train_batch_size=int(dpo["per_device_batch"]),
        gradient_accumulation_steps=int(dpo["gradient_accumulation_steps"]),
        num_train_epochs=int(dpo["num_train_epochs"]),
        learning_rate=float(dpo["learning_rate"]),
        lr_scheduler_type=str(dpo.get("lr_scheduler", "cosine")),
        # transformers 5.x removed warmup_ratio; warmup_steps accepts a
        # float < 1 as a ratio of total steps, which is exactly the old
        # warmup_ratio semantics. The config key keeps its old name.
        warmup_steps=float(dpo.get("warmup_ratio", 0.1)),
        gradient_checkpointing=bool(dpo.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=float(dpo.get("max_grad_norm", 1.0)),
        logging_steps=int(dpo.get("logging_steps", 10)),
        save_steps=int(dpo.get("save_steps", 200)),
        save_total_limit=2,
        eval_strategy=str(dpo.get("eval_strategy", "no")),
        seed=int(config.get("seed", 42)),
        bf16=True,
        optim="paged_adamw_32bit",  # the standard QLoRA optimizer
        remove_unused_columns=False,
        report_to="none",  # local logs only; no wandb account needed
    )
    # These define the experiment. If the installed TRL stops accepting any
    # of them, fail loudly instead of training with wrong defaults.
    critical = {
        "beta", "loss_type", "max_length",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "num_train_epochs", "learning_rate", "lr_scheduler_type",
        "warmup_steps", "max_grad_norm", "seed", "bf16", "optim",
    }
    return DPOConfig(**_filter_kwargs_for(DPOConfig, kwargs, critical))


def train(config: dict, config_path: str | None = None,
          out_dir_override: str | None = None) -> Path:
    """Full DPO run. Returns the run output dir. GPU box only."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import DPOTrainer

    seed = int(config.get("seed", 42))
    utils.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_cfg = config["model"]
    hf_kwargs = utils.hf_model_kwargs(model_cfg)
    model_name = hf_kwargs["name"]
    model_revision = hf_kwargs["revision"]  # pinned hub revision, or None
    trust_remote_code = hf_kwargs["trust_remote_code"]
    out_dir = Path(out_dir_override or config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pin the exact config that produced this run.
    cfg_path = out_dir / "config.yaml"
    if config_path:
        cfg_hash = utils.sha256_of_file(config_path)
        shutil.copy(config_path, cfg_path)
    else:
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
        cfg_hash = utils.sha256_of_file(cfg_path)
    (out_dir / "config_hash.txt").write_text(cfg_hash + "\n")

    tok = AutoTokenizer.from_pretrained(model_name,
                                        revision=model_revision,
                                        trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_cfg = config.get("quantization", {})
    if quant_cfg.get("load_in_4bit", False):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=bool(
                quant_cfg.get("bnb_4bit_use_double_quant", True)),
        )
    else:
        bnb = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    model.config.use_cache = False

    if config.get("full_finetune", False):
        peft_model = model
        print("full fine-tune: all parameters train, "
              "reference model frozen by TRL")
    else:
        lora_cfg = config["lora"]
        model = prepare_model_for_kbit_training(model)
        peft_cfg = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            task_type="CAUSAL_LM",
            bias="none",
        )
        peft_model = get_peft_model(model, peft_cfg)
        peft_model.print_trainable_parameters()

    splits_dir = Path(config["data"].get("splits_dir", "results/splits"))
    train_ds = load_dataset("json",
                            data_files=str(splits_dir / "train.jsonl"))["train"]
    print(f"train pairs: {len(train_ds)}")

    dpo_args = build_dpo_config(config, out_dir)
    trainer_kwargs = {
        "model": peft_model,
        "ref_model": None,  # TRL builds a frozen copy of the base weights
        "args": dpo_args,
        "train_dataset": train_ds,
        _tokenizer_kwarg(DPOTrainer): tok,
    }
    trainer = DPOTrainer(**trainer_kwargs)

    # Live log stream: TRL only exposes log_history after train(), but the
    # in-pod watchdog (infra/monitor.py) polls training_log.jsonl every
    # 60s to kill the run on NaN loss. Without this callback the file did
    # not exist until training finished and the watchdog was blind
    # (run 4, 2026-09-28: status.json stuck at "waiting_for_log").
    log_path = out_dir / "training_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    def _append_log(entry: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    try:
        from transformers import TrainerCallback

        class _LiveLogCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    _append_log({"step": state.global_step, **logs})

        trainer.add_callback(_LiveLogCallback())
        _live_log_ok = True
    except ImportError:
        _live_log_ok = False

    print("starting DPO training. Watch the first 50 steps: NaN loss or a "
          "flat margin means stop and debug, not 'let it run overnight'.")
    trainer.train()

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # Training diagnostics: TRL logs loss, rewards/chosen, rewards/rejected,
    # rewards/margins, rewards/kl into state.log_history. With the live
    # callback above the file is already complete; without it (old
    # transformers), write the whole history here as a fallback so the
    # file always exists with the same content either way.
    if not _live_log_ok:
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in trainer.state.log_history:
                f.write(json.dumps(entry, default=str) + "\n")

    metrics = {
        "config_hash": cfg_hash,
        "model": model_name,
        "beta": float(config["dpo"]["beta"]),
        "train_pairs": len(train_ds),
        "global_step": trainer.state.global_step,
    }
    utils.save_json(metrics, out_dir / "metrics.json")
    print(f"done. adapter + logs in {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DPO fine-tune from a YAML config.")
    parser.add_argument("--config", required=True,
                        help="Path to a YAML config in configs/.")
    parser.add_argument("--set", action="append", default=[],
                        help="Config override, e.g. --set dpo.beta=0.2. "
                             "Repeatable.")
    parser.add_argument("--output-dir", default=None,
                        help="Override config['output']['dir'].")
    args = parser.parse_args()
    config = load_config(args.config)
    utils.validate_config(config)
    utils.apply_overrides(config, args.set)
    train(config, config_path=args.config,
          out_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
