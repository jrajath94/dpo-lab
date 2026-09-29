# dpo-lab: honest DPO at weekend scale

This repo fine-tunes Qwen2.5-1.5B-Instruct with Direct Preference Optimization (DPO) on public preference data, using the standard TRL implementation with QLoRA, and evaluates the result against the base model with held-out preference margins, a blinded judge generation comparison, and a benchmark regression guard. It is one small offline experiment on one GPU. It is not production post-training.

## RESULTS (run 6, beta_0.1, verified)

Run 6 finished 2026-09-29. Every number below is assembled by `scripts/assemble_results.py` from raw artifacts and re-derived independently by `scripts/verify_run.py` (VERIFY OK, 45,103 checks).

Honest claim line:

"Ran DPO fine-tuning experiments on Qwen2.5-1.5B-Instruct with HuggingFaceH4/ultrafeedback_binarized; measured 54.2% held-out win rate over the base model (n=400, 95% Wilson CI [0.494, 0.591])."

| Result | Value |
|---|---|
| Held-out preference margin accuracy, base model | 0.572 (length-normalized; raw logprob accuracy 0.471) |
| Held-out preference margin accuracy, DPO model | 0.668 |
| Blinded generation win rate (DPO over base) | 54.2% (strict: ties count as non-wins) |
| Win rate n | 400 (200 prompts x 2 orders) |
| 95% Wilson CI on win rate | [0.494, 0.591] |
| Ties in the blinded comparison | 36 |
| Win rate excluding ties | 0.596, 95% Wilson CI [0.545, 0.645] (n=364) |
| MMLU 5-shot accuracy, base model | not measured (see limits) |
| MMLU 5-shot accuracy, DPO model | not measured (see limits) |
| Mean KL from base to reference during training (final) | not logged during training |
| Beta used in the reported run | 0.1 |
| Beta sweep results (0.05 / 0.1 / 0.2 win rates) | single-beta run (0.1 only); sweep not run, noted as a limit |
| Training loss, final logged value | 0.596 (train_loss at step 313 of 313) |

Raw artifacts will live in `results/runs/<run>/evals/`. Never trust a number in this table unless you can point at the file it came from.

## Quickstart

Prereqs: Python 3.10+, a CUDA box for train/evaluate. Accept the Qwen license on HuggingFace and export `HF_TOKEN`. For the judge step, export `OPENROUTER_API_KEY`.

```bash
pip install -e .
bash scripts/00_smoke_test.sh          # CPU-only: imports, unit tests, config checks
bash scripts/01_prepare_data.sh        # CPU-only: build results/splits/ (seed 42)
bash scripts/02_train.sh               # GPU box: train beta 0.1 into results/runs/beta_0.1/
bash scripts/03_evaluate.sh results/runs/beta_0.1   # GPU box: margin, generation, judge, MMLU, plots
```

Optional beta sweep:

```bash
bash scripts/04_beta_sweep.sh          # trains and evaluates beta in {0.05, 0.1, 0.2}
python scripts/05_compare_betas.py     # compares results/runs/*/evals/
```

The judge step is skipped if `OPENROUTER_API_KEY` is not set. You can re-run just the judge later with the command printed by `03_evaluate.sh`.

## Repo map

```
dpo-lab/
  README.md                 this file
  pyproject.toml            dependencies
  configs/
    dpo_qwen25_15b_qlora.yaml   main run: 1.5B, QLoRA, single 24 GB GPU
    dpo_qwen25_3b.yaml          stretch config: 3B, needs a 40 GB card
  src/dpo_weekend/
    data.py                 load, filter, split, and persist preference data
    train.py                TRL DPOTrainer driver (config-driven)
    evaluate.py             margin eval, generation eval, judge, regression guard
    utils.py                seeding, margin math, small helpers
    plots.py                training diagnostic plots
  scripts/
    00_smoke_test.sh         CPU-only sanity checks
    01_prepare_data.sh       build the seeded splits into results/splits/
    02_train.sh              launch training
    03_evaluate.sh           run the full eval suite for one run
    04_beta_sweep.sh         beta sweep over {0.05, 0.1, 0.2}
    05_compare_betas.py      summarize the sweep
    judge_on_vm.py           judge helper
  docs/
    PRD.md                  what this project is trying to prove
    TRD.md                  design and implementation details
    EXECUTION.md            runbook for the GPU box
    design-notes.md         DPO in plain English: loss, beta, failure modes
  tests/                    pure-function tests, no GPU needed
  infra/                    RunPod launch and monitoring tooling
  results/                  artifacts land here after runs
```

## The run recipe

Base model: Qwen2.5-1.5B-Instruct, pinned to revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Apache 2.0 license. Data: `HuggingFaceH4/ultrafeedback_binarized`, 10k train pairs, 1k held-out pairs for the margin eval, 200 held-out prompts for generation, seed 42. Method: DPO via TRL `DPOTrainer` with QLoRA (4-bit NF4 base, LoRA r=64, alpha=16 on q/k/v/o/gate/up/down projections, frozen bf16 reference model). Hyperparams: beta 0.1, learning rate 1e-4, 1 epoch, effective batch 32, cosine schedule, 0.1 warmup. Compute: one RTX 4090 on RunPod.

## Evals and how to read each number

**1. Held-out preference margin accuracy.** For each of the 1k held-out pairs, compute the implicit reward margin: `beta * (log pi(chosen) - log pi_ref(chosen) - log pi(rejected) + log pi_ref(rejected))`. Accuracy is the fraction with margin > 0. We report it for the base model and for the DPO model. How to read it: if DPO accuracy is not clearly above base accuracy, the run failed. This is the first number to check. Current values: base 0.572 (length-normalized; raw 0.471), DPO 0.668.

**2. Blinded generation win rate.** Base and DPO models each generate on the 200 held-out prompts. Pairs are blinded and order-swapped to cancel position bias. The judge is `openai/gpt-4o-mini` called through OpenRouter. Raw labels are stored in `judge_labels.json` so anyone can recount them. How to read it: report the win rate with its n and a 95% Wilson interval. Treat it as noisy. The judge is one model with its own biases, so a small edge proves little. Current value: 54.2% strict (n=400, 95% Wilson CI [0.494, 0.591], 36 ties); 0.596 excluding ties (95% CI [0.545, 0.645], n=364).

**3. MMLU 5-shot regression guard.** 200 random MMLU questions, 5-shot, before and after DPO. How to read it: this checks the model did not collapse. Small moves are normal. A large drop means the run is bad. Current values: not measured. The pinned lm-eval 0.4.13 needs Python >= 3.13 (PEP 728 TypedDict `extra_items`) and the pod runs Python 3.11, so the guard could not import. This is a dependency/interpreter incompatibility, not a result. Recorded as a known limit.

**4. Training diagnostics.** Training loss, chosen and rejected logprob curves, margin distribution over training, mean KL to the reference model, gradient norms. How to read them: steadily rising margins with KL staying bounded means DPO is doing what it should. Loss collapsing to zero while KL explodes means the model found a degenerate shortcut. Saved as plots in each run's `plots/` directory.

## LIMITS

- The judge is a single LLM (`openai/gpt-4o-mini` via OpenRouter). There is no human agreement measurement. Judge labels are biased and noisy, so generation win rates are weak evidence.
- Single seed (42). Nothing about seed robustness is measured.
- QLoRA, not a full fine-tune. The base weights are frozen at 4-bit; only adapters move. Results may differ from a full fine-tune.
- 1.5B parameters. This is small-model scale. Findings may not transfer to larger models.
- 10k preference pairs. This is a small dataset for alignment work.
- 1 epoch. One pass over the data. No conclusions about multi-epoch behavior.
- Single beta (0.1). The planned 0.05/0.1/0.2 sweep was not run, so nothing is known about beta sensitivity from this repo.
- The strict win-rate 95% CI [0.494, 0.591] includes 0.50: with ties counted as non-wins, the edge over base is not significant at 95% confidence. Excluding the 36 ties, DPO won 59.6% of decisive comparisons (95% CI [0.545, 0.645]).
- Length bias: DPO's answer was longer in 266 of 400 comparisons, and the judge preferred DPO 63.5% of the time when DPO was longer vs 49.3% when base was longer. Part of the win rate may be length preference, not quality.
- Position bias: DPO won 57.0% when shown first vs 51.5% when shown second. Order-swapping cancels this in the aggregate, but the judge is position-sensitive.
- MMLU regression guard not measured (lm-eval/Python incompatibility on the pod). No benchmark evidence the model did not regress elsewhere.

## What this does NOT prove

- This is not production RLHF. Labs run online RL systems at massive scale with human annotation pipelines. This is one GPU, one offline dataset, and someone else's trainer.
- No reward model was trained. DPO skips the reward model entirely. Nothing here involves training, evaluating, or serving a reward model.
- No online RL was run. There are no rollouts, no environment, no RL loop of any kind.
- Nothing here says anything about safety. 10k pairs cannot make a model meaningfully safer or better aligned. Any claim at this scale is about the method working, not about the resulting model.

Honest interview framing: "I ran the standard DPO recipe at weekend scale on a 1.5B model: real preference data, TRL, a held-out margin eval, a blinded generation comparison with an LLM judge. The DPO model won 54.2% of blinded comparisons over the base model (n=400, 95% Wilson CI [0.494, 0.591], 36 ties; 59.6% excluding ties). The MMLU regression guard could not run on the pod (dependency incompatibility), documented as a limit. I can talk about length bias in preference data and judge position bias from having watched them happen."

## Reproducing the run

Everything is pinned. The base model is pinned to revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The dataset is `HuggingFaceH4/ultrafeedback_binarized`. Splits are seeded with seed 42 and saved to disk by `01_prepare_data.sh`, so every run starts from identical data. Hyperparams live in `configs/dpo_qwen25_15b_qlora.yaml`: beta 0.1, lr 1e-4, 1 epoch, effective batch 32, cosine schedule, warmup 0.1. Training used one RTX 4090 on RunPod.

To reproduce:

1. Start from a fresh checkout.
2. Run `bash scripts/00_smoke_test.sh` to confirm the environment is sane.
3. Run `bash scripts/01_prepare_data.sh` to build the splits.
4. Run `bash scripts/02_train.sh` on a 24 GB CUDA box.
5. Run `bash scripts/03_evaluate.sh results/runs/beta_0.1` with `OPENROUTER_API_KEY` set for the judge step.

To reproduce the beta comparison, run `bash scripts/04_beta_sweep.sh` and then `python scripts/05_compare_betas.py`.

## License

MIT. No JPMorgan code, data, or mechanisms are used anywhere in this project.
