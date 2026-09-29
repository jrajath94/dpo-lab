# dpo-lab: Technical Requirements Document

## Architecture overview

Two machines. The GPU pod (RunPod, RTX 4090, 24GB) does data prep,
training, margin eval, generation, the MMLU guard, plots, and packaging.
The owner's VM runs only the judge step. The pod never sees an OpenRouter
key. This split exists for credential hygiene and is the one known
deviation from a single-machine design.

Key rules for the split:

- The pod reads HF_TOKEN from the environment only. No raw keys in
  code, logs, files, or the repo.
- The judge runs on the owner's VM, not the pod. The pod produces
  generations and ships them; the VM calls OpenRouter through the
  credential surrogate. Raw OpenRouter keys never leave the VM.
- If a key ever appears in a log or artifact, the run is aborted and
  the artifact is scrubbed before anything is published.

## Data pipeline

Input: `HuggingFaceH4/ultrafeedback_binarized`, the community-standard
binarized preference dataset. The pipeline uses the preference split
(train_prefs, 61,966 pairs) when the mirror has one, else the train
split. Chosen because comparable public DPO runs use it, so results
are comparable. No custom-dataset trust issues.

Known warts, stated up front: the chosen answer skews longer (length
bias), some pairs have two bad answers, and there may be prompt overlap
across splits. The pipeline tests for overlap rather than assuming it
away.

Steps:

1. Download the dataset from Hugging Face.
2. Normalize with TRL's `extract_prompt`. The raw dataset carries the
   prompt implicitly in the chosen/rejected columns; extract_prompt
   produces the `prompt`, `chosen`, `rejected` columns DPOTrainer
   expects, as conversational message lists.
3. Drop malformed pairs, identical chosen/rejected pairs, and
   near-duplicate prompts (whitespace/case normalized). The count of
   each is recorded in data_stats.json.
4. Filter: cap at max_seq_length 1024.
5. Seed-42 shuffle, then split: 10,000 train pairs, 1,000 held-out
   margin-eval pairs, 200 generation prompts. The three splits are
   disjoint by construction, and the test suite asserts no prompt
   overlap between them.
6. Persist every split as JSONL plus manifest.json (per-split counts,
   seed, source dataset, split used, max_seq_length, model) and
   data_stats.json (filter counts and the chosen/rejected mean
   lengths). Splits live in `results/splits`.

A spot check of random pairs by hand is part of the pipeline. You
should know what you are teaching the model.

## Training

Model selection, surveyed 2026-09-28. The newest Qwen text releases are
Qwen3.8 (August 2026) and Qwen3.5 (March 2026). Neither is a drop-in
target for this run. Qwen3.8-Flash-Next ships under a non-Apache
"other" license with an experimental model architecture. The small
Qwen3.5 dense models (2B, 0.8B) are multimodal vision-language models
with a hybrid linear-attention architecture and thinking modes, which
would need chat-template and eval-harness changes plus an untested DPO
recipe. Qwen3-1.7B (April 2025, Apache 2.0) is also a hybrid thinking
model. The newest pure-text small instruct model remains
Qwen2.5-1.5B-Instruct, so the plan stands. The fallback (gated access
failure) did not trigger: access was verified live on 2026-09-28
(HTTP 200 on the weight files with HF_TOKEN).

Model: `Qwen/Qwen2.5-1.5B-Instruct`, Apache 2.0. The instruct variant,
not the base. Intended revision:
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Reason: at this scale, DPO on a raw base model mostly teaches
formatting. The instruct variant keeps the preference signal about
response quality. Deliberate scope choice.

Revision gap, stated plainly: nothing in the config or the code pins
this revision today. train.py and evaluate.py call from_pretrained
without a revision argument, so a future weight update on the hub
would silently change the base model. The fix is to add
model.revision to the config and pass it through in both files.
Until then, the resolved revision must be recorded in the run notes
from the run itself.

Quantization: QLoRA. 4-bit base weights (NF4, bf16 compute, double
quant), trainable LoRA adapters with r=64, alpha=16, dropout 0.05 on
q/k/v/o/gate/up/down projections. The reference model is a frozen copy
of the base weights built by TRL (ref_model=None in train.py); it is
never updated.

Hyperparameters (locked in `configs/dpo_qwen25_15b_qlora.yaml`):

- beta: 0.1 (sweep over 0.05 / 0.2 deferred to a second run)
- lr: 1e-4 on LoRA params only
- epochs: 1
- batch: 2 per device x 16 grad accumulation = 32 effective
- scheduler: cosine, warmup ratio 0.1
- max grad norm: 1.0
- seed: 42

Wiring: TRL DPOTrainer, sigmoid loss type, eval callbacks off (evals
are separate scripts, not trainer hooks).

The implicit-reward math, in five lines:

    r(x, y) = beta * log( pi(y|x) / pi_ref(y|x) )
    margin  = r(x, y_w) - r(x, y_l)
    P(prefer w) = sigmoid(margin)
    loss = -log sigmoid( beta * (logratio_w - logratio_l) )
    accuracy = fraction of held-out pairs with margin > 0

The loss only moves the margin. It cannot see answer quality directly.
That is why the generation eval exists.

## Eval pipeline

Four evals, all able to fail.

1. **Held-out margin accuracy** (1,000 pairs, before and after).
   Implicit rewards under the tuned policy versus the frozen reference.
   Report accuracy and the margin histogram shift. Note the two
   accuracies are comparable but the two margin scales are not: the
   base number is a raw logprob difference, the DPO number is a
   beta-scaled implicit margin. That is why the plots draw them on
   separate axes.

2. **Blinded A/B generation win rate** (200 prompts). Generate with the
   base and tuned models at temperature 0.7, top_p 0.9, max 512 new
   tokens. Protocol:
   - The judge sees A/B pairs with the order swapped per prompt, so
     position bias averages out.
   - The judge is an external model called through OpenRouter. The
     judge model is committed in the config (`openai/gpt-4o-mini`,
     overridable via JUDGE_MODEL at judge time). A judge change
     invalidates old numbers.
   - Raw labels are saved to `evals/judge_labels.json`: two rows per
     prompt (one per order, 400 comparisons for 200 prompts), each row
     with the order presented and the verdict.
   - Report raw win rate and length-controlled win rate. The 95%
     Wilson confidence intervals are computed at results-assembly
     time from judge_labels.json with
     dpo_weekend.utils.wilson_interval, not inside the judge step.

3. **MMLU 200-question 5-shot regression guard** (before and after).
   Collapse check only. Report both accuracies. A collapse kills the
   "nothing broke" claim.

4. **Training diagnostics.** Loss curve, chosen/rejected logprobs,
   margin histogram, mean KL to the reference, gradient norms. These
   are plots and tables in `results/runs/beta_0.1`, not vibes.

## Reproducibility

- Seed 42 everywhere.
- Config hash pinning: the run directory records the exact config file
  used, and results reference that hash.
- All splits persisted as JSONL with a manifest. Any step can be
  re-run from its inputs.
- Dependencies come from pyproject.toml as EXACT pins (not ranges):
  transformers==5.17.0, datasets==5.0.1, trl==1.14.0, peft==0.21.0,
  accelerate==1.15.0, bitsandbytes==0.50.2, lm-eval==0.4.13, plus
  torch>=2.4 (torch 2.8.0 ships in the image and is inherited via
  --system-site-packages). The ranges were retired 2026-09-28 after
  silent TRL drift (1.14.0 removed DPOConfig's max_prompt_length)
  killed a run; the pins match the set proven on the pod. The pod
  still freezes the resolved set with `pip freeze > requirements.lock`
  after install, and the lock ships inside results.tar.gz as a
  second record. See the Environment section for the image and the
  install strategy.

## Environment

The exact pod, from infra/launch_pod.py and infra/bootstrap.sh:

- Image: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- GPU: NVIDIA GeForce RTX 4090, 24GB, RunPod secure cloud, on-demand
  (~$0.74/hr; a ~10h run lands near $7.40)
- Python 3.11 from the image. The image already ships torch and the
  CUDA stack system-wide.
- venv strategy: `python3 -m venv --system-site-packages .venv`. The
  venv inherits the image's torch so pip never re-downloads ~4GB of
  torch and nvidia wheels. Then `pip install -e ".[eval,test]"`.
- Dependency pins, from pyproject.toml: torch>=2.4 (image torch 2.8.0
  inherited via --system-site-packages), transformers==5.17.0,
  datasets==5.0.1, trl==1.14.0, peft==0.21.0, accelerate==1.15.0,
  bitsandbytes==0.50.2, pyyaml>=6.0, numpy>=1.26, requests>=2.31,
  matplotlib>=3.8, plus lm-eval==0.4.13 and pytest>=8.0 from the
  extras. The GPU stack is pinned exactly (ranges retired 2026-09-28
  after TRL drift killed a run). The post-install `pip freeze` still
  ships inside results.tar.gz as a second record.
- Pod env vars, set by launch_pod.py and never baked into the code
  tarball: HF_TOKEN (gated Qwen weights and tokenizer), SERVE_TOKEN
  (fresh per run, guards the :8080 results server).

## Commands per phase

All commands run from the repo root (`~/workspace/dpo-lab` on the VM,
`/workspace/dpo-lab` on the pod). Every script cds to its own parent
first, so the working directory is always the repo root.

- Smoke test (CPU-only, runs anywhere):
  `bash scripts/00_smoke_test.sh`
  No args. Runs the package import, the full pytest suite, and the
  config/run-path checks.

- Data prep (CPU-only, needs HF_TOKEN for the gated tokenizer):
  `bash scripts/01_prepare_data.sh [config]`
  Config defaults to `configs/dpo_qwen25_15b_qlora.yaml`. Override the
  output dir with OUT_DIR (default `results/splits`).

- Training (GPU pod only, needs HF_TOKEN and `results/splits`):
  `bash scripts/02_train.sh [config]`
  Optional env overrides: BETA (e.g. `BETA=0.05` rewrites dpo.beta),
  OUT_DIR (e.g. `OUT_DIR=results/runs/beta_0.05`). Exits at once if
  HF_TOKEN is unset or `train.jsonl` is missing.

- Evals (GPU pod; the judge step auto-skips with no key):
  `bash scripts/03_evaluate.sh <run-dir> [splits-dir]`
  Example: `bash scripts/03_evaluate.sh results/runs/beta_0.1`.
  Runs margin (base, then dpo), generation (base, then dpo), the judge
  step only if OPENROUTER_API_KEY is set (never on the pod), the MMLU
  guard (base, then dpo), then plots. Eval outputs land in
  `<run-dir>/evals/`, plots in `<run-dir>/plots/`.

- Judge (owner's VM only):
  `python3 scripts/judge_on_vm.py --base-gens results/runs/beta_0.1/evals/gens_base.jsonl --dpo-gens results/runs/beta_0.1/evals/gens_dpo.jsonl --config configs/dpo_qwen25_15b_qlora.yaml --out results/runs/beta_0.1/evals/judge_labels.json`
  Auth goes through the vault surrogate (custom.openrouter); no raw
  key is requested, printed, or stored. Resume-friendly: prompt_ids
  already labeled in an existing --out file are skipped. Optional:
  --judge-model to override, --sleep to change the 0.5s politeness
  delay between calls.

- Beta sweep (deferred second run, not this one):
  `BETAS="0.05 0.1 0.2" bash scripts/04_beta_sweep.sh [config]`
  then `python scripts/05_compare_betas.py`, which writes
  `results/beta_comparison.md`.

- Pod launch and watch (VM):
  `python3 infra/launch_pod.py` (only on coordinator go),
  `python3 infra/pod_status.py [path]` to poll status.json or tail
  bootstrap.log through the RunPod proxy.
- Fetch results (VM, when status.json says done):
  `https://<pod-id>-8080.proxy.runpod.net/results.tar.gz?token=<serve_token>`
  Unpack into `~/workspace/dpo-lab/`, then terminate the pod.

## Pre-flight checks

No long-pole step launches on a red check.

1. Environment matches this TRD: the image above, one RTX 4090
   visible. Bootstrap prints the GPU name and the torch/CUDA status
   at install time; confirm both before data prep runs.
2. Credentials attached and valid. HF_TOKEN is set as a pod env var
   and passes the gated-access check from bootstrap.sh:
   `HfApi(HF_TOKEN).model_info("Qwen/Qwen2.5-1.5B-Instruct")` must
   return, and `whoami()` must name an account that accepted the
   Qwen license. No OpenRouter key exists on the pod; launch_pod.py
   only sets HF_TOKEN and SERVE_TOKEN.
3. Judge auth ready on the VM. The credential surrogate
   (custom.openrouter, via
   `/opt/hatch/skills/skill-creator/bin/dynamic_credentials.py`)
   must be importable before the judge step runs. The pod is never
   part of this path.
4. Gated model and dataset reachable. The gated-access check covers
   the model. The dataset (`HuggingFaceH4/ultrafeedback_binarized`)
   is public; data prep is the reachability proof, and it fails fast
   before any training spend if the download breaks.
5. Dry run green. `scripts/00_smoke_test.sh` is the dry run: the full
   unit suite (split construction, seed determinism, no prompt
   overlap, config validation) on CPU, no network, no GPU. It must
   pass before the pod launches. Training has no small-scale trial
   mode, so the first 50 steps of the real run are watched by hand
   instead (02_train.sh prints the check).

## Failure modes and fixes

- NaN loss. monitor.py watches `training_log.jsonl` every 60 seconds.
  Any NaN loss kills the training process at once and writes
  status.json with reason `nan_loss`. Fix: abort the run, keep the
  run directory (aborted runs are data), inspect the config and the
  data, then relaunch into a fresh run dir. The owner decides retry
  vs write-up.
- Flat margin while loss falls. If the last 20 logged margins all sit
  within 0.02 of zero while the loss has dropped more than 0.10,
  monitor.py kills training (reason
  `flat_margin_while_loss_falls`). The model is gaming the loss, not
  learning preferences. Fix: abort and investigate; do not let it run
  to epoch end.
- CUDA OOM. Nothing catches this automatically; the process dies and
  the log ends mid-step. Fix: confirm no other process holds the GPU
  (nvidia-smi), then relaunch with a smaller per-device batch
  (`--set dpo.per_device_batch=1`) and a larger
  `--set dpo.gradient_accumulation_steps=32` to keep the effective
  batch at 32. Gradient checkpointing is already on.
- HF 401 or gated denial. The bootstrap gated-access check fails and
  the pipeline stops before any GPU spend. Fix: accept the Qwen
  license at huggingface.co/Qwen/Qwen2.5-1.5B-Instruct with the
  account that owns HF_TOKEN, then relaunch.
- Dataset download failure. `01_prepare_data.sh` exits nonzero
  (`set -euo pipefail`) and bootstrap stops before training. Fix:
  check network and HF status, then rerun 01; splits rebuild
  deterministically from seed 42.
- Judge parse-error storms. A judge reply that does not parse becomes
  a TIE with `parse_error=true`, counted in `n_parse_errors`. A few
  are normal; a storm means the judge model or template drifted. Fix:
  inspect the reasons in judge_labels.json. judge_on_vm.py refuses to
  mix labels judged under a different model or template (loud error);
  either match the original or delete the file and judge all prompts
  under the new protocol. Resume is safe: already-labeled prompt_ids
  are skipped.
- Packaging an empty results dir. bootstrap.sh refuses to declare
  done unless `results/runs/beta_0.1` holds both adapter_config.json
  and training_log.jsonl; otherwise status.json says failed with
  reason `packaging_refused_empty_results`. Fix: read bootstrap.log,
  repair the failed phase, relaunch from the last good artifact.

## Verification criteria per phase

Each phase is done only when its artifacts exist and pass the check.

- Data prep. Artifacts: `results/splits/train.jsonl`,
  `margin_eval.jsonl`, `generation.jsonl`, `manifest.json`,
  `data_stats.json`. Check: manifest counts read exactly 10000,
  1000, 200; manifest seed is 42; data_stats.json records the raw,
  empty, identical, dupe-prompt, and overlong counts plus the
  chosen/rejected mean lengths (watch the length bias).
- Training. Artifacts: `results/runs/beta_0.1/adapter_config.json`,
  `config.yaml`, `config_hash.txt`, `training_log.jsonl`,
  `metrics.json`. Check: every row in training_log.jsonl has a finite
  loss (02_train.sh prints the first/last loss check),
  config_hash.txt matches the sha256 of config.yaml, and metrics.json
  records the beta and the train pair count.
- Evals. Artifacts: `evals/margin_base.json`,
  `evals/margin_dpo.json`, `evals/gens_base.jsonl`,
  `evals/gens_dpo.jsonl`, `evals/mmlu_base.json`,
  `evals/mmlu_dpo.json`, and `plots/` with loss.png,
  rewards_chosen.png, rewards_rejected.png, margins_train.png,
  kl.png, grad_norm.png, rewards_chosen_vs_rejected.png, and
  margin_hist_base_vs_dpo.png. Check: each JSON parses, both
  generation files hold 200 rows, and every plot file exists and
  renders.
- Judge. Artifact: `evals/judge_labels.json`. Check: n equals 400
  (200 prompts times 2 orders), n_prompts equals 200, judge_model
  names the model actually used, and every label row carries order
  and verdict.
- Assembly. The run directory holds config.yaml and config_hash.txt,
  the splits manifest reference (`results/splits/manifest.json`),
  training_log.jsonl, metrics.json, checkpoint info
  (adapter_config.json plus adapter weights), all four eval outputs
  in evals/, all eight plots, evals/judge_labels.json with Wilson
  intervals computed, and RUN_NOTES.md. A credential grep over every
  file returns nothing.

## Rollback plan

Never relaunch from scratch when a good artifact exists. Work back
from the failure to the last good artifact and resume there.

- Data prep failed. Fix the cause (network, tokenizer access), delete
  `results/splits` if it is partial, rerun 01_prepare_data.sh. Splits
  are deterministic under seed 42, so a rebuild is identical.
- Training failed (watchdog, timeout, OOM). Keep the failed run
  directory; it is data. Fix the config or the environment, then
  relaunch 02_train.sh into a fresh OUT_DIR
  (e.g. `results/runs/beta_0.1_retry1`) so the failed run is never
  overwritten.
- Evals failed. The run directory already holds the adapter and the
  training log. Rerun 03_evaluate.sh against the same run dir; eval
  outputs overwrite cleanly.
- Judge failed or was interrupted. Rerun scripts/judge_on_vm.py with
  the same --out; prompt_ids already labeled are skipped, so a resume
  costs nothing.
- Judge model or template changed mid-run. Old labels are invalid.
  Either rerun with the original model/template or delete
  judge_labels.json and judge all prompts under the new protocol.
- Publish blocked. A red QA gate blocks publish unconditionally. Fix
  the gate, do not waive it.
