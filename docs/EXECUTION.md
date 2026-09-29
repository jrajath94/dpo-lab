# dpo-lab: Execution Runbook

Each phase ends with a gate. A closed gate means the phase is done.
The owner decides on aborts. The runbook decides the stop conditions.

## Phase 0: environment

- Spin up the RunPod RTX 4090 pod (image and install steps per the
  TRD Environment section). Deps install from pyproject.toml ranges;
  the resolved set is frozen to requirements.lock after install.
- Run `scripts/00_smoke_test.sh`: package imports, the full unit
  suite, and config/run-path checks. CPU-only, no network, no GPU.
  (CUDA and bf16 are confirmed at install time when bootstrap prints
  the torch/CUDA status, and by the nvidia-smi line in 02_train.sh.)
- Set HF_TOKEN on the pod via environment only. Bootstrap runs the
  gated-model access check (HfApi model_info on
  Qwen/Qwen2.5-1.5B-Instruct plus whoami) before anything else. No
  key in any file.
- Gate: smoke test green, gated-access check passed, and every TRD
  pre-flight check green before the long-pole steps start.

## Phase 1: TDD hardening (done on the VM)

- Tests for split construction, seed determinism, no prompt overlap,
  and config loading run green before any GPU time is spent.
- `tests/test_splits.py` covers the data contract. If a test fails
  here, nothing launches.
- Gate: full test suite green on the VM.

## Phase 2: data prep (pod)

- Run `scripts/01_prepare_data.sh`. Download, extract_prompt,
  seed-42 shuffle, filter to max_seq_length 1024, split into 10k
  train / 1k margin-eval / 200 generation prompts.
- Persist JSONL splits plus manifest in `results/splits`.
- Spot-check random pairs by hand for the both-bad problem and note
  the length skew in RUN_NOTES.md.
- Gate: manifest counts exact (10k train / 1k margin-eval / 200
  generation), seed recorded, hand spot-check written up.

## Phase 3: training (pod, with watchdog)

- Run `scripts/02_train.sh` with `configs/dpo_qwen25_15b_qlora.yaml`.
- One epoch, beta 0.1, everything per the TRD.
- Watchdog (infra/monitor.py) polls training_log.jsonl every 60 s
  and enforces two automatic stop conditions:
  - NaN loss: the training process is killed at once.
  - Flat margin while loss falls: the last 20 logged margins all sit
    within 0.02 of zero while the loss has dropped more than 0.10.
    That means the model is gaming the loss, not learning
    preferences. Kill the run and investigate; do not let it run to
    epoch end.
- Operator checks, by hand in the log and the plots (kl.png,
  grad_norm.png): mean KL to the reference climbing fast means the
  policy is running away; gradient norms exploding or collapsing to
  zero means stop. Neither is automated. A runaway on either is an
  abort the owner calls.
- If any condition fires, the run is marked aborted in RUN_NOTES.md
  with the trigger and the step. Aborted runs are kept, not deleted.
  They are data.
- Gate: training completes one epoch with no watchdog trigger, and
  the adapter checkpoint is saved.

## Phase 4: evals (pod)

- Run `scripts/03_evaluate.sh results/runs/beta_0.1` (splits dir
  defaults to results/splits).
- Held-out margin accuracy before and after, margin histogram plot.
- Generate 200 prompts with base and tuned models. Ship the
  generations to the VM for judging. The pod does not judge: leave
  OPENROUTER_API_KEY unset on the pod so the script's judge step
  skips itself there.
- MMLU 200-question 5-shot before and after.
- Training diagnostics plots: loss, chosen/rejected logprobs, margin
  histogram, mean KL, gradient norms.
- Gate: every eval produced output files and none of them errored
  silently. Missing files are a failed gate, not an empty cell.

## Phase 5: judge (VM)

- Runs on the owner's VM for credential hygiene:
  `python3 scripts/judge_on_vm.py --base-gens results/runs/beta_0.1/evals/gens_base.jsonl --dpo-gens results/runs/beta_0.1/evals/gens_dpo.jsonl --config configs/dpo_qwen25_15b_qlora.yaml --out results/runs/beta_0.1/evals/judge_labels.json`
- OpenRouter is called through the credential surrogate. No raw key
  leaves the VM.
- Order-swapped A/B pairs, judge model committed from the config
  (`openai/gpt-4o-mini` unless JUDGE_MODEL was overridden at judge
  time; record whichever was used).
- Save every raw label to `results/runs/beta_0.1/evals/judge_labels.json`:
  two rows per prompt (both orders, 400 labels for 200 prompts).
- Compute raw and length-controlled win rates with 95% Wilson
  intervals at assembly time, from judge_labels.json via
  dpo_weekend.utils.wilson_interval. The judge step saves labels and
  rates; the intervals are computed downstream.
- Gate: `judge_labels.json` complete (two rows per prompt, order and
  verdict on every row) and the intervals computed.

## Phase 6: results assembly and QA gates (VM)

- Assemble `results/runs/beta_0.1`: config.yaml and config_hash.txt,
  the splits manifest reference (`results/splits/manifest.json`),
  training_log.jsonl, metrics.json, checkpoint info
  (adapter_config.json plus the adapter weights), all four eval
  outputs in evals/, all eight plots in plots/,
  `evals/judge_labels.json` with Wilson intervals computed, and
  RUN_NOTES.md (written by hand at assembly; no script generates
  it).
- QA gates, all mandatory:
  - All tests green.
  - Every plot exists and renders.
  - `judge_labels.json` complete.
  - RUN_NOTES.md written honestly: what was decided, what the
    watchdog saw, what the evals say, what failed or looked weird.
  - No key, token, or credential string in any file. Grep before
    publish.
- Gate: every QA item checked off.

## Phase 7: publish

- Package the run directory for release.
- Owner decides the venue and the timing. Nothing is published until
  the owner says so.
- Gate: owner sign-off.

## Rollback and abort criteria

The per-phase rollback table (which artifact to resume from) lives in
the TRD. These are the stop rules:

- **Abort training** on a watchdog trigger (NaN loss, flat margin
  while the loss falls) or on an operator-called abort (KL runaway,
  gradient norm blowout). Owner decides retry vs write-up. Deleting
  the run is not an option.
- **Abort the judge step** if the judge model or template changes
  mid-run, or if any credential leaks into logs or artifacts. Resume
  the judge from the saved generations; already-labeled prompt_ids
  are skipped.
- **Abort publish** if any QA gate is red. A red gate blocks publish
  unconditionally.
- **Who decides what:** the runbook decides stop conditions. The owner
  decides retries, config changes, and publish. No one else.
