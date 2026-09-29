# RUN_NOTES.md — dpo-lab run history

Honest log of every GPU run. All times EDT, 2026-09-28/29. Spend cap: $40.

## Run 1 (~19:00, pod n/a) — died in 6 min — $0.12
`NameError` in `data.py`: called `utils.validate_config()` without importing
`utils`. Fixed with the import, a regression test that executes `main()`, a
pod-side fail-fast gate, and a packaging gate that refuses empty results.

## Run 2 (pod `njffukia7gmpth`, 19:05–19:12) — crashed before training — ~$0.09
`trl==1.14.0` removed `max_prompt_length` from `DPOConfig`; pyproject had only
`>=` floors so the new version was picked up silently. Fixed with
`_filter_kwargs_for` in `train.py` (drops non-critical rejected kwargs with a
loud warning; raises `TypeError` on rejected critical kwargs) and exact
dependency pins. The packaging gate correctly refused the empty results.

## Run 3 (pod `dy15d0y3h17mjw`, 19:31–19:45) — stopped before training spend — ~$0.17
Pod-side smoke gate caught `warmup_ratio` also removed (transformers 5.x).
The critical-kwarg assertion fired as designed: zero training spend. Fixed by
mapping the config value to `warmup_steps` (float < 1 keeps ratio semantics;
verified identical in transformers 5.17.0 source). Every other kwarg verified
against the actual trl-1.14.0 + transformers-5.17.0 wheel sources.

## Run 4 (pod `c23vl7zyt915zy`, 19:43–20:56) — training OK, eval crashed — ~$0.89
Training completed: 308/308 steps in 55:37, train loss 0.5962, no NaN,
rewards/margins -0.0015 → 0.68, loss 0.6945 → 0.5501, rewards/acc 0.41 → 0.71.
Eval crashed at margin-eval [1/5]: transformers 5.x changed
`apply_chat_template(..., tokenize=True)` to return a `BatchEncoding` instead
of `list[int]`. `_response_span` built ids from the field names and
`sequence_logprob` crashed. The same drift silently broke `data.py`'s
length pre-filter (`len(BatchEncoding)` is 2, so every pair passed; training
was unaffected because TRL truncates at `max_length` internally). No evals
produced; 1.6GB results (adapter + logs) died with the pod.

## Run 5 (pod `16lj5rd1ehjfyh`, 21:00–22:14) — training OK, eval crashed — ~$0.90
Training completed: 308/308 steps (31 log rows), loss 0.6933 → 0.5152,
margin → 0.70, no NaN. The BatchEncoding fix worked: margin evals [1/5] and
[2/5] completed. Margin results (n=1000 held-out pairs): DPO implicit-reward
accuracy 0.674 (mean margin +0.536); base raw logprob accuracy 0.471 (mean
margin -33.4). The base number is length-confounded (median -9.3 nats, heavy
left tail: chosen responses skew longer and raw sums penalize length), so the
eval now also reports a length-normalized base twin and the assembler prefers
it. Eval crashed at generation [3/5]: transformers 5.x `model.generate()`
raises `ValueError` on the `generator` kwarg. Fixed by seeding the global
torch/CUDA RNGs per generation instead; AST regression test asserts no
`generate()` call passes `generator=`. Also fixed in this round:
`training_log.jsonl` now streams live via a `TrainerCallback` (run 4's in-pod
watchdog was blind mid-run), and `03_evaluate.sh` no longer aborts all
remaining eval steps when one fails.

## Run 6 (pod `kb0t2i610nqcps`, ~23:11–05:27) — training + eval OK, MMLU guard failed — ~$0.90
Training completed: 308/308 steps (31 log rows), loss 0.6906 → 0.5148,
margin → 0.696, no NaN. Eval, all with the run-5 fixes live:
[1/5] margin base OK (raw 0.471, len-norm 0.572 — the debias twin works);
[2/5] margin DPO OK (0.668, mean +0.534); [3/5] and [4/5] generation OK
(the `generator=` crash is fixed); [5/5] judge skipped on pod by design
(runs on VM); plots OK. [6/6] MMLU guard FAILED both base and dpo — NOT a
transformers drift: lm-eval 0.4.13 imports `TypedDict(..., extra_items=T)`
(PEP 728, needs Python >= 3.13) and the pod runs Python 3.11, so the import
itself raises TypeError. The run_step isolation worked as designed: plots
still ran, tarball packaged, status "done". MMLU recorded as a known limit
(dependency/interpreter incompatibility), not re-run: the guard is a
regression check, not the deliverable.

Post-run (VM, 2026-09-29): blinded judge complete via OpenRouter
(openai/gpt-4o-mini), 200 held-out prompts x 2 orders = 400 labels,
36 ties, 0 parse errors. Assembled + independently verified
(verify_run.py: VERIFY OK, 45,103 checks):
- Blinded win rate (strict, ties as non-wins): 0.5425, 95% Wilson CI
  [0.494, 0.591]. The CI includes 0.50, so the strict edge is not
  significant at 95%.
- Excluding ties: 0.596, 95% Wilson CI [0.545, 0.645] (n=364).
- Length bias: DPO longer in 266/400; judge preferred DPO 63.5% when DPO
  longer vs 49.3% when base longer.
- Position bias: DPO won 57.0% shown first vs 51.5% shown second;
  order-swapping cancels it in the aggregate.
- Held-out margin accuracy: base 0.572 (len-norm; raw 0.471) -> DPO 0.668.
Honest claim: "Ran DPO fine-tuning experiments on Qwen2.5-1.5B-Instruct
with HuggingFaceH4/ultrafeedback_binarized; measured 54.2% held-out win
rate over the base model (n=400, 95% Wilson CI [0.494, 0.591])."
Spend total: ~$3.10 of $40.
