# dpo-lab: Product Requirements Document

## Problem

The owner has production LLM and agentic AI experience but zero hands-on
post-training depth. No DPO, no RLHF, no preference data work. A DeepMind
Gemini Post-training SWE posting asks for "designing, training, and
evaluating machine learning models". That gap is real and this project
fills it with an honest artifact, not a claim.

## Who reads this and why

- The owner, to know what was built and what it can honestly claim.
- A hiring manager or interviewer, to see post-training work that is
  scoped, reproducible, and self-critical.
- The owner in six months, to remember why each decision was made.

## Goals

1. Fine-tune Qwen2.5-1.5B-Instruct with DPO on the public
   HuggingFaceH4/ultrafeedback_binarized dataset, using TRL's DPOTrainer
   with QLoRA on one 24GB GPU.
2. Measure the result with evals that are allowed to fail: held-out
   preference margin accuracy, blinded LLM-judged generation win rate,
   an MMLU regression guard, and training diagnostics.
3. Produce one honest one-line claim plus a written record of what the
   project does not prove.

## Non-goals

- Not production RLHF. No reward model is trained. No online RL. No
  human annotators.
- Not a better model. 10k pairs is tiny. Claims are about the method
  working, not about shipping a superior model.
- Not a new implementation. The learning goal is DPO mechanics and eval
  discipline, so the project uses TRL's DPOTrainer instead of a
  hand-rolled loss. That is a deliberate scope choice, stated plainly.
- Not a beta sweep in this run. Beta 0.1 ships first. The 0.05 / 0.2
  sweep is a deferred second run.

## Success criteria

Every eval has an explicit pass/fail bar. A failed eval is a result,
not a bug.

1. **Held-out margin accuracy must beat the base model.** The implicit
   reward of each answer is beta * log(pi(y|x) / pi_ref(y|x)). Accuracy
   is the fraction of the 1,000 held-out pairs where the chosen answer
   scores higher than the rejected one. Pass: post-training accuracy is
   strictly above the pre-training accuracy on the same pairs, and the
   absolute margin distribution shifts positive.
2. **Blinded A/B generation win rate with an order-swapped judge.**
   The judge compares base and tuned generations on 200 prompts, each
   prompt judged twice with the A/B order swapped (400 labels total).
   Raw labels are saved. Pass bar: raw and length-controlled win rates
   are reported with 95% Wilson confidence intervals, computed at
   results-assembly time from the saved labels. The eval passes as a
   measurement whether or not the model wins. A flat or negative win
   rate next to a rising margin accuracy is an explicitly acceptable
   outcome and must be written into RUN_NOTES.md as one.
3. **MMLU 200-question 5-shot regression guard must not collapse.**
   Pass: post-training accuracy is within the pre-training noise band
   (no collapse). This is a safety check only. A small dip is fine.
   A collapse kills the "nothing broke" claim.
4. **Training diagnostics must be legible.** Loss declines, margin
   histogram shifts right, mean KL to the reference stays bounded, and
   gradient norms stay stable. Any of these failing is an abort
   condition, not a success story.

## The honest one-line claim

"DPO training on 10k preference pairs shifted the implicit-reward margin
on held-out pairs from X to Y, with a blinded judge win rate of Z on
200 generation prompts and no MMLU collapse."

X, Y, and Z come from the run. They are not in this document.

## Explicit anti-claims

- This is not RLHF. The word RLHF must never appear as a claim about
  this work. DPO is DPO.
- 10k pairs cannot produce a meaningfully better model. Any quality
  difference the judge finds is small and reported with uncertainty.
- Results generalize to this dataset, this model, this beta. Not to
  DPO in general.
