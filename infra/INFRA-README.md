# dpo-lab infra (RunPod launch package)

Launch-only files. The hardened code tree lives at `~/workspace/dpo-lab/`;
nothing here runs until the coordinator says the code is ready.

## Files

- `launch_pod.py` - builds the pod spec and creates the RTX 4090 pod via the
  RunPod REST API. Packs the code tree as a base64 tarball (excludes `infra/`,
  `results/`, caches). Saves `pod_id.txt` and `serve_token.txt` (mode 600).
- `bootstrap.sh` - the pod's start command. Unpacks code, builds a venv with
  `--system-site-packages` (inherits the image's torch), `pip install -e
  ".[eval,test]"`, then runs: smoke test (00) -> HF gated-access check ->
  data prep (01) -> training (02, beta 0.1) under the monitor.py watchdog ->
  evals (03, judge step auto-skips: no OpenRouter key on pod) -> packages
  `results.tar.gz` -> serves `/workspace/serve/` on :8080 with token auth and
  stays alive for the VM to fetch.
- `monitor.py` - training watchdog. Watches
  `results/runs/beta_0.1/training_log.jsonl`; stops training on NaN loss or on
  margin flat at zero while loss falls. Writes `/workspace/serve/status.json`
  every 60 s for the VM to poll.
- `serve.py` - token-auth static server for the results handoff.
- `pod_status.py` - VM-side poller: prints the pod's `status.json` (or any
  path, e.g. `bootstrap.log`) through RunPod's HTTPS proxy URL.

## Pod env vars (set by launch_pod.py, never baked into the tarball)

- `HF_TOKEN` - read from `~/.config/llm-secrets.env` on the VM. Needed for the
  gated Qwen2.5-1.5B-Instruct weights/tokenizer.
- `SERVE_TOKEN` - freshly generated per run; guards the :8080 results server.
- `CODE_GZ_B64`, `BOOTSTRAP_B64`, `MONITOR_B64`, `SERVE_B64` - the payload.

## Fetch-then-terminate flow

1. `python3 ~/workspace/dpo-lab/infra/launch_pod.py` (only on coordinator go).
2. Watch: `python3 ~/workspace/dpo-lab/infra/pod_status.py`
   (or `pod_status.py bootstrap.log` to tail the log).
3. When `status.json` reports `done` (or `failed`), fetch:
   `https://<pod-id>-8080.proxy.runpod.net/results.tar.gz?token=<serve_token>`
4. Unpack into `~/workspace/dpo-lab/` on the VM, run the judge locally
   with `scripts/judge_on_vm.py` (VM only; auth via the OpenRouter
   credential surrogate, never an OpenRouter key on the pod).
   Write labels to `results/runs/beta_0.1/evals/judge_labels.json`.
5. Terminate the pod via the RunPod API. Billing is per-second; do not leave
   it running.

## Cost

RTX 4090 secure cloud, on-demand: **$0.74/hr** (verified Sept 2026 against
runpod.io/pricing and independent catalogs). ~10 h run -> **~$7.40**,
far under the $40 budget. 8-hour in-pod training cap is enforced by
bootstrap.sh (`timeout 28800`).

## VM quirk

Python HTTP clients get Cloudflare 1010 (banned TLS fingerprint) on
api.runpod.io / rest.runpod.io. `launch_pod.py` therefore makes
all API calls through `curl` (whose fingerprint passes), with the API key
passed via stdin config - never on a command line, never logged. `pod_status.py`
uses the same curl-via-pipe trick for the `*.proxy.runpod.net` URLs, since
urllib is banned there too.
