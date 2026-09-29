#!/usr/bin/env python3
"""Launch the dpo-lab RunPod pod (RTX 4090, secure cloud).

Builds the pod spec from the proven Sept-21 pattern:
  - code packed as a base64 tarball (env CODE_GZ_B64), unpacked by bootstrap.sh
  - bootstrap.sh / monitor.py / serve.py injected as base64 env vars
  - HF_TOKEN passed as a pod env var (never in the tarball, never logged)
  - fresh SERVE_TOKEN generated per run for the :8080 results server
  - pod stays alive after the pipeline so the VM can fetch results.tar.gz,
    then the VM terminates it.

Saves: infra/pod_id.txt, infra/serve_token.txt (mode 600).

VM quirk (2026-09-28): python http clients get Cloudflare 1010 (banned TLS
fingerprint) on api.runpod.io / rest.runpod.io, so all API calls go through
curl (its fingerprint passes). The key reaches curl via stdin config,
never on a command line.
"""
import base64
import io
import json
import os
import secrets
import subprocess
import tarfile

LAB = os.path.expanduser("~/workspace/dpo-lab")
INFRA = os.path.join(LAB, "infra")
SECRETS = os.path.expanduser("~/.config/llm-secrets.env")

IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"


def load_secret(name):
    with open(SECRETS) as f:
        for line in f:
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} not found in {SECRETS}")


def pack_code():
    """Tar the dpo-lab tree (top-level dir dpo-lab/), minus infra/results/caches/docs.

    infra/monitor.py and infra/serve.py are additionally shipped as
    dpo-lab/.ship/{monitor,serve}.py: the RunPod REST create-pod endpoint
    500s on request bodies much over ~100KB, so these small files ride in the
    tarball instead of as separate base64 env vars (2026-09-29).
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(LAB):
            rel = os.path.relpath(root, os.path.dirname(LAB))
            # prune directories we never ship to the pod
            dirs[:] = [d for d in dirs
                       if d not in ("infra", "results", "results-fetch",
                                    "docs", ".git", ".pytest_cache",
                                    "__pycache__", ".venv")]
            for fn in files:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(root, fn)
                tar.add(full, arcname=os.path.join(rel, fn))
        for fn in ("monitor.py", "serve.py"):
            tar.add(os.path.join(INFRA, fn),
                    arcname=os.path.join("dpo-lab", ".ship", fn))
    return base64.b64encode(buf.getvalue()).decode()


def b64_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def api(method, path, body):
    """RunPod REST call via curl; key and body via pipes, never argv or disk.

    VM quirk: python http clients get Cloudflare 1010 (banned TLS
    fingerprint) on api.runpod.io / rest.runpod.io, so all API calls go
    through curl (whose fingerprint passes). The key reaches curl through a
    pipe-fed --config file; the JSON body through a second pipe. Neither
    touches the command line, the logs, or the disk.
    """
    import threading
    cfg = "\n".join([
        f'header = "Authorization: Bearer {load_secret("RUNPOD_API_KEY")}"',
        'header = "Content-Type: application/json"',
        'data-binary = "@/dev/fd/4"',
        f'request = "{method}"',
        f'url = "https://rest.runpod.io/v1{path}"',
        "silent = true", "show-error = false",
        'write-out = "\\n%{http_code}"', "max-time = 60",
    ])
    body_bytes = json.dumps(body).encode()
    r_cfg, w_cfg = os.pipe()
    r_body, w_body = os.pipe()
    os.write(w_cfg, cfg.encode())
    os.close(w_cfg)
    # Pin the read ends to fds 3 and 4 so the child sees them at
    # /dev/fd/3 (config) and /dev/fd/4 (body) regardless of allocation order.
    for src, dst in ((r_cfg, 3), (r_body, 4)):
        if src != dst:
            os.dup2(src, dst)
            os.close(src)
    def feed():
        try:
            os.write(w_body, body_bytes)
        finally:
            os.close(w_body)
    t = threading.Thread(target=feed, daemon=True)
    t.start()
    try:
        p = subprocess.run(
            ["curl", "--config", "/dev/fd/3"],
            pass_fds=(3, 4),
            stdin=subprocess.DEVNULL, capture_output=True, timeout=90,
        )
    finally:
        os.close(3)
        os.close(4)
        t.join()
    out = p.stdout.decode()
    *body_lines, code = out.rsplit("\n", 1)
    payload = "\n".join(body_lines)
    try:
        return int(code.strip()), json.loads(payload) if payload.strip() else {}
    except (ValueError, json.JSONDecodeError):
        return code, {"raw": payload[:300]}


def main():
    hf_token = load_secret("HF_TOKEN")
    serve_token = secrets.token_urlsafe(32)

    print("packing code tree...")
    code_b64 = pack_code()
    print(f"  tarball: {len(code_b64) * 3 / 4 / 1024:.0f} KB")

    env = {
        "CODE_GZ_B64": code_b64,
        "BOOTSTRAP_B64": b64_file(os.path.join(INFRA, "bootstrap.sh")),
        "SERVE_TOKEN": serve_token,
        "HF_TOKEN": hf_token,
    }

    body = {
        "name": "dpo-lab-run",
        "imageName": IMAGE,
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
        "cloudType": "SECURE",
        "supportPublicIp": True,
        "ports": ["8080/http"],
        "volumeInGb": 50,
        "containerDiskInGb": 50,
        "dockerStartCmd": [
            "bash", "-c",
            'echo "$BOOTSTRAP_B64" | base64 -d > /workspace/bootstrap.sh'
            " && chmod +x /workspace/bootstrap.sh"
            " && exec bash /workspace/bootstrap.sh",
        ],
        "env": env,
    }

    print("creating pod...")
    code, resp = api("POST", "/pods", body)
    print("POST /pods ->", code)
    if code not in (200, 201):
        raise SystemExit(f"pod creation failed: {json.dumps(resp)[:500]}")
    pod_id = resp.get("id") or (resp.get("pod") or {}).get("id")
    if not pod_id:
        raise SystemExit(f"no pod id in response: {json.dumps(resp)[:500]}")

    with open(os.path.join(INFRA, "pod_id.txt"), "w") as f:
        f.write(pod_id + "\n")
    tok_path = os.path.join(INFRA, "serve_token.txt")
    with open(tok_path, "w") as f:
        f.write(serve_token + "\n")
    os.chmod(tok_path, 0o600)

    print(f"pod launched: {pod_id}")
    print(f"watch:  python3 {INFRA}/pod_status.py")
    print(f"proxy:  https://{pod_id}-8080.proxy.runpod.net/bootstrap.log?token=<see serve_token.txt>")
    print("when status.json says done: fetch results.tar.gz, THEN terminate the pod.")


if __name__ == "__main__":
    main()
