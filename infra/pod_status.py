#!/usr/bin/env python3
"""Poll the dpo-lab pod's status.json via RunPod's HTTPS proxy URL.

VM quirk (2026-09-28): python http clients get Cloudflare 1010 (banned TLS
fingerprint) on *.proxy.runpod.net, so this fetches through curl (whose
fingerprint passes). The token-bearing URL travels via a pipe-fed curl
config, never on a command line.
"""
import os
import subprocess
import sys

INFRA = os.path.expanduser("~/workspace/dpo-lab/infra")
pod_id = open(os.path.join(INFRA, "pod_id.txt")).read().strip()
token = open(os.path.join(INFRA, "serve_token.txt")).read().strip()
path = sys.argv[1] if len(sys.argv) > 1 else "status.json"
url = f"https://{pod_id}-8080.proxy.runpod.net/{path}?token={token}"

cfg = "\n".join([
    f'url = "{url}"',
    "silent = true",
    "show-error = false",
    'write-out = "\\n%{http_code}"',
    "max-time = 30",
])
r, w = os.pipe()
os.write(w, cfg.encode())
os.close(w)
if r != 3:
    os.dup2(r, 3)
    os.close(r)
try:
    p = subprocess.run(["curl", "--config", "/dev/fd/3"], pass_fds=(3,),
                       stdin=subprocess.DEVNULL, capture_output=True,
                       timeout=60)
finally:
    os.close(3)
out = p.stdout.decode()
*body_lines, code = out.rsplit("\n", 1)
body = "\n".join(body_lines)
code = code.strip()
if code == "200":
    # Full body to stdout when --full is passed (e.g. fetching a whole
    # bootstrap.log for diagnosis); truncated preview otherwise so a poll
    # loop never floods the terminal. Run 2 post-mortem: the 2000-char
    # truncation hid the real traceback until a custom fetch was written.
    if "--full" in sys.argv:
        print(body)
    else:
        print(body[:2000])
        if len(body) > 2000:
            print(f"... [truncated {len(body) - 2000} chars; rerun with --full]")
else:
    print(f"UNREACHABLE: HTTP {code}: {body[:200]}")
