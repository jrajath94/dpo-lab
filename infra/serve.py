#!/usr/bin/env python3
"""Minimal token-authenticated static file server for the results handoff.

Serves /workspace/serve/ on port 8080. Every request must carry
?token=<SERVE_TOKEN> or it gets a 403. RunPod maps the pod's 8080/http port
to a public https://<pod-id>-8080.proxy.runpod.net URL; the VM fetches the
results tarball through that URL and then terminates the pod.
"""
import http.server
import os
import urllib.parse

TOKEN = os.environ["SERVE_TOKEN"]
SERVE_DIR = "/workspace/serve"
PORT = 8080


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        token = urllib.parse.parse_qs(query).get("token", [""])[0]
        if token != TOKEN:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"forbidden")
            return
        return super().do_GET()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    os.chdir(SERVE_DIR)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"serving {SERVE_DIR} on port {PORT} (token auth)", flush=True)
    server.serve_forever()
