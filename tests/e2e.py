#!/usr/bin/env python3
"""End-to-end check against a REAL `hermes` process and a fake runaway model.

The fake OpenAI-compatible provider answers every agent request with another terminal
tool call, forever, and reports 1050 tokens per call. Without the plugin Hermes would run
to its max-turns cap. Each scenario uses a throwaway HERMES_HOME (never the user's own),
the plugin symlinked in and enabled.

    python tests/e2e.py            # needs `hermes` on PATH; prints PASS/FAIL per scenario
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USAGE = {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050}
agent_requests = []  # requests that offered tools = main agent loop calls
lock = threading.Lock()
script = {"delegate_next": False}  # scenario E: answer the next agent call with delegate_task


class FakeProvider(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json({"object": "list", "data": [{"id": "fake-runaway", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        agentic = bool(req.get("tools"))
        with lock:
            if agentic:
                agent_requests.append(time.time())
            n = len(agent_requests)
            delegate = agentic and script["delegate_next"]
            script["delegate_next"] = script["delegate_next"] and not agentic
        if agentic:
            name, args = (("delegate_task", {"tasks": [{"goal": "Keep working until done."}]}) if delegate
                          else ("terminal", {"command": f"echo step {n}"}))
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{n}_{time.time_ns()}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}
            finish = "tool_calls"
        else:  # auxiliary calls (titles etc.)
            msg, finish = {"role": "assistant", "content": "Fake title"}, "stop"
        if not req.get("stream"):
            return self._json({"id": "x", "object": "chat.completion", "created": 0, "model": "fake-runaway",
                               "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": USAGE})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        delta = {"role": "assistant"}
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            delta["tool_calls"] = [{"index": 0, "id": tc["id"], "type": "function", "function": tc["function"]}]
        else:
            delta["content"] = msg["content"]
        for chunk in ({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                      {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                      {"choices": [], "usage": USAGE}):
            chunk.update(id="x", object="chat.completion.chunk", created=0, model="fake-runaway")
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def make_home(port: int, enabled: bool = True, max_turns: int = 40) -> Path:
    home = Path(tempfile.mkdtemp(prefix="ha-e2e-"))
    (home / "plugins").mkdir()
    (home / "plugins" / "hermes-allowance").symlink_to(ROOT)
    (home / "config.yaml").write_text(f"""model:
  provider: custom
  default: fake-runaway
  base_url: http://127.0.0.1:{port}/v1
  api_key: no-key-needed
agent:
  max_turns: {max_turns}
approvals:
  mode: "off"
memory:
  memory_enabled: false
  user_profile_enabled: false
plugins:
  enabled:
    - {"hermes-allowance" if enabled else "none"}
""")
    return home


def run_hermes(home: Path, env_extra: dict, toolsets: str = "terminal") -> tuple[int, str, int]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HERMES_", "PAPERCLIP_")) and not k.endswith(("_API_KEY", "_TOKEN"))}
    env.update(HERMES_HOME=str(home), **env_extra)
    before = len(agent_requests)
    p = subprocess.run(["hermes", "chat", "-q", "Keep working until done.", "-Q", "--yolo", "-t", toolsets],
                       env=env, capture_output=True, text=True, timeout=300)
    time.sleep(2)  # let any straggling child request land before counting
    return p.returncode, p.stdout + p.stderr, len(agent_requests) - before


def main() -> int:
    if not shutil.which("hermes"):
        print("SKIP: hermes not on PATH")
        return 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeProvider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, failures = server.server_address[1], 0

    def check(name, ok, detail):
        nonlocal failures
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    # 0: control. Same runaway model, plugin NOT enabled: only Hermes' own turn cap stops it.
    rc, out, n = run_hermes(make_home(port, enabled=False, max_turns=12), {"HERMES_ALLOWANCE_MODEL_CALLS": "4"})
    check("0 control, plugin off", n >= 12, f"provider saw {n} agent calls (runs to max_turns=12)")

    # A: model-call cap, refused with a final answer
    home = make_home(port)
    rc, out, n = run_hermes(home, {"HERMES_ALLOWANCE_MODEL_CALLS": "4"})
    check("A model_calls=4", n == 4 and "[hermes-allowance] Stopped" in out and rc == 0,
          f"provider saw {n} agent calls, exit {rc}, stop message {'present' if 'hermes-allowance] Stopped' in out else 'MISSING'}")
    receipts = (home / "allowance" / "receipts.jsonl")
    check("A receipt", receipts.exists() and "model_calls 5 > limit 4" in receipts.read_text(),
          receipts.read_text().strip()[:160] if receipts.exists() else "no receipt")

    # B: tool-call cap trips the whole unit
    home = make_home(port)
    rc, out, n = run_hermes(home, {"HERMES_ALLOWANCE_TOOL_CALLS": "2"})
    check("B tool_calls=2", n == 3 and "tool_calls 3 > limit 2" in out,
          f"provider saw {n} agent calls (2 tools ran, 3rd blocked, next model call refused), exit {rc}")

    # C: one budget per Paperclip issue, shared across separate processes, priced at $0
    home = make_home(port)
    issue = {"PAPERCLIP_TASK_ID": "ISS-42", "HERMES_ALLOWANCE_TOKENS": "3000"}
    rc1, out1, n1 = run_hermes(home, issue)
    rc2, out2, n2 = run_hermes(home, issue)
    check("C paperclip issue, tokens=3000", n1 == 3 and n2 == 0 and "paperclip:ISS-42" in out2,
          f"run 1: {n1} calls (3x1050 > 3000), run 2 same issue: {n2} calls, exits {rc1}/{rc2}")
    rc3, out3, n3 = run_hermes(home, {**issue, "PAPERCLIP_TASK_ID": "ISS-43"})
    check("C other issue unaffected", n3 == 3, f"ISS-43 got its own allowance: {n3} calls")

    status = subprocess.run(["hermes", "allowance", "status"], capture_output=True, text=True,
                            env={**os.environ, "HERMES_HOME": str(home)}, timeout=120)
    check("D `hermes allowance status`", "paperclip:ISS-42" in status.stdout and "TRIPPED" in status.stdout,
          status.stdout.strip().splitlines()[-1][:150] if status.stdout.strip() else status.stderr[-200:])

    # E: a subagent spends from its parent's allowance (one budget for the whole session tree)
    home = make_home(port)
    script["delegate_next"] = True
    rc, out, n = run_hermes(home, {"HERMES_ALLOWANCE_MODEL_CALLS": "6"}, toolsets="terminal,delegation")
    status = subprocess.run(["hermes", "allowance", "status"], capture_output=True, text=True,
                            env={**os.environ, "HERMES_HOME": str(home)}, timeout=120).stdout
    units = [l for l in status.splitlines() if l.startswith("session:")]
    check("E subagent rollup, model_calls=6", n == 6 and len(units) == 1 and "subagents=1" in units[0],
          f"parent+child together made {n} calls; units: {[u[:110] for u in units]}")
    server.shutdown()
    print("ALL PASS" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
