#!/usr/bin/env python
"""Tiny OpenAI-compatible mock server that answers eval items with their GOLD answers.
Used only to test the harness offline (scorers, raw-text tool-call fallback parser, compare, perf_bench).

  .venv-eval/bin/python eval/mock_server.py --port 18999 [--tool-mode server|xml] [--corrupt-every K]

--tool-mode xml returns tool calls as raw Qwen3.5-XML text in `content` (as a server without a
tool parser would), exercising the fallback parser. --corrupt-every K returns a wrong answer for
items whose id hash % K == 0 (to test compare.py deltas). Streaming is supported for perf_bench.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import read_jsonl  # noqa: E402
from tasks import TASKS, vqa_norm  # noqa: E402

ARGS = None
INDEX: dict[str, tuple[str, dict]] = {}


def last_user_text(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            c = m["content"]
            if isinstance(c, list):
                return "".join(p.get("text", "") for p in c if p.get("type") == "text")
            return c
    return ""


def build_index():
    for task, (build, _) in TASKS.items():
        for item in read_jsonl(HERE / "data" / f"{task}.jsonl"):
            msgs, _ = build(item, HERE / "data")
            INDEX[last_user_text(msgs)] = (task, item)


def xml_call(c):
    s = f"<tool_call>\n<function={c['name']}>\n"
    for k, v in c["arguments"].items():
        v = json.dumps(v) if isinstance(v, (dict, list)) else (str(v) if not isinstance(v, bool) else str(v))
        s += f"<parameter={k}>\n{v}\n</parameter>\n"
    return s + "</function>\n</tool_call>"


def gold_response(task, item, corrupt):
    """Return (content, tool_calls)."""
    if task == "ifeval":
        return "this is a mock response without commas.", None
    if task == "gsm8k":
        ans = str(int(item["answer"]) + 1) if corrupt else item["answer"]
        body = item["solution"].split("####")[0].strip()
        return f"{body}\nThe final answer is {ans}", None
    if task == "mmlu_pro":
        a = "ABCDEFGHIJ"[(("ABCDEFGHIJ".index(item["answer"]) + 1) % len(item["options"]))] if corrupt else item["answer"]
        return f"Let's think. The answer is ({a}).", None
    if task == "humaneval":
        sol = "    return None\n" if corrupt else item["canonical_solution"]
        return f"Here is the code:\n```python\n{item['prompt']}{sol}```", None
    if task == "vqa":
        return ("zzz" if corrupt else Counter(vqa_norm(a) for a in item["answers"]).most_common(1)[0][0]), None
    if task == "tools":
        if not item["gold_calls"] or corrupt:
            return "I can answer that directly: mock.", None
        calls = []
        for g in item["gold_calls"]:
            args = {k: v[0] for k, v in g["arguments"].items() if v and v[0] != "__optional__"}
            calls.append({"name": g["name"], "arguments": args})
        if ARGS.tool_mode == "xml":
            return "\n".join(xml_call(c) for c in calls), None
        return None, [{"id": f"call_{i}", "type": "function",
                       "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                      for i, c in enumerate(calls)]
    return "unknown", None


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        msgs = body["messages"]
        text = last_user_text(msgs)
        prompt_tokens = sum(len(json.dumps(m["content"])) for m in msgs) // 4
        if body.get("stream"):
            return self._stream(body, prompt_tokens)
        task_item = INDEX.get(text)
        if task_item is None:
            content, tcs = "mock: unknown prompt", None
        else:
            task, item = task_item
            corrupt = ARGS.corrupt_every and int(hashlib.md5(item["id"].encode()).hexdigest(), 16) % ARGS.corrupt_every == 0
            content, tcs = gold_response(task, item, corrupt)
        msg = {"role": "assistant", "content": content}
        if tcs:
            msg["tool_calls"] = tcs
        ct = max(1, len((content or "") + json.dumps(tcs or "")) // 4)
        self._send(200, {"id": "mock", "object": "chat.completion", "model": body["model"],
                         "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tcs else "stop"}],
                         "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": ct,
                                   "total_tokens": prompt_tokens + ct}})

    def _stream(self, body, prompt_tokens):
        n = int(body.get("max_tokens") or 16)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(obj):
            data = f"data: {obj if isinstance(obj, str) else json.dumps(obj)}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
        time.sleep(ARGS.ttft)
        for i in range(n):
            chunk({"choices": [{"index": 0, "delta": {"content": "tok "}, "finish_reason": None}]})
            time.sleep(ARGS.itl)
        chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
        if (body.get("stream_options") or {}).get("include_usage"):
            chunk({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                            "total_tokens": prompt_tokens + n}})
        chunk("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18999)
    ap.add_argument("--tool-mode", choices=["server", "xml"], default="server")
    ap.add_argument("--corrupt-every", type=int, default=0)
    ap.add_argument("--ttft", type=float, default=0.02)
    ap.add_argument("--itl", type=float, default=0.002)
    ARGS = ap.parse_args()
    build_index()
    print(f"mock server on 127.0.0.1:{ARGS.port} ({len(INDEX)} items indexed)", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), H)
    srv.handle_error = lambda *a: None  # quiet on client disconnects
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
