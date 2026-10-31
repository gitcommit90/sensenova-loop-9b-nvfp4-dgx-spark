#!/usr/bin/env python
"""Streaming throughput/latency benchmark for an OpenAI-compatible Chat Completions server.

  .venv-eval/bin/python eval/perf_bench.py --base-url http://localhost:30000/v1 \
      --model SenseNova-Loop-9B-Preview --concurrency 1,2,4,8 --num-requests 16 \
      --input-tokens 1024 --output-tokens 256 --out eval/runs/perf_bf16.json

Prompt: deterministic pseudo-text (fixed seed) cut to exactly --input-tokens tokens with the model's
own tokenizer.json (the chat template adds a few more; actual prompt_tokens come from `usage`).
Each request gets a unique numeric prefix so prefix caching cannot serve it (disable with
--allow-prefix-cache). Requests use temperature 0, ignore_eos=true and min_tokens=max_tokens
(vLLM/SGLang extensions; disable with --no-ignore-eos) and thinking off.
Metrics per concurrency level: TTFT p50/p90/mean, per-request decode tok/s ((n_out-1)/(t_last-t_first)),
inter-token latency, E2E latency p50/p90, aggregate output tok/s (sum out tokens / level wall time),
request throughput, errors, actual generated lengths (from usage; falls back to chunk count).
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_TOK = HERE.parent / "upstream/SenseNova-Loop-9B-Preview/tokenizer.json"
WORDS = ("the of and to in is was for on that with as by at from his her an were are which this be has "
         "had it not or have one their they its first new after who two been more year also other some "
         "into all time only most over three city river mountain village market winter summer story "
         "engine signal harbor garden library window letter bridge forest island station machine "
         "children ancient quiet bright careful distant golden narrow sudden gentle measured").split()


def build_prompt_text(n_tokens: int, tokenizer_path: str | None):
    rng = random.Random(1234)
    sents = []
    for _ in range(n_tokens):  # plenty of text
        k = rng.randint(8, 18)
        s = " ".join(rng.choice(WORDS) for _ in range(k))
        sents.append(s[0].upper() + s[1:] + ".")
    text = " ".join(sents)
    if tokenizer_path and Path(tokenizer_path).exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))
        ids = tok.encode(text, add_special_tokens=False).ids[:n_tokens]
        text = tok.decode(ids)
        return text, len(tok.encode(text, add_special_tokens=False).ids), "tokenizer.json"
    words = text.split()[: int(n_tokens * 0.75)]
    return " ".join(words), None, "approx(0.75 words/token)"


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def one_request(args, session, text, idx):
    prefix = "" if args.allow_prefix_cache else f"[req {idx:06d}] "
    body = {"model": args.model, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prefix + text + "\n\nContinue this text."}],
            "max_tokens": args.output_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    if not args.no_ignore_eos:
        body.update({"ignore_eos": True, "min_tokens": args.output_tokens})
    t0 = time.perf_counter()
    t_first = t_last = None
    n_chunks = 0
    usage = None
    finish = None
    try:
        with session.post(args.base_url.rstrip("/") + "/chat/completions", json=body, stream=True,
                          timeout=args.timeout) as r:
            if r.status_code != 200:
                return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                j = json.loads(data)
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        n_chunks += 1
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    t_end = time.perf_counter()
    if t_first is None:
        return {"ok": False, "error": "no tokens received"}
    n_out = (usage or {}).get("completion_tokens") or n_chunks
    decode_t = (t_last - t_first) if t_last and t_last > t_first else None
    return {"ok": True, "ttft_s": t_first - t0, "e2e_s": t_end - t0, "output_tokens": n_out,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "n_chunks": n_chunks, "finish_reason": finish,
            "decode_tok_s": (n_out - 1) / decode_t if decode_t and n_out > 1 else None,
            "itl_s": decode_t / (n_out - 1) if decode_t and n_out > 1 else None,
            "usage_present": usage is not None}


def run_level(args, session, text, c, base_idx):
    n = args.num_requests if args.num_requests else max(8, 4 * c)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(c) as ex:
        res = list(ex.map(lambda i: one_request(args, session, text, base_idx + i), range(n)))
    wall = time.perf_counter() - t0
    ok = [r for r in res if r["ok"]]

    def stat(key):
        xs = [r[key] for r in ok if r.get(key) is not None]
        return {"p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "mean": statistics.fmean(xs) if xs else None}
    outs = [r["output_tokens"] for r in ok]
    return {"concurrency": c, "n_requests": n, "n_ok": len(ok), "n_errors": len(res) - len(ok),
            "errors": [r["error"] for r in res if not r["ok"]][:5], "wall_s": wall,
            "ttft_s": stat("ttft_s"), "e2e_s": stat("e2e_s"), "decode_tok_s_per_req": stat("decode_tok_s"),
            "itl_s": stat("itl_s"),
            "output_tokens_mean": statistics.fmean(outs) if outs else None,
            "output_tokens_min": min(outs) if outs else None, "output_tokens_max": max(outs) if outs else None,
            "prompt_tokens_mean": statistics.fmean([r["prompt_tokens"] for r in ok if r.get("prompt_tokens")] or [0]),
            "n_short_outputs": sum(1 for o in outs if o < args.output_tokens),
            "aggregate_output_tok_s": sum(outs) / wall if wall else None,
            "requests_per_s": len(ok) / wall if wall else None,
            "usage_reported": all(r.get("usage_present") for r in ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--num-requests", type=int, default=16, help="per level (0 => max(8, 4*c))")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--input-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOK))
    ap.add_argument("--allow-prefix-cache", action="store_true")
    ap.add_argument("--no-ignore-eos", action="store_true")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text, n_tok, how = build_prompt_text(args.input_tokens, args.tokenizer)
    levels = [int(x) for x in args.concurrency.split(",")]
    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=max(levels) * 2 + 4))
    for i in range(args.warmup):
        w = one_request(args, session, text, 900000 + i)
        if not w["ok"]:
            print("warmup error:", w["error"], file=sys.stderr)
    res = {"label": args.label, "model": args.model, "base_url": args.base_url, "host": platform.node(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "config": {**vars(args), "prompt_text_tokens": n_tok, "prompt_build": how}, "levels": []}
    base = 0
    for c in levels:
        lv = run_level(args, session, text, c, base)
        base += 10000
        res["levels"].append(lv)
        f = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"c={c:3d} ok={lv['n_ok']}/{lv['n_requests']} TTFT p50={f(lv['ttft_s']['p50'])}s "
              f"p90={f(lv['ttft_s']['p90'])}s decode/req p50={f(lv['decode_tok_s_per_req']['p50'])} tok/s "
              f"E2E p50={f(lv['e2e_s']['p50'])}s agg={f(lv['aggregate_output_tok_s'])} tok/s "
              f"out_len={lv['output_tokens_mean']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
