#!/usr/bin/env python
"""Fast A/B screening benchmark for the SGLang server (OpenAI chat API, streaming).

  .venv-eval/bin/python runtime-v2/bench/screen.py --base-url http://127.0.0.1:30080 \
      --model sensenova-loop-9b-nvfp4 --concurrency 1,4,8 --num-requests 4 --output-tokens 256 \
      --prompt-kind synthetic --repeat 2 --out runtime-v2/runs/screen_baseline.json

Prompts follow eval/perf_bench.py: unique per-request prefix (plus a per-run tag, so no prefix-cache
hits even across runs), temperature 0, ignore_eos + min_tokens = max_tokens, thinking off.
--prompt-kind: synthetic (perf_bench text, default 1024 tok), code (fixed coding prompt), doc (~4k-token
document + summarization instruction). MTP acceptance: per-request `sglext.spec_tokens_details`
(requested via return_spec_tokens_details) and the server's cumulative /server_info avg_spec_accept_length.
"""
from __future__ import annotations

import argparse, json, statistics, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "harness"))
from perf_bench import DEFAULT_TOK, build_prompt_text, pct  # noqa: E402

CODE_PROMPT = ("Implement an LRU cache in Python with O(1) get and put (no functools), explain the design "
               "and its complexity, then write a thorough pytest test suite covering eviction order, updates "
               "of existing keys, capacity 1 and invalid capacity. Finally discuss thread-safety trade-offs.")


def make_prompt(args):
    if args.prompt_kind == "code":
        return CODE_PROMPT, ""
    n = args.input_tokens or (4096 if args.prompt_kind == "doc" else 1024)
    text, _, _ = build_prompt_text(n, args.tokenizer)
    if args.prompt_kind == "doc":
        return "Document:\n" + text, "\n\nSummarize the document above in five bullet points, then list its recurring themes."
    return text, "\n\nContinue this text."


def one_request(args, s, body_text, suffix, tag):
    body = {"model": args.model, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[req {tag}] " + body_text + suffix}],
            "max_tokens": args.output_tokens, "temperature": 0.0, "ignore_eos": True,
            "min_tokens": args.output_tokens, "return_spec_tokens_details": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    times, usage, spec = [], None, None
    try:
        with s.post(args.api + "/chat/completions", json=body, stream=True, timeout=args.timeout) as r:
            if r.status_code != 200:
                return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
            for line in r.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                j = json.loads(data)
                usage = j.get("usage") or usage
                spec = ((j.get("sglext") or {}).get("spec_tokens_details")) or spec
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        times.append(time.perf_counter())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not times:
        return {"ok": False, "error": "no tokens received"}
    n_out = (usage or {}).get("completion_tokens") or len(times)
    dt = times[-1] - times[0]
    if isinstance(spec, list):
        spec = spec[0] if spec else None
    return {"ok": True, "ttft_s": times[0] - t0, "e2e_s": time.perf_counter() - t0, "out": n_out,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "decode_tok_s": (n_out - 1) / dt if dt > 0 and n_out > 1 else None,
            "itl_s": dt / (n_out - 1) if dt > 0 and n_out > 1 else None,
            "gaps": [b - a for a, b in zip(times, times[1:])],
            "accept_len": (spec or {}).get("spec_accept_length")}


def server_accept(args):
    for path in ("/server_info", "/get_server_info"):
        try:
            j = requests.get(args.root + path, timeout=10).json()
            for st in j.get("internal_states") or []:
                if "avg_spec_accept_length" in st:
                    return st["avg_spec_accept_length"]
            return None
        except Exception:  # noqa: BLE001
            continue
    return None


def run_level(args, s, prompt, c, rep):
    n = args.num_requests or max(4, 2 * c)
    tags = [f"{args.run_tag}-c{c}-r{rep}-{i:05d}" for i in range(n)]
    acc0 = server_accept(args)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(c) as ex:
        res = list(ex.map(lambda t: one_request(args, s, prompt[0], prompt[1], t), tags))
    wall = time.perf_counter() - t0
    ok = [r for r in res if r["ok"]]
    col = lambda k: [r[k] for r in ok if r.get(k) is not None]  # noqa: E731
    gaps = [g for r in ok for g in r["gaps"]]
    ms = lambda x: None if x is None else 1000 * x  # noqa: E731
    acc = col("accept_len")
    return {"concurrency": c, "repeat": rep, "n_requests": n, "n_ok": len(ok), "n_errors": n - len(ok),
            "errors": [r["error"] for r in res if not r["ok"]][:5], "wall_s": wall,
            "aggregate_output_tok_s": sum(col("out")) / wall if wall else None,
            "decode_tok_s_per_req_p50": pct(col("decode_tok_s"), 0.5),
            "ttft_ms_p50": ms(pct(col("ttft_s"), 0.5)),
            "itl_ms_p50": ms(pct(col("itl_s"), 0.5)), "itl_ms_p95": ms(pct(col("itl_s"), 0.95)),
            "chunk_gap_ms_p50": ms(pct(gaps, 0.5)), "chunk_gap_ms_p95": ms(pct(gaps, 0.95)),
            "out_tokens_mean": statistics.fmean(col("out")) if ok else None,
            "prompt_tokens_mean": statistics.fmean(col("prompt_tokens")) if col("prompt_tokens") else None,
            "spec_accept_len_req_mean": statistics.fmean(acc) if acc else None,
            "server_avg_accept_len_before": acc0, "server_avg_accept_len_after": server_accept(args)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="server root or .../v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--num-requests", type=int, default=0, help="per level; 0 => max(4, 2*c)")
    ap.add_argument("--input-tokens", type=int, default=0, help="0 => 1024 (synthetic) / 4096 (doc)")
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--prompt-kind", choices=["synthetic", "code", "doc"], default="synthetic")
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOK))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.root = args.base_url.rstrip("/").removesuffix("/v1")
    args.api = args.root + "/v1"
    args.run_tag = uuid.uuid4().hex[:8]
    prompt = make_prompt(args)
    levels = [int(x) for x in args.concurrency.split(",")]
    s = requests.Session()
    s.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=max(levels) * 2 + 4))
    for i in range(args.warmup):
        w = one_request(args, s, prompt[0], prompt[1], f"{args.run_tag}-warm{i}")
        if not w["ok"]:
            print("warmup error:", w["error"], file=sys.stderr)
    out = {"label": args.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "config": {k: v for k, v in vars(args).items()}, "levels": []}
    f = lambda x: "  n/a" if x is None else f"{x:7.2f}"  # noqa: E731
    for c in levels:
        for rep in range(args.repeat):
            lv = run_level(args, s, prompt, c, rep)
            out["levels"].append(lv)
            print(f"c={c:2d} r={rep} ok={lv['n_ok']}/{lv['n_requests']} agg={f(lv['aggregate_output_tok_s'])} tok/s "
                  f"dec/req={f(lv['decode_tok_s_per_req_p50'])} TTFT={f(lv['ttft_ms_p50'])}ms "
                  f"ITL p50/p95={f(lv['itl_ms_p50'])}/{f(lv['itl_ms_p95'])}ms "
                  f"accept={f(lv['spec_accept_len_req_mean'])} srv={f(lv['server_avg_accept_len_after'])}",
                  flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
