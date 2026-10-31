#!/usr/bin/env python
"""Run the fixed eval suite against an OpenAI-compatible Chat Completions endpoint.

Example:
  .venv-eval/bin/python eval/run_eval.py --base-url http://localhost:30000/v1 \
      --model SenseNova-Loop-9B-Preview --out eval/runs/bf16 --concurrency 16
Writes <out>/raw_<task>.jsonl (one line per item, full response + score) and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import requests  # noqa: E402

from common import EVAL_VERSION, chat, read_jsonl, wilson_ci, write_jsonl  # noqa: E402
from tasks import MAX_TOKENS, TASKS, agreement_key  # noqa: E402

DATA = HERE / "data"


def run_item(task, item, args, session):
    build, score = TASKS[task]
    messages, tools = build(item, DATA)
    resp = chat(session, args.base_url, args.model, messages, max_tokens=MAX_TOKENS[task], tools=tools,
                enable_thinking=args.enable_thinking, timeout=args.timeout)
    try:
        sc = score(item, resp) if resp["ok"] else {"pred": None, "correct": False}
    except Exception as e:  # noqa: BLE001
        sc = {"pred": None, "correct": False, "score_error": f"{type(e).__name__}: {e}"}
    sc["correct"] = bool(sc["correct"])
    return {
        "id": item["id"], "task": task,
        "prompt_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True)[:200000].encode()).hexdigest()[:16],
        "content": resp["content"], "reasoning_content": resp["reasoning_content"],
        "tool_calls": resp["tool_calls"], "finish_reason": resp["finish_reason"],
        "usage": resp["usage"], "completion_tokens": (resp["usage"] or {}).get("completion_tokens"),
        "latency_s": resp["latency_s"], "error": resp["error"],
        "agree_key": agreement_key(task, sc, resp), **sc,
    }


def summarize_task(rows):
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson_ci(k, n)
    toks = [r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]
    s = {"n": n, "correct": k, "accuracy": k / n if n else None, "wilson95": [lo, hi],
         "n_errors": sum(1 for r in rows if r["error"]),
         "n_truncated": sum(1 for r in rows if r["finish_reason"] == "length"),
         "mean_completion_tokens": sum(toks) / len(toks) if toks else None,
         "sum_completion_tokens": sum(toks)}
    if rows and "vqa_soft_acc" in rows[0]:
        s["vqa_soft_acc"] = sum(r.get("vqa_soft_acc", 0) for r in rows) / n
    if rows and rows[0]["task"] == "tools":
        s["names_correct"] = sum(1 for r in rows if r.get("names_correct")) / n
        s["parse_source_counts"] = {}
        for r in rows:
            src = r.get("parse_source", "none")
            s["parse_source_counts"][src] = s["parse_source_counts"].get(src, 0) + 1
        fmt = [r.get("format", {}) for r in rows if r.get("parse_source") != "none"]
        s["format_well_formed"] = (sum(1 for f in fmt if f.get("well_formed")) / len(fmt)) if fmt else None
        s["raw_tool_call_tags_in_content"] = sum(1 for r in rows if "<tool_call>" in (r["content"] or ""))
        by = {}
        for r in rows:
            c = r["id"].split("_")[1]
            by.setdefault(c, [0, 0])
            by[c][0] += r["correct"]
            by[c][1] += 1
        s["by_category"] = {c: f"{a}/{b}" for c, (a, b) in sorted(by.items())}
    if rows and rows[0]["task"] == "ifeval":
        flat = [x for r in rows for x in r.get("per_instruction", [])]
        s["instruction_level_acc"] = sum(flat) / len(flat) if flat else None
        s["n_instructions"] = len(flat)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://localhost:30000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N items per task")
    ap.add_argument("--enable-thinking", action="store_true", help="default OFF")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {args.api_key}"
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=max(16, args.concurrency * 2))
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    summary = {"eval_version": EVAL_VERSION, "model": args.model, "base_url": args.base_url,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "enable_thinking": args.enable_thinking,
               "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": MAX_TOKENS},
               "concurrency": args.concurrency, "limit": args.limit, "host": platform.node(),
               "data_sha256": {t: v["sha256"] for t, v in manifest["tasks"].items()}, "tasks": {}}
    try:
        summary["server_models"] = session.get(args.base_url.rstrip("/") + "/models", timeout=10).json()
    except Exception as e:  # noqa: BLE001
        summary["server_models"] = f"unavailable: {e}"

    t_all = time.time()
    for task in args.tasks.split(","):
        items = read_jsonl(DATA / f"{task}.jsonl")
        if args.limit:
            items = items[: args.limit]
        t0 = time.time()
        with ThreadPoolExecutor(args.concurrency) as ex:
            rows = list(ex.map(lambda it: run_item(task, it, args, session), items))
        write_jsonl(out / f"raw_{task}.jsonl", rows)
        s = summarize_task(rows)
        s["wall_s"] = round(time.time() - t0, 1)
        summary["tasks"][task] = s
        print(f"{task:10s} n={s['n']:3d} acc={s['accuracy']:.3f} "
              f"CI95=[{s['wilson95'][0]:.3f},{s['wilson95'][1]:.3f}] err={s['n_errors']} "
              f"trunc={s['n_truncated']} mean_tok={s['mean_completion_tokens']} {s['wall_s']}s", flush=True)
    summary["wall_s"] = round(time.time() - t_all, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
