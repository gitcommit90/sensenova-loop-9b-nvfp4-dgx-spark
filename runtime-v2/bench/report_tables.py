"""Before/after markdown tables from runs/final_<label>/ (perf_matched.json, screen_{code,doc}.json, eval)."""
import json, sys
from pathlib import Path
labels = sys.argv[1:] or ["frozen_v1", "v2_fast", "v2_exact"]
R = Path(__file__).resolve().parents[1] / "runs"
def load(l, f):
    p = R / f"final_{l}" / f
    return json.load(open(p)) if p.exists() else None
out = []
out.append("### Matched workload (eval/perf_bench.py: ~1051 prompt tok, 256 out, unique prompts, 8 req/level, 2 warmup)\n")
out.append("| runtime | c | aggregate tok/s | per-req decode tok/s p50 | TTFT p50 / p90 (s) | ITL p50 / p90 (ms) | E2E p50 / p90 (s) | out tok min-max | errors |")
out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for l in labels:
    d = load(l, "perf_matched.json")
    if not d: continue
    for lv in d["levels"]:
        out.append(f"| {l} | {lv['concurrency']} | {lv['aggregate_output_tok_s']:.1f} | {lv['decode_tok_s_per_req']['p50']:.1f} | "
                   f"{lv['ttft_s']['p50']:.2f} / {lv['ttft_s']['p90']:.2f} | {lv['itl_s']['p50']*1e3:.0f} / {lv['itl_s']['p90']*1e3:.0f} | "
                   f"{lv['e2e_s']['p50']:.1f} / {lv['e2e_s']['p90']:.1f} | {lv['output_tokens_min']}-{lv['output_tokens_max']} | {lv['n_errors']} |")
for kind, desc in (("code", "short interactive coding/reasoning prompt (~90 tok in, 256 out)"), ("doc", "long document (~4k tok in, 256 out)")):
    out.append(f"\n### {kind}: {desc}\n")
    out.append("| runtime | c | n | aggregate tok/s | per-req decode p50 | TTFT p50 (ms) | ITL p50 / p95 (ms) | MTP accept len | errors |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for l in labels:
        d = load(l, f"screen_{kind}.json")
        if not d: continue
        for lv in d["levels"]:
            a = lv.get("spec_accept_len_req_mean")
            out.append(f"| {l} | {lv['concurrency']} | {lv['n_requests']} | {lv['aggregate_output_tok_s']:.1f} | {lv['decode_tok_s_per_req_p50']:.1f} | "
                       f"{lv['ttft_ms_p50']:.0f} | {lv['itl_ms_p50']:.0f} / {lv['itl_ms_p95']:.0f} | {a:.2f} | {lv['n_errors']} |" if a else
                       f"| {l} | {lv['concurrency']} | {lv['n_requests']} | {lv['aggregate_output_tok_s']:.1f} | {lv['decode_tok_s_per_req_p50']:.1f} | {lv['ttft_ms_p50']:.0f} | {lv['itl_ms_p50']:.0f} / {lv['itl_ms_p95']:.0f} | - | {lv['n_errors']} |")
out.append("\n### 300-item regression (thinking off, temp 0, eval concurrency 16)\n")
out.append("| runtime | ifeval | gsm8k | mmlu_pro | humaneval | tools | vqa | pooled | truncated | errors |")
out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for l in labels:
    s = load(l, "eval/summary.json")
    if not s: continue
    t = s["tasks"]; c = sum(round(v["accuracy"] * v["n"]) for v in t.values()); n = sum(v["n"] for v in t.values())
    tr = sum(v.get("n_truncated", v.get("truncated", 0)) or 0 for v in t.values()); er = sum(v.get("n_errors", 0) for v in t.values())
    out.append(f"| {l} | " + " | ".join(f"{t[k]['accuracy']:.3f}" for k in ("ifeval", "gsm8k", "mmlu_pro", "humaneval", "tools", "vqa")) + f" | {c}/{n} = {c/n:.3f} | {tr} | {er} |")
print("\n".join(out))
